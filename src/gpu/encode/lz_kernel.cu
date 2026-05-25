// ── StreamLZ GPU LZ encode kernel ───────────────────────────────
// Warp-parallel LZ77 match-finding / parsing kernel. Serves codec
// levels L1–L5: the driver maps each level to a hash-bit width and,
// at the higher levels, switches on the serial chain parser.
//
// One warp per chunk, one chunk per CUDA block. Match-finder hash
// tables live in global memory (the driver's useGlobalHash policy
// returns true for every level); the extern __shared__ shared_ht
// fallback is currently unreachable in production — see the comment
// at the shared_ht declaration below.
//
// Two-pass block design: the parser scans block 1 [0..64KB) and
// block 2 [64KB..src_size) separately so that literal runs and
// match extensions never cross the 64KB boundary. This matches the
// CPU encoder's architecture exactly.
//
// Two parser modes:
//   use_chain=0 (default): warp-parallel greedy scan (scanBlock)
//   use_chain=1:           serial chain-hash lazy parser (scanBlockChain)
//
// This is a single translation unit: the cohesive sections live in
// the lz_*.cuh headers below and are #include-d here so nvcc still
// compiles ONE lz_kernel.cu into ONE lz_kernel.ptx/.cubin.
//
// Built to lz_kernel.ptx/.cubin by tools/build_gpu_enc.bat
// (nvcc -arch=sm_89 -O3).

#include "lz_format.cuh"
#include "lz_token_emit.cuh"
#include "lz_greedy_parser.cuh"
#include "lz_chain_parser.cuh"

// ── Output framing constants ────────────────────────────────────
// OFF32_COUNT_FIELD_BITS / OFF32_COUNT_PACK_MAX come from
// ../common/gpu_wire_format.cuh (transitively via lz_format.cuh).
static constexpr uint32_t STREAM_HEADER_BYTES = 3;  // 3-byte big-endian sub-stream count header
static constexpr uint32_t OFF16_HEADER_BYTES  = 2;  // 2-byte off16 count header

// ── slzLzEncodeKernel ───────────────────────────────────────────
// LZ encode kernel entry point. One block per chunk, one warp (32
// lanes) per block; __launch_bounds__(32, 1) pins that 1-warp design.
//
// Parameters:
//   input        -- all chunk source bytes, concatenated
//   output       -- per-chunk output scratch (also used as working
//                    buffer for the sub-streams before framing)
//   descs        -- per-chunk CompressChunkDesc (src/dst offsets, is_first)
//   global_hash  -- per-chunk hash-table storage; greedy mode uses
//                    hash_size u32 per chunk, chain mode uses
//                    2*hash_size + NEXT_HASH_SIZE/2 u32 per chunk
//   comp_sizes   -- out: per-chunk compressed byte count
//   total_chunks -- number of chunks (blockIdx.x >= total_chunks: no-op)
//   hash_bits    -- log2 of the hash-table size for this level
//   use_chain    -- 0 = greedy parser, 1 = serial chain parser
//   l4_features  -- non-zero enables match-range rehash in the greedy parser
//
// Output layout written by lane 0 (consumed by driver.zig and the
// decode kernels — this is a format contract):
//   [INITIAL_LITERAL_COPY_BYTES verbatim bytes]   (only if desc.is_first)
//   [3-byte BE lit_count][literals]
//   [3-byte BE token_count][tokens]
//   [2-byte LE cmd_stream2_offset]  (only if src_size > LZ_BLOCK_SIZE)
//   [2-byte LE off16_count][off16 stream]
//   [3-byte packed off32 block-1/2 counts]
//   [2-byte LE block-1 off32 ext]   (only if count >= OFF32_COUNT_PACK_MAX)
//   [2-byte LE block-2 off32 ext]   (only if count >= OFF32_COUNT_PACK_MAX)
//   [off32 stream][length stream]
extern "C" __global__ void __launch_bounds__(32, 1) slzLzEncodeKernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const CompressChunkDesc* __restrict__ descs,
    uint32_t* __restrict__ global_hash,
    uint32_t* __restrict__ comp_sizes,
    uint32_t total_chunks,
    uint32_t hash_bits,
    uint32_t use_chain,
    uint32_t l4_features
) {
    const uint32_t chunk_id = blockIdx.x;
    const uint32_t lane = threadIdx.x & LANE_MASK;
    if (chunk_id >= total_chunks) return;

    const uint32_t hash_size = 1u << hash_bits;
    const uint32_t hash_mask = hash_size - 1;

    const CompressChunkDesc& desc = descs[chunk_id];
    const uint8_t* src = input + desc.src_offset;
    const uint32_t src_size = desc.src_size;

    uint8_t* dst = output + desc.dst_offset;
    const uint32_t lit_data_start = (desc.is_first ? INITIAL_LITERAL_COPY_BYTES : 0) + STREAM_HEADER_BYTES;

    // Sub-stream working buffers carved out of dst. The /4 and /2
    // divisors size each region for its worst case relative to src_size.
    OutputStreams streams;
    streams.lit_buf   = dst + lit_data_start;
    streams.token_buf = dst + src_size;
    streams.off16_buf = streams.token_buf + (src_size / 4);
    streams.off32_buf = streams.off16_buf + (src_size / 2);
    streams.len_buf   = streams.off32_buf + (src_size / 4);
    streams.lit_count = 0;
    streams.token_count = 0;
    streams.off16_count = 0;
    streams.off32_pos = 0;
    streams.off32_count = 0;
    streams.length_count = 0;

    uint32_t off32_count_block1 = 0, off32_count_block2 = 0;
    uint32_t cmd_stream2_offset = 0;
    uint32_t anchor = desc.is_first ? INITIAL_LITERAL_COPY_BYTES : 0;
    int32_t recent_offset = -8;

    if (use_chain) {
        // ── Chain parser mode ────────────────────────────────────
        // Three hash tables per block laid out contiguously in global_hash:
        //   first_hash: hash_size u32
        //   long_hash:  hash_size u32
        //   next_hash:  NEXT_HASH_SIZE u16 (= NEXT_HASH_SIZE/2 u32 words)
        uint32_t table_stride = hash_size + hash_size + (NEXT_HASH_SIZE / 2);
        uint32_t* base = global_hash + (uint64_t)chunk_id * table_stride;
        uint32_t* chain_first_hash = base;
        uint32_t* chain_long_hash  = base + hash_size;
        uint16_t* chain_next_hash  = (uint16_t*)(base + hash_size + hash_size);

        // Initialize all three tables to 0
        uint32_t total_words = hash_size + hash_size + (NEXT_HASH_SIZE / 2);
        for (uint32_t i = lane; i < total_words; i += WARP_SIZE)
            base[i] = 0;
        __syncwarp();

        // ── Block 1 pass ────
        {
            uint32_t block1_end = (src_size < LZ_BLOCK_SIZE) ? src_size : LZ_BLOCK_SIZE;
            scanBlockChain(src, src_size,
                           chain_first_hash, chain_long_hash, chain_next_hash,
                           hash_bits, hash_mask,
                           streams,
                           anchor, recent_offset,
                           anchor, block1_end, /*block2_start=*/0);
            off32_count_block1 = streams.off32_count;
            streams.off32_count = 0;
        }

        // ── Block 2 pass ────
        if (src_size > LZ_BLOCK_SIZE) {
            cmd_stream2_offset = streams.token_count;
            uint32_t block2_start_pos = (anchor > LZ_BLOCK_SIZE) ? anchor : LZ_BLOCK_SIZE;
            scanBlockChain(src, src_size,
                           chain_first_hash, chain_long_hash, chain_next_hash,
                           hash_bits, hash_mask,
                           streams,
                           anchor, recent_offset,
                           block2_start_pos, src_size, /*block2_start=*/LZ_BLOCK_SIZE);
            off32_count_block2 = streams.off32_count;
        }
    } else {
        // ── Greedy parser mode (default) ─────────────────────────
        // useGlobalHash() returns true for every level, so the driver
        // always passes a non-null global_hash.
        uint32_t* ht = global_hash + (uint64_t)chunk_id * hash_size;

        for (uint32_t i = lane; i < hash_size; i += WARP_SIZE)
            ht[i] = HASH_EMPTY;
        __syncwarp();

        // ── Block 1 pass ────
        {
            uint32_t block1_end = (src_size < LZ_BLOCK_SIZE) ? src_size : LZ_BLOCK_SIZE;
            scanBlock(src, src_size, ht, hash_bits, hash_mask,
                      streams,
                      anchor, recent_offset,
                      anchor, block1_end, /*block2_start=*/0, l4_features);
            off32_count_block1 = streams.off32_count;
            streams.off32_count = 0;
        }

        // ── Block 2 pass ────
        if (src_size > LZ_BLOCK_SIZE) {
            cmd_stream2_offset = streams.token_count;
            uint32_t block2_start_pos = (anchor > LZ_BLOCK_SIZE) ? anchor : LZ_BLOCK_SIZE;
            scanBlock(src, src_size, ht, hash_bits, hash_mask,
                      streams,
                      anchor, recent_offset,
                      block2_start_pos, src_size, /*block2_start=*/LZ_BLOCK_SIZE, l4_features);
            off32_count_block2 = streams.off32_count;
        }
    }

    __syncwarp();

    if (lane == 0) {
        uint32_t out_pos = 0;
        if (desc.is_first) {
            memcpy(dst, src, INITIAL_LITERAL_COPY_BYTES);
            out_pos = INITIAL_LITERAL_COPY_BYTES;
        }

        dst[out_pos] = (uint8_t)((streams.lit_count >> 16) & 0xFF);
        dst[out_pos + 1] = (uint8_t)((streams.lit_count >> 8) & 0xFF);
        dst[out_pos + 2] = (uint8_t)(streams.lit_count & 0xFF);
        out_pos += STREAM_HEADER_BYTES + streams.lit_count;

        dst[out_pos] = (uint8_t)((streams.token_count >> 16) & 0xFF);
        dst[out_pos + 1] = (uint8_t)((streams.token_count >> 8) & 0xFF);
        dst[out_pos + 2] = (uint8_t)(streams.token_count & 0xFF);
        out_pos += STREAM_HEADER_BYTES;
        memcpy(dst + out_pos, streams.token_buf, streams.token_count);
        out_pos += streams.token_count;

        if (src_size > LZ_BLOCK_SIZE) {
            uint16_t cs2o = (cmd_stream2_offset > 0)
                ? (uint16_t)cmd_stream2_offset : (uint16_t)streams.token_count;
            storeU16LE(dst + out_pos, cs2o);
            out_pos += 2;
        }

        dst[out_pos] = (uint8_t)(streams.off16_count & 0xFF);
        dst[out_pos + 1] = (uint8_t)((streams.off16_count >> 8) & 0xFF);
        out_pos += OFF16_HEADER_BYTES;
        memcpy(dst + out_pos, streams.off16_buf, streams.off16_count * 2);
        out_pos += streams.off16_count * 2;

        uint32_t off32_count1_clamped = (off32_count_block1 < OFF32_COUNT_PACK_MAX)
            ? off32_count_block1 : OFF32_COUNT_PACK_MAX;
        uint32_t off32_count2_clamped = (off32_count_block2 < OFF32_COUNT_PACK_MAX)
            ? off32_count_block2 : OFF32_COUNT_PACK_MAX;
        uint32_t packed_off32_counts = (off32_count1_clamped << OFF32_COUNT_FIELD_BITS)
                                     | off32_count2_clamped;
        dst[out_pos++] = (uint8_t)(packed_off32_counts & 0xFF);
        dst[out_pos++] = (uint8_t)((packed_off32_counts >> 8) & 0xFF);
        dst[out_pos++] = (uint8_t)((packed_off32_counts >> 16) & 0xFF);
        if (off32_count_block1 >= OFF32_COUNT_PACK_MAX) {
            storeU16LE(dst + out_pos, (uint16_t)off32_count_block1);
            out_pos += 2;
        }
        if (off32_count_block2 >= OFF32_COUNT_PACK_MAX) {
            storeU16LE(dst + out_pos, (uint16_t)off32_count_block2);
            out_pos += 2;
        }
        memcpy(dst + out_pos, streams.off32_buf, streams.off32_pos);
        out_pos += streams.off32_pos;

        memcpy(dst + out_pos, streams.len_buf, streams.length_count);
        out_pos += streams.length_count;

        comp_sizes[chunk_id] = out_pos;
    }
}
