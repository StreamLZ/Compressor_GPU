// ── v4 #15: K=4 pipelined raw-mode sub-chunk decoder ────────────
// Parser warp (threadIdx.y==0) runs the PP scan and stages parsed
// batch data to a shared-memory double buffer. A 3-warp copier team
// (threadIdx.y 1-3, 96 lanes) runs the flat literal + flat
// independent-match passes, then warp 1 alone runs the dependent
// matches. Overlap: the parser fills batch N+1 while the team
// executes batch N — the parser's global reads hide under the team's
// global writes (long_scoreboard 22.9 → 8.9 measured at K=2), and
// the tripled copy width raised SM throughput further (K=4 vs K=2:
// enwik8 L1 1.90 → 1.77 ms, enwik9 L1 17.15 → 16.22; silesia pays
// ~4% back — binary corpora are parser-bound and blocks/SM halve).
//
// The parser can run ahead because its state advancement (cmd_pos /
// lit_pos / off16_pos / dst_pos / recent_offset) comes entirely from
// the warp scans, never from the copies.
//
// Synchronization: ONE __syncthreads per batch for the slot handoff;
// the copier team orders its internal phases with hardware named
// barrier 1 (__barrier_sync_count — NOT inline-asm bar.sync, which
// acts as an optimizer wall and measured +0.8 ms), so the parser
// never waits mid-batch. Intra-block spin-waits are NOT safe on
// NVIDIA hardware (no warp-scheduler fairness; instant deadlock
// measured), and a cuda::pipeline mbarrier ring measured SLOWER than
// __syncthreads pacing (see FAILED_EXPERIMENTS.md).
//
// Serial long tokens (TOKEN_LONG_LITERAL / TOKEN_LONG_NEAR) drain
// the pipeline first, then all warps execute the long token's
// copies cooperatively, then the pipeline re-primes. One serial
// iteration per long token, same as the non-pipelined kernel's
// PP-prefix truncation contract (~5-10% of iterations per the v4 #3
// measurement, so the drain cost is bounded).
//
// Lane contract notes:
//  - fillBatch runs on the parser warp only; all its __ballot_sync /
//    __shfl_sync / warpScanU32 calls are warp-local and converged.
//  - dep_mask is the ballot of dependent-match lanes UNMASKED:
//    invalid lanes (>= batch_size) carry my_match_len == 0 and are
//    excluded by the ballot itself. (An explicit `& ((1u <<
//    batch_size) - 1u)` guard is UB at batch_size == 32 — on sm_89
//    the wrapped shift made the mask 0 and silently dropped every
//    dependent match of full batches; cost a debugging session.)
#pragma once

#include "lz_decode_core.cuh"

// Double buffer: the parser runs at most one batch ahead.
// (A cuda::pipeline mbarrier ring was measured 2026-06-11 and LOST:
// S=2 2.18 ms / S=4 2.64 ms vs 1.90 ms for __syncthreads pacing on
// enwik8 L1 — four mbarrier ops per ~150 ns batch cost more than one
// hardware BAR.SYNC on a small block. See FAILED_EXPERIMENTS.md.)
#define SLZ_PIPE_STAGES 2

// v4 #15 escalation 2: K=4 warps per block — 1 parser + a 3-warp
// copier team (96 copy lanes vs the original 32). The copier team
// synchronizes internally with a NAMED barrier (bar.sync 1, 96) so
// the parser never participates in mid-batch copier syncs and keeps
// parsing batch N+1. Block-level handoff stays one __syncthreads per
// batch. 128-thread blocks: 12 blocks/SM keeps the SM's 1536-thread
// budget exactly (parser count per SM halves vs K=2; the bet is that
// tripled copy width on the flat passes more than compensates).
#define SLZ_PIPE_WARPS 4
#define SLZ_PIPE_BLOCK_THREADS (SLZ_PIPE_WARPS * WARP_SIZE)
#define SLZ_PIPE_TEAM_THREADS ((SLZ_PIPE_WARPS - 1) * WARP_SIZE)
#define SLZ_PIPE_MIN_BLOCKS_PER_SM 12

// Copier-team barrier: hardware named barrier 1 (barrier 0 belongs
// to __syncthreads), counted for the 3 copier warps only. bar.sync
// orders the participants' memory accesses like __syncthreads does.
__device__ __forceinline__ void pipeTeamBar() {
    __barrier_sync_count(1, SLZ_PIPE_TEAM_THREADS);
}

struct PipeBatch {
    uint32_t batch_size; // tokens staged (>= 1)
    uint32_t total_lit;
    uint32_t total_im;
    uint32_t dst_pos;    // batch's absolute dst base
    uint32_t lit_pos;    // batch's absolute lit base
    uint32_t dep_mask;   // ballot of dependent-match lanes
    uint32_t lit_prefix[WARP_SIZE];  // exclusive lit-byte prefix
    uint32_t dst_adj[WARP_SIZE];     // dst_local - lit_local
    uint32_t im_prefix[WARP_SIZE];   // exclusive indep-match-byte prefix
    uint32_t im_dst_adj[WARP_SIZE];  // copy_dst - im_local
    uint32_t im_src_adj[WARP_SIZE];  // match_src - im_local
    uint32_t match_len[WARP_SIZE];
    int32_t  match_offset[WARP_SIZE];
};

// ── Parser: fill one PP batch into `s`, advance parser state ────
// Returns true when a batch (>= 1 token) was staged; false when the
// window starts with a long token (caller drains + runs the serial
// handler) or cmd is exhausted (caller checks cmd_pos).
// Runs on the parser warp only.
template <bool OFF16_SPLIT>
__device__ bool fillBatch(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ off16_hi, const uint8_t* __restrict__ off16_lo,
    uint32_t& cmd_pos, uint32_t& lit_pos, uint32_t& off16_pos,
    uint32_t& dst_pos, int32_t& recent_offset,
    int lane, PipeBatch* s
) {
    uint32_t remaining = cmd_size - cmd_pos;
    uint32_t batch_size = remaining < WARP_SIZE ? remaining : WARP_SIZE;
    uint8_t my_cmd = ((uint32_t)lane < batch_size) ? cmd[cmd_pos + lane] : 0;
    bool my_is_long = ((uint32_t)lane < batch_size) && (my_cmd < TOKEN_SHORT_MIN);
    uint32_t any_long = __ballot_sync(FULL_WARP_MASK, my_is_long);
    if (any_long != 0)
        batch_size = (uint32_t)(__ffs(any_long) - 1);
    if (batch_size == 0)
        return false; // long token at window start — serial handler's job

    // ── PP scan (identical to decodeSubChunkRawMode's fast path) ──
    bool my_valid = (uint32_t)lane < batch_size;
    uint32_t my_lit_len   = my_valid ? (uint32_t)(my_cmd & TOKEN_LIT_MASK) : 0u;
    uint32_t my_match_len = my_valid ? (uint32_t)((my_cmd >> TOKEN_MATCH_SHIFT) & TOKEN_MATCH_MASK) : 0u;
    uint32_t my_use_recent = my_valid ? (uint32_t)((my_cmd >> TOKEN_USE_RECENT_SHIFT) & TOKEN_USE_RECENT_MASK) : 0u;
    uint32_t my_consumes_off16 = (my_valid && !my_use_recent) ? 1u : 0u;

    uint32_t my_off16_local, total_off16_used;
    warpScanU32(my_consumes_off16, my_off16_local, total_off16_used);

    int32_t my_match_offset = recent_offset;
    if (my_consumes_off16) {
        uint32_t entry_idx = off16_pos + my_off16_local;
        if (entry_idx < off16_count) {
            uint16_t v;
            if constexpr (OFF16_SPLIT) {
                v = (uint16_t)off16_lo[entry_idx] | ((uint16_t)off16_hi[entry_idx] << 8);
            } else {
                v = readU16LE(off16_raw + entry_idx * OFF16_ENTRY_BYTES);
            }
            my_match_offset = -(int32_t)v;
        }
    }

    uint32_t fresh_mask = __ballot_sync(FULL_WARP_MASK, my_consumes_off16 != 0);
    static_assert(WARP_SIZE == 32,
                  "(2u << lane) - 1u inclusive-prefix idiom assumes WARP_SIZE == 32");
    uint32_t my_prefix = fresh_mask & ((2u << lane) - 1u);
    int src_lane = (my_prefix != 0) ? lastBitSet(my_prefix) : 0;
    int32_t shuffled_off = __shfl_sync(FULL_WARP_MASK, my_match_offset, src_lane);
    if (my_use_recent && my_prefix != 0)
        my_match_offset = shuffled_off;

    uint32_t my_total = my_lit_len + my_match_len;
    uint32_t my_dst_local, total_dst;
    warpScanU32(my_total, my_dst_local, total_dst);
    uint32_t my_lit_local, total_lit;
    warpScanU32(my_lit_len, my_lit_local, total_lit);

    // Independent-match classification (v4 #2).
    uint32_t my_copy_dst = dst_pos + my_dst_local + my_lit_len;
    int32_t  my_src      = (int32_t)my_copy_dst + my_match_offset;
    bool my_is_indep = (my_match_len > 0) &&
                       (my_src + (int32_t)my_match_len <= (int32_t)dst_pos);
    uint32_t my_im_len = my_is_indep ? my_match_len : 0u;
    uint32_t my_im_local, total_im;
    warpScanU32(my_im_len, my_im_local, total_im);

    // ── Stage to the shared slot ─────────────────────────────────
    s->lit_prefix[lane]   = my_lit_local;
    s->dst_adj[lane]      = my_dst_local - my_lit_local;
    s->im_prefix[lane]    = my_im_local;
    s->im_dst_adj[lane]   = my_copy_dst - my_im_local;
    s->im_src_adj[lane]   = (uint32_t)my_src - my_im_local;
    s->match_len[lane]    = my_match_len;
    s->match_offset[lane] = my_match_offset;
    uint32_t dep_ballot = __ballot_sync(FULL_WARP_MASK, my_match_len > 0 && !my_is_indep);
    if (lane == 0) {
        s->batch_size = batch_size;
        s->total_lit  = total_lit;
        s->total_im   = total_im;
        s->dst_pos    = dst_pos;
        s->lit_pos    = lit_pos;
        s->dep_mask   = dep_ballot; // see header note: NO (1<<bs)-1 mask
    }

    // ── Advance parser state (scan results only, no copies) ─────
    cmd_pos   += batch_size;
    off16_pos += total_off16_used;
    dst_pos   += total_dst;
    lit_pos   += total_lit;
    if (fresh_mask != 0) {
        int last_fresh = lastBitSet(fresh_mask);
        recent_offset = __shfl_sync(FULL_WARP_MASK, my_match_offset, last_fresh);
    }
    return true;
}

// ── Copier team: execute one staged batch's copies (3 warps) ─────
// All reads come from the PipeBatch, all writes go to dst.
//
// Phase 1 (96 lanes): the flat-literal and flat-independent-match
// passes merge into ONE flat work pool — they are hazard-free
// against each other (im reads only pre-batch-final bytes by the
// v4 #2 classification, and their write ranges are disjoint by
// construction), so a single index space [0, total_lit + total_im)
// feeds all team lanes.
// Phase 2 (warp y==1 only, after a team barrier): dependent matches
// in token order — they may read phase-1 output, so they wait for
// it; warps 2-3 park at the closing team barrier meanwhile. The
// parser is NOT a participant in either team barrier and keeps
// parsing batch N+1 throughout.
__device__ void executeBatchTeam(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ lit,
    int lane,           // lane within own warp
    uint32_t team_lane, // 0..95 across the 3 copier warps
    uint32_t warp_y,    // threadIdx.y (1..3)
    const PipeBatch* s
) {
    uint32_t bs = s->batch_size;
    uint32_t batch_dst = s->dst_pos;
    uint32_t batch_lit = s->lit_pos;

    // Phase 1: flat literal pass then flat independent-match pass,
    // all team lanes. No barrier between them — they are hazard-free
    // against each other (im reads only pre-batch-final bytes by the
    // v4 #2 classification; write ranges disjoint by construction).
    // Two tight loops measured decisively faster than one merged
    // branchy pool (1.90 vs 3.00 ms on enwik8 L1 at K=2).
    for (uint32_t i = team_lane; i < s->total_lit; i += SLZ_PIPE_TEAM_THREADS) {
        uint32_t k = 0;
        #pragma unroll
        for (uint32_t step = 16; step >= 1; step >>= 1) {
            uint32_t cand = k + step;
            if (cand < bs && s->lit_prefix[cand] <= i) k = cand;
        }
        dst[batch_dst + s->dst_adj[k] + i] = lit[batch_lit + i];
    }
    for (uint32_t i = team_lane; i < s->total_im; i += SLZ_PIPE_TEAM_THREADS) {
        uint32_t k = 0;
        #pragma unroll
        for (uint32_t step = 16; step >= 1; step >>= 1) {
            uint32_t cand = k + step;
            if (cand < bs && s->im_prefix[cand] <= i) k = cand;
        }
        dst[s->im_dst_adj[k] + i] = dst[s->im_src_adj[k] + i];
    }
    pipeTeamBar(); // phase-1 writes visible to the dep warp

    // Phase 2: dependent matches in token order, warp 1 only.
    // copy_dst reconstructs as im_dst_adj[k] + im_prefix[k] =
    // (my_copy_dst - my_im_local) + my_im_local = my_copy_dst —
    // valid for every lane because the same my_im_local cancels.
    if (warp_y == 1) {
        uint32_t dep_mask = s->dep_mask;
        while (dep_mask != 0) {
            uint32_t k = (uint32_t)(__ffs(dep_mask) - 1);
            dep_mask &= dep_mask - 1;
            uint32_t k_match_len = s->match_len[k];
            int32_t  k_match_off = s->match_offset[k];
            uint32_t copy_dst = s->im_dst_adj[k] + s->im_prefix[k];
            uint32_t match_src = (uint32_t)((int32_t)copy_dst + k_match_off);
            warpMatchCopy(dst, copy_dst, match_src, k_match_len, -k_match_off, lane);
            __syncwarp();
        }
    }
    pipeTeamBar(); // dep writes done before the team touches the next slot
}

// ── Main pipelined decode loop ───────────────────────────────────
// Two-flag double buffer, ONE __syncthreads per batch: the loop
// condition reads s_have[cur] after the barrier that ordered the
// parser's pre-barrier write of that slot's flag.
//   prime:   parser fills slot 0          (team idle)
//   steady:  parser fills slot 1-cur, team executes slot cur
//   drain:   parser produced nothing — team executed the last slot
//            in the same iteration, both fall through
//   serial:  if cmd remains (long token), all warps execute it
//            cooperatively, then re-prime
template <bool OFF16_SPLIT>
__device__ __noinline__ void decodeSubChunkRawModePipelined(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ lit, uint32_t lit_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ off16_hi, const uint8_t* __restrict__ off16_lo,
    const uint8_t* __restrict__ length_stream, uint32_t length_remaining,
    uint8_t* __restrict__ dst, uint32_t dst_size,
    uint32_t initial_copy,
    uint32_t dst_offset,
    PipeBatch* s_batch // [SLZ_PIPE_STAGES], caller-allocated shared
) {
    const int lane = threadIdx.x & LANE_MASK;
    const uint32_t warp_y = threadIdx.y;
    const bool is_parser = (warp_y == 0);
    const uint32_t team_lane = (warp_y - 1) * WARP_SIZE + lane; // copier warps only
    (void)dst_size;

    uint32_t cmd_pos = 0, lit_pos = 0, off16_pos = 0;
    uint32_t dst_pos = dst_offset + initial_copy;
    int32_t recent_offset = INITIAL_RECENT_OFFSET;
    uint32_t length_offset = 0;

    __shared__ uint32_t s_have[SLZ_PIPE_STAGES]; // per-slot staged flags
    __shared__ uint32_t s_state[6];   // parser → team state broadcast
    __shared__ uint32_t s_long[5];    // serial-token broadcast

    while (true) {
        // ── Prime: parser stages the first batch of this run ─────
        if (is_parser) {
            bool got = (cmd_pos < cmd_size) && fillBatch<OFF16_SPLIT>(
                cmd, cmd_size, off16_raw, off16_count, off16_hi, off16_lo,
                cmd_pos, lit_pos, off16_pos, dst_pos, recent_offset,
                lane, &s_batch[0]);
            if (lane == 0) s_have[0] = got ? 1 : 0;
        }
        __syncthreads();

        // ── Steady state: overlap fill(N+1) with execute(N) ─────
        uint32_t cur = 0;
        while (s_have[cur]) {
            if (is_parser) {
                bool got = (cmd_pos < cmd_size) && fillBatch<OFF16_SPLIT>(
                    cmd, cmd_size, off16_raw, off16_count, off16_hi, off16_lo,
                    cmd_pos, lit_pos, off16_pos, dst_pos, recent_offset,
                    lane, &s_batch[1 - cur]);
                if (lane == 0) s_have[1 - cur] = got ? 1 : 0;
            } else {
                executeBatchTeam(dst, lit, lane, team_lane, warp_y, &s_batch[cur]);
            }
            __syncthreads(); // slot 1-cur staged; slot cur executed
            cur = 1 - cur;
        }

        // ── Drained. Broadcast parser state to the copier team ──
        if (is_parser && lane == 0) {
            s_state[0] = cmd_pos;   s_state[1] = lit_pos;
            s_state[2] = off16_pos; s_state[3] = dst_pos;
            s_state[4] = (uint32_t)recent_offset;
            s_state[5] = length_offset;
        }
        __syncthreads();
        cmd_pos = s_state[0];   lit_pos = s_state[1];
        off16_pos = s_state[2]; dst_pos = s_state[3];
        recent_offset = (int32_t)s_state[4];
        length_offset = s_state[5];

        if (cmd_pos >= cmd_size) break; // cmd exhausted — done

        // ── Serial long token (both warps idle → safe to write) ──
        uint32_t slit = 0, smatch = 0;
        int32_t soff = recent_offset;
        uint32_t suse_recent = 0;
        if (is_parser && lane == 0) {
            uint32_t token = cmd[cmd_pos];
            if (token == TOKEN_LONG_LITERAL) {
                suse_recent = 1;
                slit = readLength(length_stream, length_offset, length_remaining)
                     + LONG_LITERAL_BASE;
            } else if (token == TOKEN_LONG_NEAR) {
                smatch = readLength(length_stream, length_offset, length_remaining)
                       + LONG_NEAR_BASE;
                if (off16_pos < off16_count) {
                    uint16_t v;
                    if constexpr (OFF16_SPLIT) {
                        v = (uint16_t)off16_lo[off16_pos] | ((uint16_t)off16_hi[off16_pos] << 8);
                    } else {
                        v = readU16LE(off16_raw + off16_pos * OFF16_ENTRY_BYTES);
                    }
                    soff = -(int32_t)v;
                    off16_pos++;
                }
            }
            s_long[0] = slit; s_long[1] = smatch;
            s_long[2] = (uint32_t)soff; s_long[3] = suse_recent;
            s_long[4] = off16_pos;
        }
        __syncthreads();
        slit = s_long[0]; smatch = s_long[1];
        soff = (int32_t)s_long[2]; suse_recent = s_long[3];
        off16_pos = s_long[4];
        cmd_pos++;
        // length_offset advanced on parser lane 0 only; rebroadcast.
        if (is_parser && lane == 0) s_state[5] = length_offset;
        __syncthreads();
        length_offset = s_state[5];

        if (slit > 0) {
            // 64-thread cooperative literal copy.
            uint32_t gl = threadIdx.y * WARP_SIZE + lane;
            for (uint32_t i = gl; i < slit; i += SLZ_PIPE_BLOCK_THREADS)
                dst[dst_pos + i] = lit[lit_pos + i];
            __syncthreads();
        }
        dst_pos += slit;
        lit_pos += slit;
        if (smatch > 0) {
            // warpMatchCopy is warp-granular; run it on the parser
            // warp only (the overlap/serial distinction lives in
            // match_dist, which both warps would compute identically
            // — one warp avoids double-writing).
            if (is_parser) {
                uint32_t msrc = (uint32_t)((int32_t)dst_pos + soff);
                warpMatchCopy(dst, dst_pos, msrc, smatch, -soff, lane);
            }
            __syncthreads();
        }
        dst_pos += smatch;
        if (!suse_recent) recent_offset = soff;
        // loop: re-prime the pipeline
    }

    // ── Trailing literals (both warps, 64 threads) ───────────────
    uint32_t trailing = (lit_size > lit_pos) ? (lit_size - lit_pos) : 0;
    uint32_t gl = threadIdx.y * WARP_SIZE + lane;
    for (uint32_t i = gl; i < trailing; i += SLZ_PIPE_BLOCK_THREADS)
        dst[dst_pos + i] = lit[lit_pos + i];
}

// ── Bridge: header parse + pipeline entry ────────────────────────
// Stream headers are parsed on the parser warp (same serial walk as
// parseAndDecodeSubChunkRaw in lz_dispatch.cuh) and broadcast to the
// copier warp through shared memory.
__device__ void parseAndDecodeSubChunkRawPipelined(
    const uint8_t* __restrict__ sc_src,
    uint32_t sc_comp_size,
    uint32_t sc_decomp_size,
    uint8_t* __restrict__ dst,
    uint32_t dst_offset,
    uint32_t base_offset,
    PipeBatch* s_batch
) {
    const int lane = threadIdx.x & LANE_MASK;
    const uint32_t warp_id = threadIdx.y;
    const uint8_t* src = sc_src;
    const uint8_t* src_end = sc_src + sc_comp_size;

    uint32_t initial_copy = 0;
    if (base_offset == 0) {
        uint32_t gl = warp_id * WARP_SIZE + lane;
        if (gl < INITIAL_LITERAL_COPY_BYTES) dst[dst_offset + gl] = src[gl];
        __syncthreads();
        src += INITIAL_LITERAL_COPY_BYTES;
        initial_copy = INITIAL_LITERAL_COPY_BYTES;
    }

    __shared__ uint64_t s_parse[8];
    if (warp_id == 0) {
        const uint8_t* lit_ptr = src;
        uint32_t lit_size = 0;
        if (lane == 0) { lit_size = parseRawStreamSize(src); lit_ptr = src; src += lit_size; }
        lit_size = __shfl_sync(FULL_WARP_MASK, lit_size, 0);
        src = broadcastSrc(sc_src, src); lit_ptr = src - lit_size;

        const uint8_t* cmd_ptr;
        uint32_t cmd_size = 0;
        if (lane == 0) { cmd_size = parseRawStreamSize(src); cmd_ptr = src; src += cmd_size; }
        cmd_size = __shfl_sync(FULL_WARP_MASK, cmd_size, 0);
        src = broadcastSrc(sc_src, src); cmd_ptr = src - cmd_size;

        // block2_cmd_offset present only for sub-chunks > 64 KB; the
        // raw pipeline path serves sc<=0.25 (64 KB) workloads where it
        // is absent, but parse it for stream-position correctness.
        if (lane == 0 && sc_decomp_size > LZ_BLOCK_SIZE) { src += 2; }
        src = broadcastSrc(sc_src, src);

        const uint8_t* off16_raw = src;
        uint32_t off16_count = 0;
        if (lane == 0) { off16_count = readU16LE(src); off16_raw = src + 2; src += 2 + off16_count * OFF16_ENTRY_BYTES; }
        off16_count = __shfl_sync(FULL_WARP_MASK, off16_count, 0);
        src = broadcastSrc(sc_src, src); off16_raw = src - off16_count * OFF16_ENTRY_BYTES;

        uint32_t len_avail = 0;
        const uint8_t* len_stream = src;
        if (lane == 0) { uint32_t tmp = readLE24(src); src += 3; (void)tmp; len_stream = src; len_avail = (uint32_t)((uintptr_t)src_end - (uintptr_t)src); }
        len_avail = __shfl_sync(FULL_WARP_MASK, len_avail, 0);
        src = broadcastSrc(sc_src, src); len_stream = src;

        if (lane == 0) {
            s_parse[0] = (uint64_t)(uintptr_t)lit_ptr;
            s_parse[1] = (uint64_t)(uintptr_t)cmd_ptr;
            s_parse[2] = (uint64_t)(uintptr_t)off16_raw;
            s_parse[3] = (uint64_t)(uintptr_t)len_stream;
            s_parse[4] = lit_size;
            s_parse[5] = cmd_size;
            s_parse[6] = off16_count;
            s_parse[7] = len_avail;
        }
    }
    __syncthreads();

    decodeSubChunkRawModePipelined<false>(
        (const uint8_t*)(uintptr_t)s_parse[1], (uint32_t)s_parse[5],
        (const uint8_t*)(uintptr_t)s_parse[0], (uint32_t)s_parse[4],
        (const uint8_t*)(uintptr_t)s_parse[2], (uint32_t)s_parse[6],
        nullptr, nullptr,
        (const uint8_t*)(uintptr_t)s_parse[3], (uint32_t)s_parse[7],
        dst, sc_decomp_size, initial_copy, dst_offset,
        s_batch
    );
}
