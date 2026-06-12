//! 1:1 port of src/encode/encode_lz.zig.
//!
//! L1 hot path: dispatches the LZ encode pipeline (lz_encode_kernel.comp)
//! to produce per-chunk compressed streams. All device ops funnel through
//! `vulkan_api.procs.*` — no direct vk*/vma calls in this file.

const std = @import("std");
const vk = @import("../decode/vulkan_api.zig");
const module_loader = @import("module_loader.zig");
// A-008 (BDA): pulled in for `getBufferDeviceAddress` — wraps
// vkGetBufferDeviceAddress to retrieve the raw u64 device address of
// the per-chunk hash buffer so the LZ encode kernel can dereference it
// through a buffer_reference (bypassing the 4 GiB SSBO range cap).
const decode_module_loader = @import("../decode/module_loader.zig");
const encode_context = @import("encode_context.zig");
const levels = @import("levels.zig");
const gpu_decode = @import("../decode/driver.zig");
// VK adaptation: per-phase QPC profile accumulators live on
// fast_framed.zig (the encode orchestrator). Pulled in for the
// SLZ_VK_PROFILE_PHASES checkpoints below.
const enc_phase = @import("fast_framed.zig");

inline fn qpcDeltaNs(start: i64, end: i64) i64 {
    return @intFromFloat(vk.qpcMs(start, end) * 1e6);
}

/// VK adaptation A-023 (testing hook). Set to a positive value before
/// calling `gpuCompressImpl` to cap `batch_count` and force the
/// batched-dispatch path. Default 0 = inactive (production behavior).
/// Mirrored by the SLZ_VK_FORCE_BATCH env var so command-line runs
/// can exercise the same path. See A-023 entry in srcVK/PortAdaptations.md.
pub var g_force_batch_count_for_test: u32 = 0;

const VkDeviceBuffer = vk.VkDeviceBuffer;
const VK_SUCCESS_RC = vk.VK_SUCCESS_RC;
const EncodeContext = encode_context.EncodeContext;
const CompressChunkDesc = encode_context.CompressChunkDesc;

// ── v4 #16: preset-dictionary staging (CUDA reference:
// src/encode/encode_lz.zig:28-81) ─────────────────────────────────

/// Host twin of the kernel's hashKey6 (lz_format.glsl): Fibonacci
/// multiplier shifted for the k=6 key, take the top hash_bits.
/// Must stay bit-identical - the host builds the dictionary table the
/// kernel probes, and the table must be byte-identical to the CUDA
/// backend's (cross-backend frame identity for dict frames).
fn hashKey6(word8: u64, hash_bits: u5) u32 {
    const product = word8 *% 0x79B97F4A7C150000;
    return @intCast(product >> @intCast(@as(u7, 64) - hash_bits));
}

const dict_table_empty: u32 = 0xFFFF_FFFF; // kernel HASH_EMPTY

/// Make the dictionary bytes + the hashKey6 position table resident
/// on the device, cached by (id, hash_bits). The table maps each
/// bucket to the HIGHEST dict position hashing there (ascending
/// insertion, last writer wins - the same order a serial pass over
/// the dictionary would leave behind, and the CPU sibling's
/// preloadDictionary semantics). Returns false on VK failure.
pub fn ensureDictOnDevice(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    id: u32,
    data: []const u8,
    hash_bits: u32,
) bool {
    if (self.dict_cached_id == id and
        self.dict_cached_len == data.len and
        self.dict_cached_hash_bits == hash_bits) return true;
    self.dict_cached_id = 0;
    if (!module_loader.init()) return false;
    const h2d_fn = vk.procs.h2d orelse return false;

    const table_len = @as(usize, 1) << @intCast(hash_bits);
    const table = allocator.alloc(u32, table_len) catch return false;
    defer allocator.free(table);
    @memset(table, dict_table_empty);
    if (data.len >= 8) {
        var p: usize = 0;
        while (p + 8 <= data.len) : (p += 1) {
            const word = std.mem.readInt(u64, data[p..][0..8], .little);
            table[hashKey6(word, @intCast(hash_bits))] = @intCast(p);
        }
    }

    if (!encode_context.ensureBuf(&self.d_dict, &self.d_dict_size, data.len)) return false;
    if (!encode_context.ensureBuf(&self.d_dict_table, &self.d_dict_table_size, table_len * 4)) return false;
    if (h2d_fn(self.d_dict, @ptrCast(data.ptr), data.len) != VK_SUCCESS_RC) return false;
    if (h2d_fn(self.d_dict_table, @ptrCast(table.ptr), table_len * 4) != VK_SUCCESS_RC) return false;
    self.dict_cached_id = id;
    self.dict_cached_len = @intCast(data.len);
    self.dict_cached_hash_bits = hash_bits;
    return true;
}

/// CUDA reference: src/encode/encode_lz.zig:21-153. L1 hot LZ encode
/// launcher. Returns true on success, false on FFI / GPU failure.
pub fn gpuCompressImpl(
    self: *EncodeContext,
    input: []const u8,
    output: []u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes_out: []u32,
    io: ?std.Io,
    level: u8,
) bool {
    if (!module_loader.init()) return false;

    // VK adaptation: parallel-test serialization for this entry is
    // applied at compressFramedOne (fast_framed.zig) — it must span
    // gpuCompressImpl + the assemble trio so the inter-call shared
    // device buffers on enc_driver.g_default don't get clobbered by
    // a sibling worker between calls. See lockEncodeDispatcherMutex
    // comment in decode/module_loader.zig.

    const h2d_fn = vk.procs.h2d orelse return false;
    const d2h_fn = vk.procs.d2h orelse return false;
    const d2h_offset_fn = vk.procs.d2h_offset orelse return false;
    const launch_fn = vk.procs.launch_kernel orelse return false;
    const sync_fn = vk.procs.ctx_sync orelse return false;

    const num_chunks: u32 = @intCast(chunk_descs.len);
    const desc_bytes = chunk_descs.len * @sizeOf(CompressChunkDesc);
    const sizes_bytes = @as(usize, num_chunks) * 4;
    const chain = levels.useChainParser(level);
    // A-008 RESOLVED via BDA (`5d8b4d3`+). Previously this site clamped
    // `hash_bits` down so `num_chunks * (1<<hash_bits) * 4` fit under
    // Vulkan's per-SSBO-binding `maxStorageBufferRange = 4 GiB - 1` cap
    // (the cap silesia L3 was hitting at sc=0.25). The kernel now
    // addresses the global hash table via VK_KHR_buffer_device_address
    // — a raw `uint64_t` device pointer dereferenced through a
    // `buffer_reference` SSBO — which has no per-binding cap, so the
    // hash table is allowed to exceed 4 GiB and we use the full
    // `levels.hashBitsForLevel(level)` value matching CUDA at every
    // workload size. CUDA reference: src/encode/encode_lz.zig:43
    // (allocates full `num_chunks × (1<<hash_bits) × 4` bytes with no cap).
    const hash_bits: u32 = levels.hashBitsForLevel(level);
    const hash_size: usize = @as(usize, 1) << @intCast(hash_bits);

    if (!encode_context.ensureBuf(&self.d_input_persist, &self.d_input_size, input.len)) return false;
    if (!encode_context.ensureBuf(&self.d_output_persist, &self.d_output_size, output.len)) return false;
    if (!encode_context.ensureBuf(&self.d_descs_persist, &self.d_descs_size, desc_bytes)) return false;
    if (!encode_context.ensureBuf(&self.d_sizes_persist, &self.d_sizes_size, sizes_bytes)) return false;

    // Global hash tables. Chain mode uses 3 tables per block (first_hash +
    // long_hash + next_hash); non-chain modes use a single hash table.
    //
    // VK adaptation A-023 (2026-06-09): per-batch hash sizing. CUDA's
    // allocator with WDDM2 paging tolerates oversubscription past
    // physical VRAM (enwik9 L3 needs 14.9 GiB hash on a 16 GiB device,
    // sitting alongside ~5 GiB of other persistent buffers). VMA on
    // Vulkan strictly requires DEVICE_LOCAL_BIT and fails fast when no
    // contiguous VRAM block can satisfy the request. The kernel
    // already uses a per-chunk hash region (ht_base = chunk_id *
    // hash_size); batching the dispatch reduces the hash buffer to
    // batch_count * hash_size * 4 and produces BYTE-IDENTICAL output
    // because each chunk reinitialises its hash to HASH_EMPTY at entry
    // (kernel lines 162-165). Try the full CUDA-mirror allocation
    // first; on failure, halve batch_count until ensureBuf accepts the
    // smaller request.
    const hash_stride: usize = if (chain)
        hash_size + hash_size + (encode_context.NEXT_HASH_ENTRIES / 2)
    else
        hash_size;
    // VK adaptation A-023 (testing hook): force_batch_count_for_test
    // caps the initial batch_count regardless of how much VRAM the
    // device has. Lets in-process regression tests drive the batched-
    // dispatch path on inputs that would otherwise fit in a single
    // dispatch (the L3 batched path needs ~14 GiB of hash to organically
    // trigger, well beyond CI hardware). When the global stays at 0
    // (production default), the cap is inactive and the regular VRAM-
    // probe loop runs unmodified. The SLZ_VK_FORCE_BATCH env var
    // mirrors the global for command-line driven verification (see
    // tools/build_vk.bat or PowerShell snippets in PortAdaptations.md
    // A-023).
    var batch_count: u32 = num_chunks;
    if (g_force_batch_count_for_test != 0 and g_force_batch_count_for_test < batch_count)
        batch_count = g_force_batch_count_for_test;
    if (std.c.getenv("SLZ_VK_FORCE_BATCH")) |raw| {
        const span = std.mem.span(raw);
        if (std.fmt.parseInt(u32, span, 10) catch null) |forced| {
            if (forced > 0 and forced < batch_count) batch_count = forced;
        }
    }
    while (true) {
        const hash_bytes = @as(usize, batch_count) * hash_stride * 4;
        if (encode_context.ensureBuf(&self.d_hash_persist, &self.d_hash_size, hash_bytes)) break;
        if (batch_count <= 1) return false;
        batch_count = (batch_count + 1) / 2;
    }
    const batched_dispatch = batch_count < num_chunks;

    const d_input = self.d_input_persist;
    const d_output = self.d_output_persist;
    const d_descs = self.d_descs_persist;
    const d_sizes = self.d_sizes_persist;

    // Upload input + descriptors. When the caller's data is already
    // device-resident at `d_input_override`, populate `d_input_persist`
    // via D2D (no host bounce) instead of an H2D from the host `input`
    // slice.
    // VK adaptation: per-phase QPC profile checkpoints (lz.h2d_input /
    // lz.h2d_descs_init / lz.pre_kernel_sync). Pure measurement; does
    // NOT change dispatch / descriptor / sync logic.
    const _prof = enc_phase.g_enc_phase_profile_enabled;
    const _t_h2d_in0 = if (_prof) vk.qpcNow() else 0;
    if (self.d_input_override != 0) {
        const d2d = vk.procs.d2d orelse return false;
        if (d2d(d_input, self.d_input_override, input.len, 0) != VK_SUCCESS_RC) return false;
    } else {
        if (h2d_fn(d_input, @ptrCast(input.ptr), input.len) != VK_SUCCESS_RC) return false;
    }
    if (_prof) enc_phase.g_enc_phase_lz_h2d_input_ns += qpcDeltaNs(_t_h2d_in0, vk.qpcNow());
    const _t_h2d_di0 = if (_prof) vk.qpcNow() else 0;
    // VK adaptation A-023: when not batching, upload the full descs / clear
    // the full sizes array up-front (CUDA-mirror). When batching, the
    // per-batch loop below uploads only the active batch's descs and
    // clears only the active batch's sizes slot — keeps the per-batch
    // device traffic proportional to the batch.
    if (!batched_dispatch) {
        if (h2d_fn(d_descs, @ptrCast(chunk_descs.ptr), desc_bytes) != VK_SUCCESS_RC) return false;
        if (vk.procs.memset_d8) |memset_fn|
            if (memset_fn(d_sizes, 0, sizes_bytes) != VK_SUCCESS_RC) return false;
    }
    if (_prof) enc_phase.g_enc_phase_lz_h2d_descs_init_ns += qpcDeltaNs(_t_h2d_di0, vk.qpcNow());
    const _t_sync0 = if (_prof) vk.qpcNow() else 0;
    if (sync_fn() != VK_SUCCESS_RC) return false;
    if (_prof) enc_phase.g_enc_phase_lz_pre_kernel_sync_ns += qpcDeltaNs(_t_sync0, vk.qpcNow());

    const t_before = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

    var p_input = d_input;
    var p_output = d_output;
    var p_descs = d_descs;
    var p_sizes = d_sizes;
    var p_total = num_chunks;
    var p_hash_bits = hash_bits;
    var p_use_chain: u32 = if (chain) 1 else 0;
    // L4+ enables the greedy parser's match-range rehash (CPU engine_level>=2).
    // L3 stays without it - that is the L3/L4 distinction. L5 uses the chain
    // parser so the flag is inert there.
    // v4 #6 experiment (2026-06-10): L2 = greedy+rehash, no entropy (CUDA mirror).
    var p_l4: u32 = if (level >= 4 or level == 2) 1 else 0;

    // A-008 (BDA): query the raw device address of the per-chunk hash
    // table buffer and pack it as two trailing u32 push-constant slots
    // (low, high). The kernel reconstructs a `HashU32Ref`
    // buffer_reference at entry from `pc.hash_addr` and dereferences
    // every hash load/store through it — bypassing the per-binding
    // `maxStorageBufferRange = 4 GiB - 1` cap that previously forced
    // the L3 hash_bits clamp. Hash buffer is allocated with
    // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT in procMallocDevice
    // (srcVK/decode/module_loader.zig:3368) and the device feature
    // `bufferDeviceAddress = VK_TRUE` is enabled at vkCreateDevice
    // (srcVK/decode/module_loader.zig:2187 — shared device with
    // decode side).
    const hash_addr: u64 = decode_module_loader.getBufferDeviceAddress(self.d_hash_persist);
    if (hash_addr == 0) return false;
    var p_hash_addr_lo: u32 = @truncate(hash_addr);
    var p_hash_addr_hi: u32 = @truncate(hash_addr >> 32);

    // v4 #16 (CUDA reference: src/encode/encode_lz.zig:214-219): pass
    // the staged dictionary only when fast_framed armed it for THIS
    // call - the cache may hold a dictionary while a dictionary-less
    // frame must encode exactly as before. VK adaptation: BDA
    // addresses in the push constants instead of pointer params.
    var dict_addr: u64 = 0;
    var dict_ht_addr: u64 = 0;
    var p_dict_len: u32 = 0;
    if (self.dict_armed) {
        dict_addr = decode_module_loader.getBufferDeviceAddress(self.d_dict);
        dict_ht_addr = decode_module_loader.getBufferDeviceAddress(self.d_dict_table);
        if (dict_addr == 0 or dict_ht_addr == 0) return false;
        p_dict_len = self.dict_cached_len;
    }
    var p_dict_addr_lo: u32 = @truncate(dict_addr);
    var p_dict_addr_hi: u32 = @truncate(dict_addr >> 32);
    var p_dict_ht_lo: u32 = @truncate(dict_ht_addr);
    var p_dict_ht_hi: u32 = @truncate(dict_ht_addr >> 32);
    var p_dict_pad: u32 = 0;

    // params layout per procs.launch_kernel contract: the first
    // n_bindings entries are pointers to VkDeviceBuffer handles
    // (the shader's std430 buffer bindings 0..n_bindings-1), the
    // remaining entries are pointers to push-constant scalars packed
    // into the push_constant_size byte buffer in declaration order.
    // lz_encode: n_bindings=4 (Input, Output, Descs, CompSizes —
    // GlobalHash was dropped at A-008 BDA), push_constant_size=24
    // (total_chunks, hash_bits, use_chain, l4_features, hash_addr_lo,
    // hash_addr_hi). procLaunchKernel's push-constant copy loop writes
    // 4 bytes per slot, so the uint64_t hash_addr requires two trailing
    // slots (low u32 then high u32) to match the std430 push_constant
    // layout the kernel declares.
    var params = [_]?*anyopaque{
        @ptrCast(&p_input),
        @ptrCast(&p_output),
        @ptrCast(&p_descs),
        @ptrCast(&p_sizes),
        @ptrCast(&p_total),
        @ptrCast(&p_hash_bits),
        @ptrCast(&p_use_chain),
        @ptrCast(&p_l4),
        @ptrCast(&p_hash_addr_lo),
        @ptrCast(&p_hash_addr_hi),
        @ptrCast(&p_dict_addr_lo),
        @ptrCast(&p_dict_addr_hi),
        @ptrCast(&p_dict_ht_lo),
        @ptrCast(&p_dict_ht_hi),
        @ptrCast(&p_dict_len),
        @ptrCast(&p_dict_pad),
    };
    var extra = [_]?*anyopaque{null};

    const shared_bytes: c_uint = 0;
    // VK adaptation: lz.kernel phase = launch + post-kernel ctx_sync.
    const _t_kern0 = if (_prof) vk.qpcNow() else 0;
    const t_lz = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzLzEncodeKernel", 0);
    // Defer endKernelTiming so the pending begin event always gets a
    // matching end record - even on launch failure.
    defer gpu_decode.endKernelTiming(t_lz, 0);

    // VK adaptation A-023: batched kernel dispatch loop. When
    // batch_count == num_chunks (the common case), this runs ONCE and
    // is semantically equivalent to the unbatched CUDA-mirror dispatch.
    // When batch_count < num_chunks (VRAM-pressure fallback), the loop
    // splits the work into back-to-back dispatches that each reuse the
    // smaller hash buffer; per-chunk results stay BYTE-IDENTICAL because
    // each chunk reinitialises its own hash table region to HASH_EMPTY
    // at entry (kernel lines 162-165, srcVK/encode/lz_encode_kernel.comp),
    // so a chunk does not see any state left by a chunk from the previous
    // batch. The per-batch comp_sizes D2H and compressed-bytes gather also
    // run inside the loop so the host receives data in chunk-index order,
    // identical to the unbatched path.
    //
    // Gather rationale (preserved from the iter-3 single-shot
    // implementation): the gather slot collapses per-chunk D2H copies
    // into ONE vkCmdCopyBuffer with regionCount = n_regions on a
    // one-shot transfer-queue cmdbuf, drained by ONE submit + ONE wait.
    // The page-aligned host pointer (EncodeContext.gpu_out_buf via
    // procMallocHost in fast_framed) hits the iter-8 LRU import cache on
    // every call after the first. Per-batch dispatch keeps this property
    // — each batch submits one gather covering its regions.
    var batch_start: u32 = 0;
    while (batch_start < num_chunks) {
        const batch_end = @min(batch_start + batch_count, num_chunks);
        const bc: u32 = batch_end - batch_start;

        if (batched_dispatch) {
            // Per-batch descs H2D + sizes memset; both write to offset 0
            // of the persistent buffers so the kernel reads them with
            // local chunk_id = 0..bc. The chunk descs already carry
            // GLOBAL src_offset / dst_offset values, so output lands at
            // the right place in d_output without further translation.
            const batch_desc_bytes = @as(usize, bc) * @sizeOf(CompressChunkDesc);
            const batch_sizes_bytes_h2d = @as(usize, bc) * 4;
            if (h2d_fn(d_descs, @ptrCast(&chunk_descs[batch_start]), batch_desc_bytes) != VK_SUCCESS_RC) return false;
            if (vk.procs.memset_d8) |memset_fn|
                if (memset_fn(d_sizes, 0, batch_sizes_bytes_h2d) != VK_SUCCESS_RC) return false;
            if (sync_fn() != VK_SUCCESS_RC) return false;
        }

        p_total = bc;
        if (launch_fn(module_loader.kernel_fn, bc, 1, 1, 32, 1, 1, shared_bytes, 0, &params, &extra, null) != VK_SUCCESS_RC)
            return false;
        if (sync_fn() != VK_SUCCESS_RC) return false;

        // Per-batch comp_sizes D2H. The kernel writes to d_sizes[0..bc]
        // (local chunk_id); we stash them at the GLOBAL position
        // [batch_start..batch_end) so subsequent gather + downstream
        // passes (Huffman / assemble) read them at the same index they
        // would have in the unbatched path.
        const _t_dsz0 = if (_prof) vk.qpcNow() else 0;
        const batch_sizes_bytes = @as(usize, bc) * 4;
        if (d2h_fn(@ptrCast(&comp_sizes_out[batch_start]), d_sizes, batch_sizes_bytes) != VK_SUCCESS_RC) return false;
        if (_prof) enc_phase.g_enc_phase_lz_d2h_comp_sizes_ns += qpcDeltaNs(_t_dsz0, vk.qpcNow());

        // Per-batch gather of compressed bytes — ONLY at L3+ (v4 #17,
        // mirrored from the CUDA finding the same day): the Huffman
        // passes are the sole consumers of the host-side LZ bytes; at
        // L1/L2 the device-resident assemble reads d_output on the GPU
        // and the finished frame returns via the single d2h_final copy,
        // so the gather was pure waste (~4.5 ms at enwik8 scale).
        const _t_gather0 = if (_prof) vk.qpcNow() else 0;
        if (level < 3) {
            // skip — no host consumer below L3
        } else if (vk.procs.d2h_offset_gather) |gather_fn| {
            // VK adaptation (Gemini Risk A, 2026-06-08): persistent
            // scratch on EncodeContext instead of per-call
            // page_allocator.alloc — one VirtualAlloc lifetime-of-encoder
            // vs one-per-encode. ensureRegionsScratch sizes for the worst
            // case of the whole dispatch; per-batch use is a subslice.
            const regions_buf = self.ensureRegionsScratch(chunk_descs.len) orelse return false;
            var n_regions: usize = 0;
            for (batch_start..batch_end) |i| {
                const cs = comp_sizes_out[i];
                if (cs > 0) {
                    const dst_off = chunk_descs[i].dst_offset;
                    regions_buf[n_regions] = .{
                        .src_offset = dst_off,
                        .dst_offset = dst_off,
                        .size = cs,
                    };
                    n_regions += 1;
                }
            }
            if (n_regions > 0) {
                if (gather_fn(@ptrCast(output.ptr), d_output, regions_buf.ptr, n_regions) != VK_SUCCESS_RC) return false;
            }
        } else {
            // Per-chunk fallback. Mirrors CUDA's src/encode/encode_lz.zig
            // :144-150 exactly. Used only if the gather slot is somehow
            // unbound.
            for (batch_start..batch_end) |i| {
                const cs = comp_sizes_out[i];
                if (cs > 0) {
                    const dst_off = chunk_descs[i].dst_offset;
                    if (d2h_offset_fn(@ptrCast(output.ptr + dst_off), d_output, cs, dst_off) != VK_SUCCESS_RC) return false;
                }
            }
        }
        if (_prof) enc_phase.g_enc_phase_lz_d2h_gather_ns += qpcDeltaNs(_t_gather0, vk.qpcNow());

        batch_start = batch_end;
    }

    if (_prof) enc_phase.g_enc_phase_lz_kernel_ns += qpcDeltaNs(_t_kern0, vk.qpcNow());

    if (t_before) |t_start| {
        if (io) |io_val| {
            // Storage lives on the encode driver facade so external callers
            // reach it unchanged via `gpu_enc.last_kernel_ns`.
            const driver = @import("driver.zig");
            driver.last_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
        }
    }

    // VK adaptation A-023 (companion): when the dispatch was batched
    // (VRAM-pressure path), release the hash buffer here so the
    // downstream assembleFrame + frameAssemble passes — which need to
    // grow `d_asm_out` to roughly the assembled-payload size (~330 MB at
    // 1 GB L5) — have room. Without this, the LZ kernel succeeds in
    // batches that fit but `d_asm_out` then fails to allocate because
    // the persistent hash buffer is still holding multi-GB of VRAM. The
    // hash buffer will be re-allocated on the next gpuCompressImpl call;
    // for non-batched dispatches (the common case) the buffer stays
    // persistent across calls so warm encodes pay zero alloc overhead.
    if (batched_dispatch) {
        if (vk.procs.free_device) |free_fn| {
            if (self.d_hash_persist != 0) {
                _ = free_fn(self.d_hash_persist);
                self.d_hash_persist = 0;
                self.d_hash_size = 0;
            }
        }
    }

    return true;
}
