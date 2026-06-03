//! Per-context decode workspace — Vulkan mirror of CUDA's `DecodeContext`
//! growing-only buffer pool.
//!
//! CUDA reference: `src/decode/decode_context.zig` lines 188-375
//! (`DecodeContext` struct holds every device buffer as a (ptr, size) pair;
//! `ensureDeviceBuf` at line 35 is the grow-only allocator) and
//! `src/decode/decode_dispatch.zig` lines 503-579 + `src/decode/scan_gpu.zig`
//! lines 47-49, 101-102, 153-211 (the call sites that drive ensureDeviceBuf
//! at decode-time).
//!
//! Why this exists:
//!
//!   Before this module, every `runDecodePipelineEx` + `decodeSlz1ToBytesEx`
//!   call allocated ~17 fresh VkBuffers + VkDeviceMemory pairs at the start
//!   and freed them at the end. The Nsight Systems trace on NVIDIA enwik8
//!   showed 23.7 ms / call in vkAllocateMemory and 10.4 ms / call in
//!   vkFreeMemory — 34 ms of overhead per decode, on a 31 ms kernel-time
//!   workload. CUDA does NOT do this: every per-call cuMemAlloc lives in a
//!   `DecodeContext` field with a matching `_size` companion, and
//!   `ensureDeviceBuf` is a no-op when the existing allocation is large
//!   enough. First call grows; subsequent calls of same-or-smaller size
//!   pay no allocator cost.
//!
//! Mirror semantics (verified against `src/decode/decode_context.zig:35`):
//!
//!   * `Slot` packs (Buffer, size_bytes, usage_bits, MemMode) — every
//!     workspace slot remembers what shape it was last allocated with so a
//!     re-call with different usage/mode reallocates instead of binding
//!     wrong memory. (CUDA only carries `(ptr, size)` because every
//!     allocation is plain cuMemAlloc with no mode flags — Vulkan needs
//!     the mode + usage discriminators.)
//!   * `ensureWorkspaceBuf` is grow-only: if `slot.buffer.size >= needed`
//!     AND usage+mode match → no-op. Otherwise free + create. Matches
//!     CUDA's "current_size.* >= needed → return" early-out at
//!     decode_context.zig:36.
//!   * Workspace lifetime = driver.Context lifetime. `deinit` frees every
//!     allocated slot once (mirror of decode_context.zig:320 `deinit`).
//!   * `Slot.size` records `req.size` from `vkGetBufferMemoryRequirements`
//!     so the "still large enough" check uses the actual allocation size,
//!     not the requested size (which may have been rounded up).

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const l1_codec = @import("l1_codec.zig");

pub const WorkspaceError = error{
    BufferCreateFailed,
    MemoryAllocateFailed,
    MemoryTypeNotFound,
    BindBufferFailed,
    MapMemoryFailed,
};

/// One pooled workspace buffer. Equivalent of CUDA's
/// `(CUdeviceptr ptr, usize size)` field pair on `DecodeContext`
/// (decode_context.zig:195-353), extended with `usage` + `mode` because
/// Vulkan VkBuffer creation requires those at allocate time (CUDA's plain
/// cuMemAlloc has no equivalent — there's only one memory pool).
///
/// `size` stores the requested logical size (NOT req.size from the memory
/// requirements). The grow check `slot.size >= needed` is the exact same
/// shape as CUDA's `current_size.* >= needed` early-out — see
/// decode_context.zig:36.
///
/// `cap` is the actual VkMemoryRequirements.size the allocation can hold.
/// `ensureWorkspaceBuf` compares `needed <= cap` to skip the reallocation
/// when a same-shape request fits inside the existing allocation; this
/// lets a 95 MB enwik8 call grow once and then a 60 MB silesia-piece call
/// stay in the existing 95 MB allocation rather than shrinking it (CUDA
/// has the same behavior because cuMemAlloc never shrinks below the
/// `current_size` recorded in the struct).
pub const Slot = struct {
    buffer: l1_codec.Buffer = .{},
    size: vk.VkDeviceSize = 0,
    cap: vk.VkDeviceSize = 0,
    usage: vk.VkBufferUsageFlags = 0,
    mode: l1_codec.MemMode = .device_local_only,
};

/// All decode-side workspace buffers — one slot per CUDA `DecodeContext`
/// field. Names mirror CUDA where possible. See the field comments in
/// `src/decode/decode_context.zig:194-287` for the per-slot purpose; the
/// VK slot below each cite carries the same role.
pub const DecodeWorkspace = struct {
    // ── Pipeline (runDecodePipelineEx) — mirrors decode_pipeline_gpu.zig
    //    DecodeResult fields, which themselves mirror CUDA's
    //    DecodeContext.d_comp_persist / d_walk_chunks / d_walk_meta /
    //    d_first_sub_idx_persist / d_total_subchunks_buf / d_n_chunks_
    //    scratch / d_scan_staged / d_compact_{lit,tok,hi,lo,raw,counts} /
    //    d_entropy_off16_scratch on the CUDA side.

    /// Compressed SLZ1 input bytes (host fills, kernels read). Mirrors
    /// CUDA's `d_comp_persist` (decode_context.zig:205). CUDA pools this
    /// per decode_dispatch.zig:519 — same shape here.
    frame: Slot = .{},
    /// S007: host-visible staging buffer the host @memcpy's compressed
    /// bytes into. A vkCmdCopyBuffer recorded at the top of the decode
    /// command buffer copies staging→frame via the GPU's DMA/copy
    /// engine. Splits the upload so the host write lands in sysmem
    /// (~15 GB/s on this dev box) rather than the BAR window (~3 GB/s
    /// on NVIDIA discrete without rebar), and the device-side compute
    /// reads from VRAM-resident `frame`. Mirrors CUDA's
    /// `procs.h2d(d_comp_persist, src, ...)` at
    /// src/decode/decode_dispatch.zig:570 which uses cudaMemcpyAsync
    /// (DMA engine) from pinned host into the device pool.
    frame_staging: Slot = .{},
    /// walk_frame's per-chunk SlzChunkDesc output. Mirrors CUDA's
    /// `d_walk_chunks` (decode_context.zig:242).
    chunks: Slot = .{},
    /// walk_frame's 6-u32 meta (n_chunks / decomp_size / status / ...).
    /// Mirrors CUDA's `d_walk_meta` (decode_context.zig:244).
    meta: Slot = .{},
    /// prefix_sum_chunks output. Mirrors CUDA's `d_first_sub_idx_persist`
    /// (decode_context.zig:250).
    first_sub_idx: Slot = .{},
    /// prefix_sum_chunks running total. Mirrors CUDA's
    /// `d_total_subchunks_buf` (decode_context.zig:252).
    total_subs: Slot = .{},
    /// Self-gate counter staged from meta. Mirrors CUDA's
    /// `d_n_chunks_scratch` (decode_context.zig:278).
    n_chunks_scratch: Slot = .{},
    /// Packed staged-huff + staged-raw outputs of scan_parse. Mirrors
    /// CUDA's `d_scan_staged` (decode_context.zig:233).
    staged: Slot = .{},
    /// Compacted per-stream HuffDecChunkDesc arrays. Mirror CUDA's
    /// `d_compact_lit/tok/hi/lo` (decode_context.zig:261-268).
    compact_lit: Slot = .{},
    compact_tok: Slot = .{},
    compact_hi: Slot = .{},
    compact_lo: Slot = .{},
    /// Interleaved hi/lo RawOff16Descs. Mirrors CUDA's `d_compact_raw`
    /// (decode_context.zig:269).
    compact_raw: Slot = .{},
    /// 6 × u32 counts: [n_lit, n_tok, n_hi, n_lo, n_raw, n_merged].
    /// Mirrors CUDA's `d_compact_counts` (decode_context.zig:271).
    compact_counts: Slot = .{},
    /// Per-sub-chunk off16 entropy scratch. Mirrors CUDA's
    /// `d_entropy_scratch` + off16 view (decode_context.zig:214-221) —
    /// the VK pipeline only allocates the off16 region today (no
    /// Huffman on the L1 path) so it gets its own slot rather than
    /// sharing with the lit/tok regions.
    off16_scratch: Slot = .{},

    // ── SLZ1 codec wrap (decodeSlz1ToBytesEx) — three more pooled
    //    buffers that the Nsight trace listed alongside the 14 pipeline
    //    slots above for a total of 17 per-call allocations.

    /// 16-u32 ChunkDescs (l1_unwrap writes, lz_decode reads). Mirrors
    /// CUDA's `d_descs_persist` (decode_context.zig:207).
    chunks_16u32: Slot = .{},
    /// HOST_VISIBLE staging buffer the CPU host-walk fills with the
    /// 16-u32 per-chunk descriptors. A vkCmdCopyBuffer staged into
    /// `chunks_16u32` runs inside the decode cmdbuf. Mirrors the
    /// `frame_staging` shape above — sysmem write at ~15 GB/s, DMA to
    /// VRAM on the GPU's copy engine, kernels read from VRAM-resident
    /// `chunks_16u32`. CUDA equivalent: H2D copy at
    /// `src/decode/decode_dispatch.zig:530`.
    chunks_16u32_staging: Slot = .{},
    /// HOST_VISIBLE staging buffer for the 6-u32 WalkChunks descriptors
    /// (`pipeline_result.chunks`) on the host-input path. CPU fills it
    /// via host_walk; a copy stages into the device-local `chunks` slot.
    /// On the D2D path the GPU walk_frame kernel writes `chunks`
    /// directly so this staging buffer is never used.
    walk_chunks_staging: Slot = .{},
    /// Decoded output in DEVICE_LOCAL VRAM (two-buffer Intel-iGPU path).
    /// Mirrors CUDA's `d_output` (decode_context.zig:195).
    dst_b: Slot = .{},
    /// Decoded output in HOST_VISIBLE+HOST_CACHED sysmem (host @memcpy
    /// reads from this). On rebar-equipped discrete + iGPU shared-mem
    /// devices this slot IS the kernel target (single-buffer path);
    /// elsewhere it's the vkCmdCopyBuffer destination of dst_b. Mirrors
    /// CUDA's pinned-host mirror `h_pinned_output` (decode_context.zig:200)
    /// in role though Vulkan uses a HOST_VISIBLE VkBuffer instead of a
    /// cuMemAllocHost block.
    dst_stage: Slot = .{},

    /// VK_EXT_external_memory_host import of the CALLER's output host
    /// buffer. Holds the imported VkDeviceMemory + transient VkBuffer
    /// pair created by `l1_codec.importHostPointerBuffer`. Cached
    /// across decode calls keyed by `(caller_ptr, caller_size)` so the
    /// per-call vkAllocateMemory + vkCreateBuffer + vkBindBufferMemory
    /// triple (measured ~7 ms / call on NVIDIA RTX 4060 Ti for the
    /// 95 MB enwik8 import) is paid once instead of every call.
    ///
    /// No CUDA mirror: CUDA's cuMemcpyDtoH_v2
    /// (src/decode/decode_dispatch.zig:485) targets the caller's
    /// pointer directly without any prior driver-side "register"
    /// step, so there is nothing on the CUDA side to cache. This
    /// slot exists purely to amortize the Vulkan-specific
    /// `VkImportMemoryHostPointerInfoEXT` registration cost across
    /// repeat calls with the same caller buffer (the canonical case:
    /// CLI bench reuses `dec_buf` across all -r runs, and library
    /// callers typically reuse a single output buffer across many
    /// decompress calls).
    ///
    /// Cache-key fields (populated alongside `buffer`):
    ///   * `caller_ptr` — the host pointer passed to importHostPointerBuffer
    ///   * `size` (inherited from Slot) — the import size (= caller's
    ///     dst_total padded up to `external_memory_host_alignment`).
    ///     Reused only when the new request's import size is `<=`
    ///     the cached size, matching the grow-only semantics of every
    ///     other Slot in this struct.
    ///
    /// Lifetime: freed once in `deinit` before `destroyDevice`. The
    /// caller's host buffer is NOT freed (we don't own it) — only the
    /// driver-side VkDeviceMemory + VkBuffer handles created by the
    /// import are torn down.
    caller_imported: ImportedSlot = .{},
};

/// Specialized variant of `Slot` for the cached
/// VK_EXT_external_memory_host import. The plain `Slot` fields don't
/// fit cleanly because (a) the import does not carry a
/// `VkBufferUsageFlags` set we'd want to compare for cache-eviction
/// purposes (caller-imported buffers are always TRANSFER_DST_BIT),
/// (b) there is no `MemMode` (the memory type is dictated by the
/// host-pointer-properties query, not selectable), and (c) we need
/// to remember which `host_ptr` produced this import so the cache
/// can detect a caller swapping in a different buffer.
pub const ImportedSlot = struct {
    buffer: l1_codec.Buffer = .{},
    /// Import size in bytes (caller's dst_total padded up to
    /// `ctx.external_memory_host_alignment`). Reuse condition:
    /// `new_import_size <= size`. Mirrors the grow-only shape of
    /// `Slot.cap` (above) and CUDA's `current_size.* >= needed`
    /// early-out (decode_context.zig:36).
    size: vk.VkDeviceSize = 0,
    /// Host pointer the cached import was bound to. Reuse requires an
    /// exact match — a different caller pointer means a different
    /// VkDeviceMemory binding (the host-pointer-properties result and
    /// the VkImportMemoryHostPointerInfoEXT.pHostPointer field both
    /// depend on the pointer value).
    caller_ptr: ?*anyopaque = null,
};

/// Mirror of CUDA's `ensureDeviceBuf` (src/decode/decode_context.zig:35).
///
/// CUDA semantics (decode_context.zig:35-43):
///   - `current_size.* >= needed` → no-op.
///   - else free the existing allocation, alloc the new size, record the
///     new size on success.
///   - Returns `GpuError!void`; the free's failure is intentionally
///     dropped (decode_context.zig:30-34 comment).
///
/// VK adapts the signature by adding `usage` + `mode` because VkBuffer
/// creation needs those at allocate time, and treats a usage/mode change
/// as a reallocation trigger (same shape would otherwise bind the wrong
/// memory type).
///
/// Returns `WorkspaceError!void`. The createBufferEx failure path leaves
/// `slot.*` in the cleared post-free state — caller's `errdefer` chain
/// frees the partially-initialized workspace via `deinit`.
pub fn ensureWorkspaceBuf(
    ctx: *driver.Context,
    slot: *Slot,
    needed: vk.VkDeviceSize,
    usage: vk.VkBufferUsageFlags,
    mode: l1_codec.MemMode,
) WorkspaceError!void {
    // CUDA: `if (current_size.* >= needed) return;` (decode_context.zig:36).
    // VK: same check, but extended with usage/mode equality — different
    // usage/mode requires a different memory type, can't reuse.
    if (slot.buffer.buf != null and
        slot.cap >= needed and
        slot.usage == usage and
        slot.mode == mode)
    {
        // No-op: existing allocation is large enough AND was created with
        // the same usage/mode. Update the logical size only.
        slot.size = needed;
        return;
    }

    // CUDA: `if (ptr.* != 0) _ = free_fn(ptr.*); current_size.* = 0;`
    // (decode_context.zig:39-40). VK equivalent: full destroyBuffer
    // (unmap → destroy VkBuffer → free VkDeviceMemory).
    if (slot.buffer.buf != null) {
        l1_codec.destroyBuffer(ctx, &slot.buffer);
        slot.* = .{};
    }

    // CUDA: `if (alloc_fn(ptr, needed) != CUDA_SUCCESS) return
    // error.OutOfDeviceMemory; current_size.* = needed;`
    // (decode_context.zig:41-42). VK: createBufferEx wraps
    // vkCreateBuffer + vkAllocateMemory + vkBindBufferMemory + optional
    // vkMapMemory.
    const new_buf = l1_codec.createBufferEx(ctx, needed, usage, mode) catch |err| {
        // Map the L1Error subset we care about to the workspace's own
        // error set. We don't surface the underlying alloc-vs-create
        // distinction because the caller's recovery is identical
        // (return up the stack, errdefer tears the workspace down).
        return switch (err) {
            error.BufferCreateFailed => error.BufferCreateFailed,
            error.MemoryAllocateFailed => error.MemoryAllocateFailed,
            error.MemoryTypeNotFound => error.MemoryTypeNotFound,
            error.BindBufferFailed => error.BindBufferFailed,
            error.MapMemoryFailed => error.MapMemoryFailed,
            else => error.MemoryAllocateFailed,
        };
    };

    slot.buffer = new_buf;
    slot.size = needed;
    slot.cap = new_buf.size; // VkMemoryRequirements.size, the real capacity
    slot.usage = usage;
    slot.mode = mode;
}

/// Reuse-or-(re)import the caller-registered output host buffer via
/// VK_EXT_external_memory_host. Returns the cached
/// `l1_codec.Buffer` (always owned by the workspace — caller MUST NOT
/// destroyBuffer it; teardown happens once in `deinit`).
///
/// Reuse condition (cache hit, no driver allocations performed):
///   * cached buffer present AND
///   * cached `caller_ptr == host_ptr` AND
///   * cached `size >= needed_size`
///
/// Otherwise the prior import (if any) is freed, a fresh import is
/// created via `l1_codec.importHostPointerBuffer`, and the cache key
/// is updated. On any importHostPointerBuffer error the slot is left
/// in the cleared post-free state and the error is propagated — the
/// caller (slz1_codec) treats this exactly like the prior per-call
/// import-failure case (fall back to the dst_stage staging path).
///
/// Mirrors CUDA's `ensureDeviceBuf` (decode_context.zig:35-43): grow
/// the cached allocation when needed, no-op when the existing
/// allocation already fits the request. The ptr-match extension is
/// VK-specific (CUDA has no equivalent "register a host pointer"
/// step to cache).
pub fn ensureImportedHostBuf(
    ctx: *driver.Context,
    slot: *ImportedSlot,
    host_ptr: *anyopaque,
    needed_size: vk.VkDeviceSize,
    usage: vk.VkBufferUsageFlags,
) l1_codec.L1Error!l1_codec.Buffer {
    // Cache-hit fast path: same caller pointer AND existing import is
    // at least as large as the new request → reuse with zero driver
    // calls. This is the canonical case (CLI bench reuses one
    // `dec_buf` across all -r runs; library callers typically reuse
    // a single output buffer across many decompress calls).
    if (slot.buffer.buf != null and
        slot.caller_ptr == host_ptr and
        slot.size >= needed_size)
    {
        return slot.buffer;
    }

    // Cache miss: either first call, caller swapped pointers, or the
    // request needs more bytes than the existing import covers.
    // Tear down the prior import (if any) before creating the new one
    // — `VkImportMemoryHostPointerInfoEXT.allocationSize` MUST match
    // the registered region exactly, so we cannot "grow" in place.
    if (slot.buffer.buf != null) {
        l1_codec.destroyBuffer(ctx, &slot.buffer);
        slot.* = .{};
    }

    const new_buf = try l1_codec.importHostPointerBuffer(
        ctx,
        host_ptr,
        needed_size,
        usage,
    );

    slot.buffer = new_buf;
    slot.size = needed_size;
    slot.caller_ptr = host_ptr;
    return new_buf;
}

/// Free every owned VkBuffer + VkDeviceMemory in the workspace, reset
/// every slot to its default. Mirrors CUDA's
/// `DecodeContext.deinit` (decode_context.zig:320-374).
pub fn deinit(ctx: *driver.Context, ws: *DecodeWorkspace) void {
    if (ctx.dev == null) return;
    inline for (std.meta.fields(DecodeWorkspace)) |f| {
        // Two slot shapes live in DecodeWorkspace: the canonical
        // grow-only `Slot` (one per CUDA `DecodeContext` field) and
        // the cached VK_EXT_external_memory_host `ImportedSlot`. Both
        // own a `buffer: l1_codec.Buffer` we tear down identically;
        // the differing key/size fields don't affect destruction.
        if (f.type == Slot or f.type == ImportedSlot) {
            const slot_ptr = &@field(ws.*, f.name);
            if (slot_ptr.buffer.buf != null) {
                l1_codec.destroyBuffer(ctx, &slot_ptr.buffer);
            }
            slot_ptr.* = .{};
        }
    }
}
