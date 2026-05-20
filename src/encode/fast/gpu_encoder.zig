//! GPU L1 compression driver — CUDA Driver API.
//! Loads the compress kernel PTX, launches across all 64KB chunks,
//! reads back per-chunk compressed payloads for the CPU to assemble
//! into the final StreamLZ frame.

const std = @import("std");

const win32 = struct {
    extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.c) ?*anyopaque;
    extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.c) ?*anyopaque;
};

const CUresult = c_int;
const CUdevice = c_int;
const CUdeviceptr = u64;
const CUDA_SUCCESS: CUresult = 0;

// ── Module state ────────────────────────────────────────────────
var lib: ?*anyopaque = null;
var ctx: usize = 0;
var module: usize = 0;
var kernel_fn: usize = 0;
var tans_module: usize = 0;
var tans_kernel_fn: usize = 0;
var tans32_kernel_fn: usize = 0;
pub var tans32_shared_kernel_fn: usize = 0;
var initialized = false;

const FnInit = *const fn (c_uint) callconv(.c) CUresult;
const FnDeviceGet = *const fn (*CUdevice, c_int) callconv(.c) CUresult;
const FnCtxCreate = *const fn (*usize, c_uint, CUdevice) callconv(.c) CUresult;
const FnModuleLoadData = *const fn (*usize, [*]const u8) callconv(.c) CUresult;
const FnModuleGetFunction = *const fn (*usize, usize, [*:0]const u8) callconv(.c) CUresult;
const FnMemAlloc = *const fn (*CUdeviceptr, usize) callconv(.c) CUresult;
const FnMemFree = *const fn (CUdeviceptr) callconv(.c) CUresult;
const FnMemcpyHtoD = *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult;
const FnMemcpyDtoH = *const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult;
const FnLaunchKernel = *const fn (usize, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, usize, [*]?*anyopaque, [*]?*anyopaque) callconv(.c) CUresult;
const FnCtxSync = *const fn () callconv(.c) CUresult;
const FnMemsetD8 = *const fn (CUdeviceptr, u8, usize) callconv(.c) CUresult;
var cuInit_fn: ?FnInit = null;
var cuDeviceGet_fn: ?FnDeviceGet = null;
var cuCtxCreate_fn: ?FnCtxCreate = null;
var cuModuleLoadData_fn: ?FnModuleLoadData = null;
var cuModuleGetFunction_fn: ?FnModuleGetFunction = null;
var cuMemAlloc_fn: ?FnMemAlloc = null;
var cuMemFree_fn: ?FnMemFree = null;
var cuMemcpyHtoD_fn: ?FnMemcpyHtoD = null;
var cuMemcpyDtoH_fn: ?FnMemcpyDtoH = null;
var cuLaunchKernel_fn: ?FnLaunchKernel = null;
var cuCtxSynchronize_fn: ?FnCtxSync = null;
var cuMemsetD8_fn: ?FnMemsetD8 = null;

fn getProc(comptime T: type, name: [*:0]const u8) ?T {
    const h = lib orelse return null;
    const raw = win32.GetProcAddress(h, name) orelse return null;
    return @ptrCast(raw);
}

pub fn init() bool {
    if (initialized) return kernel_fn != 0;
    initialized = true;

    // Reuse CUDA context from the decode driver if available.
    // This avoids creating a second context which would clobber the first.
    const dec_gpu = @import("../../decode/fast/gpu_driver.zig");
    if (!dec_gpu.init()) return false;

    lib = win32.LoadLibraryA("nvcuda.dll");
    if (lib == null) return false;

    cuModuleLoadData_fn = getProc(FnModuleLoadData, "cuModuleLoadData");
    cuModuleGetFunction_fn = getProc(FnModuleGetFunction, "cuModuleGetFunction");
    cuMemAlloc_fn = getProc(FnMemAlloc, "cuMemAlloc_v2");
    cuMemFree_fn = getProc(FnMemFree, "cuMemFree_v2");
    cuMemcpyHtoD_fn = getProc(FnMemcpyHtoD, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH_fn = getProc(FnMemcpyDtoH, "cuMemcpyDtoH_v2");
    cuLaunchKernel_fn = getProc(FnLaunchKernel, "cuLaunchKernel");
    cuCtxSynchronize_fn = getProc(FnCtxSync, "cuCtxSynchronize");
    cuMemsetD8_fn = getProc(FnMemsetD8, "cuMemsetD8_v2");

    const ptx = @embedFile("gpu_encode_kernel.ptx") ++ "\x00";
    if ((cuModuleLoadData_fn orelse return false)(&module, ptx.ptr) != CUDA_SUCCESS) return false;

    const get_fn = cuModuleGetFunction_fn orelse return false;
    if (get_fn(&kernel_fn, module, "slzCompressL1Kernel") != CUDA_SUCCESS) return false;

    // Load tANS entropy kernel (5-state + 32-lane)
    const tans_ptx = @embedFile("gpu_tans_kernel.ptx") ++ "\x00";
    if ((cuModuleLoadData_fn orelse return false)(&tans_module, tans_ptx.ptr) != CUDA_SUCCESS) return false;
    if (get_fn(&tans_kernel_fn, tans_module, "slzTansEncodeKernel") != CUDA_SUCCESS) return false;
    // 32-lane tANS encoder (chunk_type=6). Loaded lazily — kernel symbol must
    // exist in PTX or this returns failure. Driver falls back to CPU path.
    _ = get_fn(&tans32_kernel_fn, tans_module, "slzTans32EncodeKernel");
    // 32-lane tANS encoder with frame-wide shared probability tables
    // (chunk_type=3). Phase 2 specialisation; optional.
    _ = get_fn(&tans32_shared_kernel_fn, tans_module, "slzTans32EncodeSharedKernel");

    return true;
}

pub fn isAvailable() bool {
    return init();
}

// ── Chunk descriptor (matches CUDA struct) ──────────────────────
pub const CompressChunkDesc = extern struct {
    src_offset: u32,
    src_size: u32,
    dst_offset: u32,
    dst_capacity: u32,
    is_first: u32,
};

/// Hash bits per level. Mirrors CPU `fast_framed.zig:955-963` engine-level
/// cap so GPU L1/L2/L3/L4/L5 produce hash distributions equivalent to CPU
/// L1/L2/L3/L4/L5 (modulo warp-parallel scan).
///
/// User L1 -> engine -2 -> cap 17
/// User L2 -> engine -1 -> cap 18
/// User L3 -> engine  1 -> cap 19
/// User L4 -> engine  2 -> cap 20
/// User L5 -> engine  4 -> 20 (no cap from engine; matches CPU practical)
fn hashBitsForLevel(level: u8) u32 {
    return switch (level) {
        1 => 17,
        2 => 18,
        3 => 19,
        4 => 20,
        5 => 20,
        else => 11,
    };
}

fn useGlobalHash(level: u8) bool {
    // All levels use global hash. L1's larger hash table (16-bit) needs
    // more than CUDA shared-mem allows; using global also dodges the
    // shared-mem-hash corruption bug seen at L2 sc>=0.5.
    _ = level;
    return true;
}

fn useChainParser(level: u8) bool {
    return level >= 5;
}

pub var last_kernel_ns: i64 = 0;

pub const TansChunkDesc = extern struct {
    src_offset: u32,
    src_size: u32,
    dst_offset: u32,
    dst_capacity: u32,
};

/// 32-lane tANS encode descriptor — matches `Tans32EncDesc` in
/// gpu_tans_kernel.cu. `src_stride` is 1 for byte streams (literals,
/// tokens) and 2 for off16 byte-plane extraction (combined with a
/// `src_offset` adjusted by +0 for lo plane or +1 for hi plane).
pub const Tans32EncDesc = extern struct {
    src_offset: u32,
    src_size: u32,    // logical symbol count after stride extraction
    src_stride: u32,  // 1 or 2
    dst_offset: u32,
    dst_capacity: u32,
};

/// Phase 2 shared-LUT encode descriptor — matches `Tans32EncSharedDesc`
/// in gpu_tans_kernel.cu. The extra `lut_id` picks which of the 4
/// frame-wide encoding tables to use (0=lit, 1=tok, 2=hi, 3=lo).
pub const Tans32EncSharedDesc = extern struct {
    src_offset: u32,
    src_size: u32,
    src_stride: u32,
    dst_offset: u32,
    dst_capacity: u32,
    lut_id: u32,
};

// Persistent buffers for the 32-lane tANS pass.
var d_tans32_descs_persist: CUdeviceptr = 0;
var d_tans32_descs_size: usize = 0;
var d_tans32_out_persist: CUdeviceptr = 0;
var d_tans32_out_size: usize = 0;
var d_tans32_sizes_persist: CUdeviceptr = 0;
var d_tans32_sizes_size: usize = 0;
// Phase 2 shared encode tables — 4 × Tans32EncodeTable in device memory.
var d_tans32_shared_tables: CUdeviceptr = 0;
var d_tans32_shared_tables_size: usize = 0;

/// Launch the 32-lane tANS encoder on a batch of streams already
/// resident in device memory (`d_input_persist` from the LZ encode
/// pass). Writes encoded bytes to a fresh device buffer and downloads
/// both the encoded data and the per-stream sizes.
///
/// `sizes_out[i]` semantics: low 31 bits = encoded byte count; if MSB
/// is set, tANS did not beat raw — caller should write a type-0 stream
/// using the source bytes instead.
///
/// Returns true on success. Caller owns the returned slices.
pub fn gpuEncodeTans32(
    allocator: std.mem.Allocator,
    descs: []const Tans32EncDesc,
    total_dst_bytes: usize,
    out_sizes: []u32,
    out_bytes: []u8,
) bool {
    if (!init()) return false;
    if (tans32_kernel_fn == 0) return false;

    const h2d_fn = cuMemcpyHtoD_fn orelse return false;
    const d2h_fn = cuMemcpyDtoH_fn orelse return false;
    const launch_fn = cuLaunchKernel_fn orelse return false;
    const sync_fn = cuCtxSynchronize_fn orelse return false;
    const memset_fn = cuMemsetD8_fn orelse return false;
    _ = allocator;

    const n: u32 = @intCast(descs.len);
    if (n == 0) return true;

    const desc_bytes: usize = descs.len * @sizeOf(Tans32EncDesc);
    const sizes_bytes: usize = descs.len * 4;

    if (!ensureBuf(&d_tans32_descs_persist, &d_tans32_descs_size, desc_bytes)) return false;
    if (!ensureBuf(&d_tans32_out_persist, &d_tans32_out_size, total_dst_bytes)) return false;
    if (!ensureBuf(&d_tans32_sizes_persist, &d_tans32_sizes_size, sizes_bytes)) return false;

    _ = h2d_fn(d_tans32_descs_persist, @ptrCast(descs.ptr), desc_bytes);
    _ = memset_fn(d_tans32_sizes_persist, 0, sizes_bytes);
    _ = sync_fn();

    // Source streams (literals, tokens, off16) live in the LZ-encode
    // OUTPUT buffer (d_output_persist) where the LZ kernel wrote them.
    var p_src = d_output_persist;
    var p_dst = d_tans32_out_persist;
    var p_descs = d_tans32_descs_persist;
    var p_sizes = d_tans32_sizes_persist;
    var p_n: u32 = n;
    var params = [_]?*anyopaque{
        @ptrCast(&p_src), @ptrCast(&p_dst), @ptrCast(&p_descs),
        @ptrCast(&p_sizes), @ptrCast(&p_n),
    };
    var extra = [_]?*anyopaque{null};

    // 1 warp (32 threads) per descriptor, 1 block per descriptor.
    if (launch_fn(tans32_kernel_fn, n, 1, 1, 32, 1, 1, 0, 0, &params, &extra) != CUDA_SUCCESS) return false;
    if (sync_fn() != CUDA_SUCCESS) return false;

    _ = d2h_fn(@ptrCast(out_sizes.ptr), d_tans32_sizes_persist, sizes_bytes);
    _ = d2h_fn(@ptrCast(out_bytes.ptr), d_tans32_out_persist, total_dst_bytes);
    return true;
}

// ── Persistent device buffers ──────────────────────────────────
var d_input_persist: CUdeviceptr = 0;
var d_input_size: usize = 0;
var d_output_persist: CUdeviceptr = 0;
var d_output_size: usize = 0;
var d_descs_persist: CUdeviceptr = 0;
var d_descs_size: usize = 0;
var d_tans_descs_persist: CUdeviceptr = 0;
var d_tans_descs_size: usize = 0;
var d_tans_out_persist: CUdeviceptr = 0;
var d_tans_out_size: usize = 0;
var d_tans_sizes_persist: CUdeviceptr = 0;
var d_tans_sizes_size: usize = 0;
var d_hash_persist: CUdeviceptr = 0;
var d_hash_size: usize = 0;
var d_sizes_persist: CUdeviceptr = 0;
var d_sizes_size: usize = 0;

fn ensureBuf(ptr: *CUdeviceptr, cur: *usize, needed: usize) bool {
    if (cur.* >= needed) return true;
    const free_fn = cuMemFree_fn orelse return false;
    const alloc_fn = cuMemAlloc_fn orelse return false;
    if (ptr.* != 0) _ = free_fn(ptr.*);
    cur.* = 0;
    if (alloc_fn(ptr, needed) != CUDA_SUCCESS) return false;
    cur.* = needed;
    return true;
}

/// Compress all chunks on GPU. Returns per-chunk compressed sizes.
/// Caller provides input data, output buffer, and chunk layout.
pub var tans_lit_sizes: ?[]u32 = null;
pub var tans_lit_data: ?[]u8 = null;
pub var tans_lit_offsets: ?[]u32 = null;

// ── 32-lane tANS pre-encoded streams (per sub-chunk) ───────────────
// sizes[i]:
//   if MSB set → tANS not beneficial; use raw bytes / memcpy.
//   else      → encoded byte count at data[offsets[i] .. offsets[i] + size].
// The encoded payload is the kernel's raw output: [128B sizes+states]
// [prob table][32 sub-streams]. The caller prepends a 5-byte type-6
// non-compact chunk header when writing to the final frame.
pub var tans32_lit_sizes: ?[]u32 = null;
pub var tans32_lit_data: ?[]u8 = null;
pub var tans32_lit_offsets: ?[]u32 = null;

pub var tans32_tok_sizes: ?[]u32 = null;
pub var tans32_tok_data: ?[]u8 = null;
pub var tans32_tok_offsets: ?[]u32 = null;

// Off16 hi/lo byte-plane streams. Both planes share one device output
// buffer and one descriptor array (hi descs first, then lo descs) but
// expose separate (sizes, data, offsets) host-side for clarity.
pub var tans32_off16hi_sizes: ?[]u32 = null;
pub var tans32_off16hi_data: ?[]u8 = null;
pub var tans32_off16hi_offsets: ?[]u32 = null;
pub var tans32_off16lo_sizes: ?[]u32 = null;
pub var tans32_off16lo_data: ?[]u8 = null;
pub var tans32_off16lo_offsets: ?[]u32 = null;

// Phase 2 shared-LUT encoded streams. Single descriptor array layout:
//   descs[0..N]     → literals (lut_id=0)
//   descs[N..2N]    → tokens   (lut_id=1)
//   descs[2N..3N]   → off16-hi (lut_id=2)
//   descs[3N..4N]   → off16-lo (lut_id=3)
// `tans32_shared_n_per_stream` records N. The body at offset O has
// length sizes[i]; the body starts with a 1-byte lut_id followed by
// [128B sizes+states][32 sub-streams].
pub var tans32_shared_sizes: ?[]u32 = null;
pub var tans32_shared_data: ?[]u8 = null;
pub var tans32_shared_offsets: ?[]u32 = null;
pub var tans32_shared_n_per_stream: u32 = 0;

/// Build literal descriptors and run the 32-lane tANS encoder.
/// Output is a per-sub-chunk encoded body (no chunk header). Caller
/// prepends the 5-byte type-6 non-compact chunk header when assembling
/// the final frame.
pub fn gpuEncodeLiteralsTans32(
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    if (!init()) return false;
    if (tans32_kernel_fn == 0) return false;
    const n = chunk_descs.len;
    if (n == 0) return false;

    var descs = allocator.alloc(Tans32EncDesc, n) catch return false;
    defer allocator.free(descs);
    var offsets = allocator.alloc(u32, n) catch return false;
    errdefer allocator.free(offsets);

    var total: u32 = 0;
    for (0..n) |i| {
        const cs = comp_sizes[i];
        const base: u32 = chunk_descs[i].dst_offset;
        const init_b: u32 = if (chunk_descs[i].is_first != 0) 8 else 0;
        offsets[i] = total;
        descs[i] = .{ .src_offset = 0, .src_size = 0, .src_stride = 1, .dst_offset = total, .dst_capacity = 0 };

        if (cs < init_b + 3) continue;
        const lit_hdr: u32 = base + init_b;
        const lit_count: u32 =
            (@as(u32, output[lit_hdr]) << 16) |
            (@as(u32, output[lit_hdr + 1]) << 8) |
            @as(u32, output[lit_hdr + 2]);
        if (lit_count == 0) continue;
        const lit_src: u32 = lit_hdr + 3;
        if (lit_src + lit_count > base + cs) continue;

        const dst_cap: u32 = lit_count + 512;
        descs[i] = .{
            .src_offset = lit_src,
            .src_size = lit_count,
            .src_stride = 1,
            .dst_offset = total,
            .dst_capacity = dst_cap,
        };
        total += dst_cap;
    }

    if (total == 0) {
        allocator.free(offsets);
        return false;
    }

    const sizes = allocator.alloc(u32, n) catch {
        allocator.free(offsets);
        return false;
    };
    errdefer allocator.free(sizes);
    const bytes = allocator.alloc(u8, total) catch {
        allocator.free(offsets);
        allocator.free(sizes);
        return false;
    };
    errdefer allocator.free(bytes);

    if (!gpuEncodeTans32(allocator, descs, total, sizes, bytes)) {
        allocator.free(offsets);
        allocator.free(sizes);
        allocator.free(bytes);
        return false;
    }

    tans32_lit_sizes = sizes;
    tans32_lit_data = bytes;
    tans32_lit_offsets = offsets;
    return true;
}

/// Build token descriptors from the downloaded GPU raw output and run
/// the 32-lane tANS encoder over all token streams in one launch.
/// Populates `tans32_tok_*` pub vars on success. Returns false (and
/// leaves the pub vars nulled) when the kernel is unavailable or any
/// step fails — caller falls back to CPU re-encode in that case.
pub fn gpuEncodeTokensTans32(
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    if (!init()) return false;
    if (tans32_kernel_fn == 0) return false;
    const n = chunk_descs.len;
    if (n == 0) return false;

    var descs = allocator.alloc(Tans32EncDesc, n) catch return false;
    defer allocator.free(descs);
    var offsets = allocator.alloc(u32, n) catch return false;
    errdefer allocator.free(offsets);

    var total: u32 = 0;
    for (0..n) |i| {
        const cs = comp_sizes[i];
        const base: u32 = chunk_descs[i].dst_offset;
        const init_b: u32 = if (chunk_descs[i].is_first != 0) 8 else 0;
        offsets[i] = total;
        descs[i] = .{
            .src_offset = 0,
            .src_size = 0,
            .src_stride = 1,
            .dst_offset = total,
            .dst_capacity = 0,
        };

        if (cs < init_b + 6) continue;
        const lit_hdr: u32 = base + init_b;
        const lit_count: u32 =
            (@as(u32, output[lit_hdr]) << 16) |
            (@as(u32, output[lit_hdr + 1]) << 8) |
            @as(u32, output[lit_hdr + 2]);
        const tok_hdr: u32 = lit_hdr + 3 + lit_count;
        if (tok_hdr + 3 > base + cs) continue;
        const tok_count: u32 =
            (@as(u32, output[tok_hdr]) << 16) |
            (@as(u32, output[tok_hdr + 1]) << 8) |
            @as(u32, output[tok_hdr + 2]);
        if (tok_count == 0) continue;
        const tok_src: u32 = tok_hdr + 3;

        // Capacity bound: 128B header + ~200B prob table + worst-case
        // bitstream (≤ src_size on success — we always fail-safe to raw
        // when total >= src_size). 512B headroom for table + padding.
        const dst_cap: u32 = tok_count + 512;
        descs[i] = .{
            .src_offset = tok_src,
            .src_size = tok_count,
            .src_stride = 1,
            .dst_offset = total,
            .dst_capacity = dst_cap,
        };
        total += dst_cap;
    }

    if (total == 0) {
        allocator.free(offsets);
        return false;
    }

    const sizes = allocator.alloc(u32, n) catch {
        allocator.free(offsets);
        return false;
    };
    errdefer allocator.free(sizes);
    const bytes = allocator.alloc(u8, total) catch {
        allocator.free(offsets);
        allocator.free(sizes);
        return false;
    };
    errdefer allocator.free(bytes);

    if (!gpuEncodeTans32(allocator, descs, total, sizes, bytes)) {
        allocator.free(offsets);
        allocator.free(sizes);
        allocator.free(bytes);
        return false;
    }

    tans32_tok_sizes = sizes;
    tans32_tok_data = bytes;
    tans32_tok_offsets = offsets;
    return true;
}

/// Same as gpuEncodeTokensTans32 but for the off16 byte planes. The
/// off16 stream layout in the raw GPU output is `[2B count][2*count
/// interleaved lo/hi bytes]`, so we use stride=2 and shift src_offset
/// by 0 (lo) or 1 (hi). Both planes run through one kernel launch.
///
/// The CPU re-encode only entropy-codes off16 when count >= 32; sub-
/// chunks below that threshold get an empty descriptor (src_size=0)
/// so the kernel returns a raw-fallback marker and the caller writes
/// the raw off16 stream verbatim.
pub fn gpuEncodeOff16Tans32(
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    if (!init()) return false;
    if (tans32_kernel_fn == 0) return false;
    const n = chunk_descs.len;
    if (n == 0) return false;

    const num_descs = n * 2;
    var descs = allocator.alloc(Tans32EncDesc, num_descs) catch return false;
    defer allocator.free(descs);
    var hi_offsets = allocator.alloc(u32, n) catch return false;
    errdefer allocator.free(hi_offsets);
    var lo_offsets = allocator.alloc(u32, n) catch return false;
    errdefer allocator.free(lo_offsets);

    var total: u32 = 0;
    for (0..n) |i| {
        const cs = comp_sizes[i];
        const base: u32 = chunk_descs[i].dst_offset;
        const init_b: u32 = if (chunk_descs[i].is_first != 0) 8 else 0;

        // Default to empty (no entropy coding for this sub-chunk).
        hi_offsets[i] = total;
        lo_offsets[i] = total;
        descs[i] = .{ .src_offset = 0, .src_size = 0, .src_stride = 2, .dst_offset = total, .dst_capacity = 0 };
        descs[n + i] = .{ .src_offset = 0, .src_size = 0, .src_stride = 2, .dst_offset = total, .dst_capacity = 0 };

        if (cs < init_b + 6) continue;
        const lit_hdr: u32 = base + init_b;
        const lit_count: u32 =
            (@as(u32, output[lit_hdr]) << 16) |
            (@as(u32, output[lit_hdr + 1]) << 8) |
            @as(u32, output[lit_hdr + 2]);
        const tok_hdr: u32 = lit_hdr + 3 + lit_count;
        if (tok_hdr + 3 > base + cs) continue;
        const tok_count: u32 =
            (@as(u32, output[tok_hdr]) << 16) |
            (@as(u32, output[tok_hdr + 1]) << 8) |
            @as(u32, output[tok_hdr + 2]);

        // cmd_stream2_offset (2 bytes) is present when sub-chunk > 64KB.
        const after_tok: u32 = tok_hdr + 3 + tok_count;
        const cmd2_size: u32 = if (chunk_descs[i].src_size > 0x10000) 2 else 0;
        const off16_hdr: u32 = after_tok + cmd2_size;
        if (off16_hdr + 2 > base + cs) continue;

        const off16_count: u32 =
            @as(u32, output[off16_hdr]) |
            (@as(u32, output[off16_hdr + 1]) << 8);
        if (off16_count < 32) continue; // matches CPU `>= 32` gate
        const off16_data: u32 = off16_hdr + 2;

        // hi descriptor: stride=2, src_offset shifted by 1
        const hi_cap: u32 = off16_count + 512;
        descs[i] = .{
            .src_offset = off16_data + 1,
            .src_size = off16_count,
            .src_stride = 2,
            .dst_offset = total,
            .dst_capacity = hi_cap,
        };
        hi_offsets[i] = total;
        total += hi_cap;

        // lo descriptor: stride=2, src_offset unshifted
        const lo_cap: u32 = off16_count + 512;
        descs[n + i] = .{
            .src_offset = off16_data,
            .src_size = off16_count,
            .src_stride = 2,
            .dst_offset = total,
            .dst_capacity = lo_cap,
        };
        lo_offsets[i] = total;
        total += lo_cap;
    }

    if (total == 0) {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        return false;
    }

    const all_sizes = allocator.alloc(u32, num_descs) catch {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        return false;
    };
    errdefer allocator.free(all_sizes);
    const bytes = allocator.alloc(u8, total) catch {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        allocator.free(all_sizes);
        return false;
    };
    errdefer allocator.free(bytes);

    if (!gpuEncodeTans32(allocator, descs, total, all_sizes, bytes)) {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        allocator.free(all_sizes);
        allocator.free(bytes);
        return false;
    }

    // Split sizes into hi (first n) and lo (next n). Reuse all_sizes
    // as backing store for both halves by aliasing — but lifetimes
    // make that messy. Just allocate two arrays and copy.
    const hi_sizes = allocator.alloc(u32, n) catch {
        allocator.free(hi_offsets); allocator.free(lo_offsets);
        allocator.free(all_sizes); allocator.free(bytes);
        return false;
    };
    const lo_sizes = allocator.alloc(u32, n) catch {
        allocator.free(hi_offsets); allocator.free(lo_offsets);
        allocator.free(hi_sizes);
        allocator.free(all_sizes); allocator.free(bytes);
        return false;
    };
    @memcpy(hi_sizes, all_sizes[0..n]);
    @memcpy(lo_sizes, all_sizes[n..]);
    allocator.free(all_sizes);

    tans32_off16hi_sizes = hi_sizes;
    tans32_off16hi_data = bytes; // shared buffer; both hi and lo offsets index into it
    tans32_off16hi_offsets = hi_offsets;
    tans32_off16lo_sizes = lo_sizes;
    tans32_off16lo_data = bytes; // SAME pointer — only one of {hi,lo} should free it
    tans32_off16lo_offsets = lo_offsets;
    return true;
}

const shared_luts_mod = @import("gpu_shared_luts.zig");

/// Phase 2: encode all four stream types across all sub-chunks using
/// the four pre-built frame-wide encoding tables. One kernel launch,
/// 4*N descriptors. Populates `tans32_shared_*` pub vars on success.
pub fn gpuEncodeAllSharedTans32(
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
    tables: [4]shared_luts_mod.Tans32EncodeTable,
) bool {
    if (!init()) return false;
    if (tans32_shared_kernel_fn == 0) return false;
    const n: u32 = @intCast(chunk_descs.len);
    if (n == 0) return false;

    const num_descs = @as(usize, n) * 4;

    var descs = allocator.alloc(Tans32EncSharedDesc, num_descs) catch return false;
    defer allocator.free(descs);
    var offsets = allocator.alloc(u32, num_descs) catch return false;
    errdefer allocator.free(offsets);

    var total: u32 = 0;

    // Helper: parse one sub-chunk's stream offsets/sizes once and fill in
    // each of the four descriptor slots.
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const cs = comp_sizes[i];
        const base: u32 = chunk_descs[i].dst_offset;
        const init_b: u32 = if (chunk_descs[i].is_first != 0) 8 else 0;

        // Default empty for all four streams.
        inline for ([_]u32{ 0, 1, 2, 3 }) |sid| {
            descs[sid * n + i] = .{
                .src_offset = 0,
                .src_size = 0,
                .src_stride = 1,
                .dst_offset = total,
                .dst_capacity = 0,
                .lut_id = sid,
            };
            offsets[sid * n + i] = total;
        }

        if (cs < init_b + 3) continue;

        const lit_hdr: u32 = base + init_b;
        const lit_count: u32 =
            (@as(u32, output[lit_hdr]) << 16) |
            (@as(u32, output[lit_hdr + 1]) << 8) |
            @as(u32, output[lit_hdr + 2]);
        const lit_src: u32 = lit_hdr + 3;
        if (lit_count > 0 and lit_src + lit_count <= base + cs) {
            const cap: u32 = lit_count + 512;
            descs[0 * n + i] = .{
                .src_offset = lit_src,
                .src_size = lit_count,
                .src_stride = 1,
                .dst_offset = total,
                .dst_capacity = cap,
                .lut_id = 0,
            };
            offsets[0 * n + i] = total;
            total += cap;
        }

        const tok_hdr: u32 = lit_src + lit_count;
        if (tok_hdr + 3 > base + cs) continue;
        const tok_count: u32 =
            (@as(u32, output[tok_hdr]) << 16) |
            (@as(u32, output[tok_hdr + 1]) << 8) |
            @as(u32, output[tok_hdr + 2]);
        const tok_src: u32 = tok_hdr + 3;
        if (tok_count > 0 and tok_src + tok_count <= base + cs) {
            const cap: u32 = tok_count + 512;
            descs[1 * n + i] = .{
                .src_offset = tok_src,
                .src_size = tok_count,
                .src_stride = 1,
                .dst_offset = total,
                .dst_capacity = cap,
                .lut_id = 1,
            };
            offsets[1 * n + i] = total;
            total += cap;
        }

        // off16 — same parse as gpuEncodeOff16Tans32
        const cmd2_size: u32 = if (chunk_descs[i].src_size > 0x10000) 2 else 0;
        const off16_hdr: u32 = tok_src + tok_count + cmd2_size;
        if (off16_hdr + 2 > base + cs) continue;
        const off16_count: u32 =
            @as(u32, output[off16_hdr]) | (@as(u32, output[off16_hdr + 1]) << 8);
        if (off16_count < 32) continue;
        const off16_data: u32 = off16_hdr + 2;
        if (off16_data + off16_count * 2 > base + cs) continue;

        const hi_cap: u32 = off16_count + 512;
        descs[2 * n + i] = .{
            .src_offset = off16_data + 1,
            .src_size = off16_count,
            .src_stride = 2,
            .dst_offset = total,
            .dst_capacity = hi_cap,
            .lut_id = 2,
        };
        offsets[2 * n + i] = total;
        total += hi_cap;

        const lo_cap: u32 = off16_count + 512;
        descs[3 * n + i] = .{
            .src_offset = off16_data,
            .src_size = off16_count,
            .src_stride = 2,
            .dst_offset = total,
            .dst_capacity = lo_cap,
            .lut_id = 3,
        };
        offsets[3 * n + i] = total;
        total += lo_cap;
    }

    if (total == 0) {
        allocator.free(offsets);
        return false;
    }

    const sizes = allocator.alloc(u32, num_descs) catch {
        allocator.free(offsets);
        return false;
    };
    errdefer allocator.free(sizes);
    const bytes = allocator.alloc(u8, total) catch {
        allocator.free(offsets);
        allocator.free(sizes);
        return false;
    };
    errdefer allocator.free(bytes);

    // ── Upload 4 encoding tables ─────────────────────────────────
    const tables_bytes = @sizeOf(shared_luts_mod.Tans32EncodeTable) * 4;
    if (!ensureBuf(&d_tans32_shared_tables, &d_tans32_shared_tables_size, tables_bytes)) {
        allocator.free(offsets); allocator.free(sizes); allocator.free(bytes);
        return false;
    }

    const h2d_fn = cuMemcpyHtoD_fn orelse return false;
    const d2h_fn = cuMemcpyDtoH_fn orelse return false;
    const launch_fn = cuLaunchKernel_fn orelse return false;
    const sync_fn = cuCtxSynchronize_fn orelse return false;
    const memset_fn = cuMemsetD8_fn orelse return false;

    var tables_copy = tables;
    _ = h2d_fn(d_tans32_shared_tables, @ptrCast(&tables_copy), tables_bytes);

    // ── Allocate device buffers ───────────────────────────────────
    const desc_bytes_dev: usize = num_descs * @sizeOf(Tans32EncSharedDesc);
    const sizes_bytes_dev: usize = num_descs * 4;
    if (!ensureBuf(&d_tans32_descs_persist, &d_tans32_descs_size, desc_bytes_dev) or
        !ensureBuf(&d_tans32_out_persist, &d_tans32_out_size, total) or
        !ensureBuf(&d_tans32_sizes_persist, &d_tans32_sizes_size, sizes_bytes_dev))
    {
        allocator.free(offsets); allocator.free(sizes); allocator.free(bytes);
        return false;
    }

    _ = h2d_fn(d_tans32_descs_persist, @ptrCast(descs.ptr), desc_bytes_dev);
    _ = memset_fn(d_tans32_sizes_persist, 0, sizes_bytes_dev);
    _ = sync_fn();

    var p_src = d_output_persist;
    var p_dst = d_tans32_out_persist;
    var p_descs_dev = d_tans32_descs_persist;
    var p_sizes_dev = d_tans32_sizes_persist;
    var p_n: u32 = @intCast(num_descs);
    var p_tables = d_tans32_shared_tables;

    var params = [_]?*anyopaque{
        @ptrCast(&p_src), @ptrCast(&p_dst), @ptrCast(&p_descs_dev),
        @ptrCast(&p_sizes_dev), @ptrCast(&p_n), @ptrCast(&p_tables),
    };
    var extra = [_]?*anyopaque{null};

    if (launch_fn(tans32_shared_kernel_fn, @intCast(num_descs), 1, 1, 32, 1, 1, 0, 0, &params, &extra) != CUDA_SUCCESS) {
        allocator.free(offsets); allocator.free(sizes); allocator.free(bytes);
        return false;
    }
    if (sync_fn() != CUDA_SUCCESS) {
        allocator.free(offsets); allocator.free(sizes); allocator.free(bytes);
        return false;
    }

    _ = d2h_fn(@ptrCast(sizes.ptr), d_tans32_sizes_persist, sizes_bytes_dev);
    _ = d2h_fn(@ptrCast(bytes.ptr), d_tans32_out_persist, total);

    tans32_shared_sizes = sizes;
    tans32_shared_data = bytes;
    tans32_shared_offsets = offsets;
    tans32_shared_n_per_stream = n;
    return true;
}

pub fn gpuCompress(
    input: []const u8,
    output: []u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes_out: []u32,
    io: ?std.Io,
    level: u8,
) bool {
    if (!init()) return false;

    const h2d_fn = cuMemcpyHtoD_fn orelse return false;
    const d2h_fn = cuMemcpyDtoH_fn orelse return false;
    const launch_fn = cuLaunchKernel_fn orelse return false;
    const sync_fn = cuCtxSynchronize_fn orelse return false;

    const num_chunks: u32 = @intCast(chunk_descs.len);
    const desc_bytes = chunk_descs.len * @sizeOf(CompressChunkDesc);
    const sizes_bytes = @as(usize, num_chunks) * 4;
    const hash_bits: u32 = hashBitsForLevel(level);
    const hash_size: usize = @as(usize, 1) << @intCast(hash_bits);
    const global = useGlobalHash(level);
    const chain = useChainParser(level);

    if (!ensureBuf(&d_input_persist, &d_input_size, input.len)) return false;
    if (!ensureBuf(&d_output_persist, &d_output_size, output.len)) return false;
    if (!ensureBuf(&d_descs_persist, &d_descs_size, desc_bytes)) return false;
    if (!ensureBuf(&d_sizes_persist, &d_sizes_size, sizes_bytes)) return false;

    // Global hash tables — chain mode uses 3 tables per block:
    //   first_hash (hash_size u32) + long_hash (hash_size u32) + next_hash (32768 u16 = 16384 u32)
    if (chain) {
        const next_hash_words: usize = 65536 / 2; // 65536 u16 entries = 32768 u32 words
        const table_stride = hash_size + hash_size + next_hash_words;
        const hash_bytes = @as(usize, num_chunks) * table_stride * 4;
        if (!ensureBuf(&d_hash_persist, &d_hash_size, hash_bytes)) return false;
    } else if (global) {
        const hash_bytes = @as(usize, num_chunks) * hash_size * 4;
        if (!ensureBuf(&d_hash_persist, &d_hash_size, hash_bytes)) return false;
    }

    const d_input = d_input_persist;
    const d_output = d_output_persist;
    const d_descs = d_descs_persist;
    const d_sizes = d_sizes_persist;

    // Upload input + descriptors, zero sizes
    _ = h2d_fn(d_input, @ptrCast(input.ptr), input.len);
    _ = h2d_fn(d_descs, @ptrCast(chunk_descs.ptr), desc_bytes);
    if (cuMemsetD8_fn) |memset_fn| _ = memset_fn(d_sizes, 0, sizes_bytes);
    _ = sync_fn();

    const t_before = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

    var p_input = d_input;
    var p_output = d_output;
    var p_descs = d_descs;
    var p_global_hash: CUdeviceptr = if (global or chain) d_hash_persist else 0;
    var p_sizes = d_sizes;
    var p_total = num_chunks;
    var p_hash_bits = hash_bits;
    var p_use_chain: u32 = if (chain) 1 else 0;
    // L4+ enables the greedy parser's match-range rehash (CPU engine_level>=2).
    // L3 stays without it — that is the L3/L4 distinction. L5 uses the chain
    // parser so the flag is inert there.
    var p_l4: u32 = if (level >= 4) 1 else 0;

    var params = [_]?*anyopaque{
        @ptrCast(&p_input),
        @ptrCast(&p_output),
        @ptrCast(&p_descs),
        @ptrCast(&p_global_hash),
        @ptrCast(&p_sizes),
        @ptrCast(&p_total),
        @ptrCast(&p_hash_bits),
        @ptrCast(&p_use_chain),
        @ptrCast(&p_l4),
    };
    var extra = [_]?*anyopaque{null};

    const shared_bytes: u32 = if (global or chain) 0 else @intCast(hash_size * 4);
    if (launch_fn(kernel_fn, num_chunks, 1, 1, 32, 1, 1, shared_bytes, 0, &params, &extra) != CUDA_SUCCESS)
        return false;

    if (sync_fn() != CUDA_SUCCESS) return false;

    if (t_before) |t_start| {
        if (io) |io_val| {
            last_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
        }
    }

    // Download comp_sizes first, then only the actual compressed bytes per block
    _ = d2h_fn(@ptrCast(comp_sizes_out.ptr), d_sizes, sizes_bytes);

    for (0..chunk_descs.len) |i| {
        const cs = comp_sizes_out[i];
        if (cs > 0) {
            const dst_off = chunk_descs[i].dst_offset;
            _ = d2h_fn(@ptrCast(output.ptr + dst_off), d_output + dst_off, cs);
        }
    }

    // ── GPU tANS entropy encoding on literal streams ──
    if (level >= 3 and tans_kernel_fn != 0) tans_pass: {
        const allocator = std.heap.page_allocator;

        // Parse literal offsets/sizes from the downloaded sub-chunk headers
        var tans_descs_host = allocator.alloc(TansChunkDesc, num_chunks) catch break :tans_pass;
        defer allocator.free(tans_descs_host);
        var lit_dst_offsets = allocator.alloc(u32, num_chunks) catch break :tans_pass;
        // NOT deferred — frame assembler reads these via tans_lit_offsets

        const tans_cap_per: usize = 65536 * 2; // 2x literal size for temp staging in assembly
        var total_tans_dst: usize = 0;

        for (0..chunk_descs.len) |i| {
            const cs = comp_sizes_out[i];
            const init_bytes: usize = if (chunk_descs[i].is_first != 0) 8 else 0;
            const src_off = chunk_descs[i].dst_offset;

            // Parse 3-byte BE literal count from the downloaded output
            var lit_count: u32 = 0;
            if (cs > init_bytes + 3) {
                const hdr_base = src_off + init_bytes;
                lit_count = (@as(u32, output[hdr_base]) << 16) |
                    (@as(u32, output[hdr_base + 1]) << 8) |
                    @as(u32, output[hdr_base + 2]);
            }

            // Literal data starts right after the 3-byte header
            const lit_offset_in_output: u32 = @intCast(src_off + init_bytes + 3);
            lit_dst_offsets[i] = @intCast(total_tans_dst);

            tans_descs_host[i] = .{
                .src_offset = lit_offset_in_output, // offset into d_output
                .src_size = lit_count,
                .dst_offset = @intCast(total_tans_dst),
                .dst_capacity = @intCast(@min(tans_cap_per, @max(lit_count * 2, lit_count + 4096))),
            };
            total_tans_dst += @min(tans_cap_per, @max(lit_count * 2, lit_count + 4096));
        }

        if (total_tans_dst == 0) break :tans_pass;

        // Allocate device buffers for tANS
        const tans_descs_bytes = num_chunks * @sizeOf(TansChunkDesc);
        const tans_sizes_bytes = @as(usize, num_chunks) * 4;
        if (!ensureBuf(&d_tans_descs_persist, &d_tans_descs_size, tans_descs_bytes)) break :tans_pass;
        if (!ensureBuf(&d_tans_out_persist, &d_tans_out_size, total_tans_dst)) break :tans_pass;
        if (!ensureBuf(&d_tans_sizes_persist, &d_tans_sizes_size, tans_sizes_bytes)) break :tans_pass;

        // Upload descriptors, zero sizes
        _ = h2d_fn(d_tans_descs_persist, @ptrCast(tans_descs_host.ptr), tans_descs_bytes);
        if (cuMemsetD8_fn) |memset_fn| _ = memset_fn(d_tans_sizes_persist, 0, tans_sizes_bytes);
        _ = sync_fn();

        // Launch tANS kernel: source is d_output (literal data already there from LZ kernel)
        var tp_src = d_output; // literals are in the LZ output buffer
        var tp_dst = d_tans_out_persist;
        var tp_descs = d_tans_descs_persist;
        var tp_sizes = d_tans_sizes_persist;
        var tp_total = num_chunks;

        var tans_params = [_]?*anyopaque{
            @ptrCast(&tp_src),
            @ptrCast(&tp_dst),
            @ptrCast(&tp_descs),
            @ptrCast(&tp_sizes),
            @ptrCast(&tp_total),
        };
        var tans_extra = [_]?*anyopaque{null};

        if (launch_fn(tans_kernel_fn, num_chunks, 1, 1, 32, 1, 1, 0, 0, &tans_params, &tans_extra) != CUDA_SUCCESS)
            break :tans_pass;

        if (sync_fn() != CUDA_SUCCESS) break :tans_pass;

        // Download tANS sizes and encoded data
        const h_tans_sizes = allocator.alloc(u32, num_chunks) catch break :tans_pass;
        const h_tans_data = allocator.alloc(u8, total_tans_dst) catch break :tans_pass;
        _ = d2h_fn(@ptrCast(h_tans_sizes.ptr), d_tans_sizes_persist, tans_sizes_bytes);
        _ = d2h_fn(@ptrCast(h_tans_data.ptr), d_tans_out_persist, total_tans_dst);

        // Store for frame assembler
        tans_lit_sizes = h_tans_sizes;
        tans_lit_data = h_tans_data;
        tans_lit_offsets = lit_dst_offsets;
    }

    return true;
}
