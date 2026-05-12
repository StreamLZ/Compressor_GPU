//! CUDA Driver API bridge for GPU LZ decompression.
//! Loads nvcuda.dll at runtime — no compile-time CUDA dependency.
//! Pre-compiled PTX is embedded via @embedFile.
//!
//! Two-pass decode pipeline:
//!   Pass 1: slzTansDecodeKernel — decodes tANS literal streams to temp buffer
//!   Pass 2: slzFullDecompressL1Kernel — LZ decode, reads pre-decoded literals

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
var initialized = false;

// ── Driver API function signatures ──────────────────────────────
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

// ── Initialization ──────────────────────────────────────────────

pub fn init() bool {
    if (initialized) return kernel_fn != 0;
    initialized = true;

    lib = win32.LoadLibraryA("nvcuda.dll");
    if (lib == null) return false;

    cuInit_fn = getProc(FnInit, "cuInit");
    cuDeviceGet_fn = getProc(FnDeviceGet, "cuDeviceGet");
    cuCtxCreate_fn = getProc(FnCtxCreate, "cuCtxCreate_v2") orelse getProc(FnCtxCreate, "cuCtxCreate");
    cuModuleLoadData_fn = getProc(FnModuleLoadData, "cuModuleLoadData");
    cuModuleGetFunction_fn = getProc(FnModuleGetFunction, "cuModuleGetFunction");
    cuMemAlloc_fn = getProc(FnMemAlloc, "cuMemAlloc_v2");
    cuMemFree_fn = getProc(FnMemFree, "cuMemFree_v2");
    cuMemcpyHtoD_fn = getProc(FnMemcpyHtoD, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH_fn = getProc(FnMemcpyDtoH, "cuMemcpyDtoH_v2");
    cuLaunchKernel_fn = getProc(FnLaunchKernel, "cuLaunchKernel");
    cuCtxSynchronize_fn = getProc(FnCtxSync, "cuCtxSynchronize");
    cuMemsetD8_fn = getProc(FnMemsetD8, "cuMemsetD8_v2");

    if ((cuInit_fn orelse return false)(0) != CUDA_SUCCESS) return false;

    var dev: CUdevice = 0;
    if ((cuDeviceGet_fn orelse return false)(&dev, 0) != CUDA_SUCCESS) return false;

    cuCtxCreate_fn = getProc(FnCtxCreate, "cuCtxCreate_v2");
    if (cuCtxCreate_fn == null) cuCtxCreate_fn = getProc(FnCtxCreate, "cuCtxCreate");
    if ((cuCtxCreate_fn orelse return false)(&ctx, 0, dev) != CUDA_SUCCESS) return false;

    const load_fn = cuModuleLoadData_fn orelse return false;
    const get_fn = cuModuleGetFunction_fn orelse return false;

    // Load LZ decode kernel (Pass 2)
    const ptx = @embedFile("gpu_decode_kernel.ptx") ++ "\x00";
    if (load_fn(&module, ptx.ptr) != CUDA_SUCCESS) return false;
    if (get_fn(&kernel_fn, module, "slzFullDecompressL1Kernel") != CUDA_SUCCESS) return false;

    // Load tANS decode kernel (Pass 1)
    const tans_ptx = @embedFile("gpu_tans_decode_kernel.ptx") ++ "\x00";
    if (load_fn(&tans_module, tans_ptx.ptr) == CUDA_SUCCESS) {
        _ = get_fn(&tans_kernel_fn, tans_module, "slzTansDecodeKernel");
    }

    return true;
}

pub fn isAvailable() bool {
    return init();
}

// ── Device buffer management ────────────────────────────────────

const fast_dec = @import("fast_lz_decoder.zig");

var d_output: CUdeviceptr = 0;
var d_output_size: usize = 0;
var d_comp_persist: CUdeviceptr = 0;
var d_comp_persist_size: usize = 0;
var d_descs_persist: CUdeviceptr = 0;
var d_descs_persist_size: usize = 0;

pub var last_kernel_ns: i64 = 0;

fn ensureDeviceBuf(ptr: *CUdeviceptr, current_size: *usize, needed: usize) bool {
    if (current_size.* >= needed) return true;
    if (ptr.* != 0) _ = (cuMemFree_fn orelse return false)(ptr.*);
    current_size.* = 0;
    if ((cuMemAlloc_fn orelse return false)(ptr, needed) != CUDA_SUCCESS) return false;
    current_size.* = needed;
    return true;
}

fn ensureDeviceOutput(size: usize) bool {
    return ensureDeviceBuf(&d_output, &d_output_size, size);
}

// ── Full GPU decode ─────────────────────────────────────────────
// Two-pass pipeline:
//   Pass 1: Launch tANS decode kernel for chunks with tANS literals
//   Pass 2: Launch LZ decode kernel (reads pre-decoded tANS literals)

pub const ChunkDesc = extern struct {
    src_offset: u32,
    comp_size: u32,
    decomp_size: u32,
    dst_offset: u32,
    flags: u32,
    memset_fill: u8,
    _pad: [3]u8 = .{ 0, 0, 0 },
};

// tANS chunk descriptor — matches gpu_tans_decode_kernel.cu TansDecChunkDesc
const TansDecChunkDesc = extern struct {
    src_offset: u32, // offset of tANS payload in compressed buffer
    src_size: u32, // tANS compressed size
    dst_offset: u32, // offset in tans_scratch (chunk_idx * 65536)
    dst_size: u32, // expected decompressed literal count
};

var d_tans_scratch: CUdeviceptr = 0;
var d_tans_scratch_size: usize = 0;
var d_tans_descs_persist: CUdeviceptr = 0;
var d_tans_descs_persist_size: usize = 0;
var d_tans_status_persist: CUdeviceptr = 0;
var d_tans_status_persist_size: usize = 0;

// ── tANS header scanning ───────────────────────────────────────
// Scans the host-side compressed data to find tANS literal headers
// and builds TansDecChunkDesc descriptors for the tANS decode kernel.

fn scanForTansChunks(
    chunk_descs: []const ChunkDesc,
    compressed_block: []const u8,
    sub_chunk_cap: u32,
    tans_descs_out: []TansDecChunkDesc,
) u32 {
    _ = sub_chunk_cap;
    var num_tans: u32 = 0;

    for (chunk_descs, 0..) |ch, chunk_idx| {
        // Skip non-LZ chunks
        if (ch.flags != 0) continue;
        if (ch.decomp_size == 0) continue;
        if (ch.src_offset >= compressed_block.len) continue;

        const chunk_end = @min(ch.src_offset + ch.comp_size, @as(u32, @intCast(compressed_block.len)));
        const chunk_src = compressed_block[ch.src_offset..chunk_end];
        if (chunk_src.len < 3) continue;

        // Parse first sub-chunk header (3-byte big-endian)
        const chunkhdr: u32 = (@as(u32, chunk_src[0]) << 16) |
            (@as(u32, chunk_src[1]) << 8) |
            @as(u32, chunk_src[2]);

        if ((chunkhdr & 0x800000) == 0) continue; // non-LZ

        // Payload starts after the 3-byte sub-chunk header
        // First sub-chunk of chunk always has base_offset == 0,
        // which means 8 init bytes are present
        const init_bytes: u32 = 8;
        const lit_off: u32 = 3 + init_bytes; // offset within chunk_src

        if (lit_off >= chunk_src.len) continue;

        // Check for tANS header at expected position
        const first_byte = chunk_src[lit_off];
        const chunk_type = (first_byte >> 4) & 0x7;

        if (chunk_type != 1) {
            // Also try without init bytes (in case init bytes were not written)
            const lit_off_no_init: u32 = 3;
            if (lit_off_no_init < chunk_src.len) {
                const fb2 = chunk_src[lit_off_no_init];
                const ct2 = (fb2 >> 4) & 0x7;
                if (ct2 == 1) {
                    // tANS at offset 3 (no init bytes)
                    if (parseTansHeader(chunk_src, lit_off_no_init, ch.src_offset, chunk_idx, tans_descs_out, &num_tans))
                        continue;
                }
            }
            continue;
        }

        // Parse tANS header at the expected position
        _ = parseTansHeader(chunk_src, lit_off, ch.src_offset, chunk_idx, tans_descs_out, &num_tans);
    }

    return num_tans;
}

fn parseTansHeader(
    chunk_src: []const u8,
    lit_off: u32,
    src_offset_base: u32,
    chunk_idx: usize,
    tans_descs_out: []TansDecChunkDesc,
    num_tans: *u32,
) bool {
    if (lit_off >= chunk_src.len) return false;
    const first_byte = chunk_src[lit_off];

    if (first_byte >= 0x80) {
        // Compact 3-byte header
        if (lit_off + 3 > chunk_src.len) return false;
        const hdr3: u32 = (@as(u32, chunk_src[lit_off]) << 16) |
            (@as(u32, chunk_src[lit_off + 1]) << 8) |
            @as(u32, chunk_src[lit_off + 2]);
        const tans_comp_size = hdr3 & 0x3FF;
        const tans_dst_size = tans_comp_size + ((hdr3 >> 10) & 0x3FF) + 1;
        const tans_payload_off = src_offset_base + lit_off + 3;

        if (num_tans.* < tans_descs_out.len) {
            tans_descs_out[num_tans.*] = .{
                .src_offset = @intCast(tans_payload_off),
                .src_size = @intCast(tans_comp_size),
                .dst_offset = @intCast(chunk_idx * 65536),
                .dst_size = @intCast(tans_dst_size),
            };
            num_tans.* += 1;
            return true;
        }
    } else {
        // Non-compact 5-byte header
        if (lit_off + 5 > chunk_src.len) return false;
        const bits: u32 = (@as(u32, chunk_src[lit_off + 1]) << 24) |
            (@as(u32, chunk_src[lit_off + 2]) << 16) |
            (@as(u32, chunk_src[lit_off + 3]) << 8) |
            @as(u32, chunk_src[lit_off + 4]);
        const tans_comp_size = bits & 0x3FFFF;
        const tans_dst_size = (((bits >> 18) | (@as(u32, chunk_src[lit_off]) << 14)) & 0x3FFFF) + 1;
        const tans_payload_off = src_offset_base + lit_off + 5;

        if (num_tans.* < tans_descs_out.len) {
            tans_descs_out[num_tans.*] = .{
                .src_offset = @intCast(tans_payload_off),
                .src_size = @intCast(tans_comp_size),
                .dst_offset = @intCast(chunk_idx * 65536),
                .dst_size = @intCast(tans_dst_size),
            };
            num_tans.* += 1;
            return true;
        }
    }
    return false;
}

pub fn fullGpuLaunch(
    chunk_descs: []const ChunkDesc,
    compressed_block: []const u8,
    dst_full: [*]u8,
    dst_start_off: usize,
    decompressed_size: usize,
    num_groups: u32,
    chunks_per_group: u32,
    sub_chunk_cap: u32,
    io: ?std.Io,
) fast_dec.DecodeError!void {
    if (!init() or kernel_fn == 0) return error.BadMode;

    const h2d_fn = cuMemcpyHtoD_fn orelse return error.BadMode;
    const d2h_fn = cuMemcpyDtoH_fn orelse return error.BadMode;
    const launch_fn = cuLaunchKernel_fn orelse return error.BadMode;
    const sync_fn = cuCtxSynchronize_fn orelse return error.BadMode;

    const total_output = dst_start_off + decompressed_size;
    if (!ensureDeviceOutput(total_output + 64)) return error.BadMode;

    if (dst_start_off > 0)
        _ = h2d_fn(d_output, @ptrCast(dst_full), dst_start_off);

    const comp_bytes = if (compressed_block.len > 0) compressed_block.len else 4;
    const desc_bytes = chunk_descs.len * @sizeOf(ChunkDesc);

    if (!ensureDeviceBuf(&d_comp_persist, &d_comp_persist_size, comp_bytes)) return error.BadMode;
    if (!ensureDeviceBuf(&d_descs_persist, &d_descs_persist_size, desc_bytes)) return error.BadMode;

    // Allocate tANS scratch: 65536 bytes per chunk for decoded tANS literals
    const tans_scratch_bytes = chunk_descs.len * 65536;
    if (!ensureDeviceBuf(&d_tans_scratch, &d_tans_scratch_size, tans_scratch_bytes)) return error.BadMode;

    if (compressed_block.len > 0)
        _ = h2d_fn(d_comp_persist, @ptrCast(compressed_block.ptr), compressed_block.len);
    _ = h2d_fn(d_descs_persist, @ptrCast(chunk_descs.ptr), desc_bytes);
    _ = sync_fn();

    // ── Pass 1: tANS decode ────────────────────────────────────
    // Scan compressed data for tANS literal headers, launch tANS kernel
    if (tans_kernel_fn != 0) {
        // Stack buffer for tANS descriptors (max one per chunk)
        var tans_descs_buf: [4096]TansDecChunkDesc = undefined;
        const max_tans = @min(chunk_descs.len, tans_descs_buf.len);

        const num_tans = scanForTansChunks(
            chunk_descs,
            compressed_block,
            sub_chunk_cap,
            tans_descs_buf[0..max_tans],
        );

        if (num_tans > 0) {
            const tans_desc_bytes = num_tans * @sizeOf(TansDecChunkDesc);
            const tans_status_bytes = num_tans * @sizeOf(u32);

            if (!ensureDeviceBuf(&d_tans_descs_persist, &d_tans_descs_persist_size, tans_desc_bytes))
                return error.BadMode;
            if (!ensureDeviceBuf(&d_tans_status_persist, &d_tans_status_persist_size, tans_status_bytes))
                return error.BadMode;

            _ = h2d_fn(d_tans_descs_persist, @ptrCast(&tans_descs_buf), tans_desc_bytes);

            // Zero the status buffer
            if (cuMemsetD8_fn) |memset_fn| {
                _ = memset_fn(d_tans_status_persist, 0, tans_status_bytes);
            }
            _ = sync_fn();

            // Launch tANS decode kernel: 1 block per tANS chunk, 32 threads
            var tp_comp = d_comp_persist;
            var tp_scratch = d_tans_scratch;
            var tp_descs = d_tans_descs_persist;
            var tp_status = d_tans_status_persist;
            var tp_num: u32 = num_tans;

            var tans_params = [_]?*anyopaque{
                @ptrCast(&tp_comp),
                @ptrCast(&tp_scratch),
                @ptrCast(&tp_descs),
                @ptrCast(&tp_status),
                @ptrCast(&tp_num),
            };
            var tans_extra = [_]?*anyopaque{null};

            if (launch_fn(tans_kernel_fn, num_tans, 1, 1, 32, 1, 1, 0, 0, &tans_params, &tans_extra) != CUDA_SUCCESS)
                return error.BadMode;

            if (sync_fn() != CUDA_SUCCESS) return error.BadMode;
        }
    }

    // ── Pass 2: LZ decode ──────────────────────────────────────
    const t_before_kern = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

    var p_comp = d_comp_persist;
    var p_descs_dev = d_descs_persist;
    var p_dst = d_output;
    var p_cpg = chunks_per_group;
    var p_total: u32 = @intCast(chunk_descs.len);
    var p_sc_cap = sub_chunk_cap;
    var p_tans_scratch = d_tans_scratch;

    var params = [_]?*anyopaque{
        @ptrCast(&p_comp),
        @ptrCast(&p_descs_dev),
        @ptrCast(&p_dst),
        @ptrCast(&p_cpg),
        @ptrCast(&p_total),
        @ptrCast(&p_sc_cap),
        @ptrCast(&p_tans_scratch),
    };
    var extra = [_]?*anyopaque{null};

    const grid_x = (num_groups + 1) / 2;
    if (launch_fn(kernel_fn, grid_x, 1, 1, 32, 2, 1, 0, 0, &params, &extra) != CUDA_SUCCESS)
        return error.BadMode;

    if (sync_fn() != CUDA_SUCCESS) return error.BadMode;

    if (t_before_kern) |t_start| {
        if (io) |io_val| {
            last_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
        }
    }

    _ = d2h_fn(@ptrCast(dst_full + dst_start_off), d_output + dst_start_off, decompressed_size);
}
