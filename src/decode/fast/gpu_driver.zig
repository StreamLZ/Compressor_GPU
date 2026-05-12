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
pub var last_tans_kernel_ns: i64 = 0;

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

// Token tANS scratch + descriptors
var d_tans_tok_scratch: CUdeviceptr = 0;
var d_tans_tok_scratch_size: usize = 0;
var d_tans_tok_descs_persist: CUdeviceptr = 0;
var d_tans_tok_descs_persist_size: usize = 0;
var d_tans_tok_status_persist: CUdeviceptr = 0;
var d_tans_tok_status_persist_size: usize = 0;

// Off16 scratch (hi bytes at chunk_idx*65536, lo bytes at chunk_idx*65536+32768)
var d_tans_off16_scratch: CUdeviceptr = 0;
var d_tans_off16_scratch_size: usize = 0;

// Off16 hi/lo tANS scratch + descriptors
var d_tans_off16hi_descs: CUdeviceptr = 0;
var d_tans_off16hi_descs_size: usize = 0;
var d_tans_off16hi_status: CUdeviceptr = 0;
var d_tans_off16hi_status_size: usize = 0;
var d_tans_off16lo_descs: CUdeviceptr = 0;
var d_tans_off16lo_descs_size: usize = 0;
var d_tans_off16lo_status: CUdeviceptr = 0;
var d_tans_off16lo_status_size: usize = 0;

// Host-side tANS descriptor buffers (avoids 64KB stack allocations)
var tans_host_buf: [4096]TansDecChunkDesc = undefined;
var tans_tok_host_buf: [4096]TansDecChunkDesc = undefined;
var tans_off16hi_host_buf: [4096]TansDecChunkDesc = undefined;
var tans_off16lo_host_buf: [4096]TansDecChunkDesc = undefined;
var raw_off16_buf: [8192]RawOff16Desc = undefined;

// ── tANS header scanning ───────────────────────────────────────
// Scans the host-side compressed data to find tANS literal, token,
// and off16 hi/lo headers. Builds TansDecChunkDesc descriptors for
// the tANS kernel. Raw (type 0) off16 sub-streams are recorded
// for direct H2D copy.

// Describes a raw (type 0 memcpy) off16 sub-stream that needs H2D copy
const RawOff16Desc = struct {
    src_offset: u32, // offset of raw bytes in compressed buffer
    size: u32, // number of bytes
    gpu_offset: u32, // offset in d_tans_off16_scratch
};

const ScanResult = struct {
    num_lit: u32,
    num_tok: u32,
    num_off16hi: u32,
    num_off16lo: u32,
    num_raw_off16: u32,
};

fn scanForTansChunks(
    chunk_descs: []const ChunkDesc,
    compressed_block: []const u8,
    sub_chunk_cap: u32,
    tans_lit_descs: []TansDecChunkDesc,
    tans_tok_descs: []TansDecChunkDesc,
    tans_off16hi_descs: []TansDecChunkDesc,
    tans_off16lo_descs: []TansDecChunkDesc,
    raw_off16_descs: []RawOff16Desc,
) ScanResult {
    _ = sub_chunk_cap;
    var num_lit: u32 = 0;
    var num_tok: u32 = 0;
    var num_off16hi: u32 = 0;
    var num_off16lo: u32 = 0;
    var num_raw: u32 = 0;

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

        // Walk streams: first sub-chunk always has base_offset == 0 → 8 init bytes
        const init_bytes: u32 = 8;
        var pos: u32 = 3 + init_bytes;

        if (pos >= chunk_src.len) continue;

        // ── Stream 1: Literals ──
        const lit_first = chunk_src[pos];
        const lit_type = (lit_first >> 4) & 0x7;
        if (lit_type == 1) {
            _ = parseTansHeader(chunk_src, pos, ch.src_offset, chunk_idx, tans_lit_descs, &num_lit);
        }
        const lit_next = skipStreamHeader(chunk_src, pos);
        if (lit_next == null) continue;
        pos = lit_next.?;

        if (pos >= chunk_src.len) continue;

        // ── Stream 2: Tokens (command stream) ──
        const tok_first = chunk_src[pos];
        const tok_type = (tok_first >> 4) & 0x7;
        if (tok_type == 1) {
            _ = parseTansHeader(chunk_src, pos, ch.src_offset, chunk_idx, tans_tok_descs, &num_tok);
        }
        const tok_next = skipStreamHeader(chunk_src, pos);
        if (tok_next == null) continue;
        pos = tok_next.?;

        if (pos >= chunk_src.len) continue;

        // Skip cmd_stream2_offset if sub-chunk > 64KB
        if (ch.decomp_size > 0x10000) {
            if (pos + 2 > chunk_src.len) continue;
            pos += 2;
        }

        // ── Off16 stream ──
        if (pos + 2 > chunk_src.len) continue;
        const off16_count: u32 = @as(u32, chunk_src[pos]) | (@as(u32, chunk_src[pos + 1]) << 8);
        if (off16_count != 0xFFFF) continue; // not entropy-coded
        pos += 2;

        // Entropy-coded off16: two sub-streams (hi bytes, lo bytes)
        if (pos >= chunk_src.len) continue;

        // ── Off16 hi stream ──
        const hi_first = chunk_src[pos];
        const hi_type = (hi_first >> 4) & 0x7;
        if (hi_type == 1) {
            // tANS-encoded hi stream: dst goes to chunk_idx * 65536 (first half)
            _ = parseTansHeaderWithDstOffset(chunk_src, pos, ch.src_offset, chunk_idx * 65536, tans_off16hi_descs, &num_off16hi);
        } else if (hi_type == 0) {
            // Raw memcpy hi stream: record for H2D copy
            const raw_info = parseType0StreamInfo(chunk_src, pos);
            if (raw_info.data_offset != 0 and num_raw < raw_off16_descs.len) {
                raw_off16_descs[num_raw] = .{
                    .src_offset = ch.src_offset + raw_info.data_offset,
                    .size = raw_info.size,
                    .gpu_offset = @intCast(chunk_idx * 65536),
                };
                num_raw += 1;
            }
        }
        const hi_next = skipStreamHeader(chunk_src, pos);
        if (hi_next == null) continue;
        pos = hi_next.?;

        if (pos >= chunk_src.len) continue;

        // ── Off16 lo stream ──
        const lo_first = chunk_src[pos];
        const lo_type = (lo_first >> 4) & 0x7;
        if (lo_type == 1) {
            // tANS-encoded lo stream: dst goes to chunk_idx * 65536 + 32768 (second half)
            _ = parseTansHeaderWithDstOffset(chunk_src, pos, ch.src_offset, chunk_idx * 65536 + 32768, tans_off16lo_descs, &num_off16lo);
        } else if (lo_type == 0) {
            // Raw memcpy lo stream: record for H2D copy
            const raw_info = parseType0StreamInfo(chunk_src, pos);
            if (raw_info.data_offset != 0 and num_raw < raw_off16_descs.len) {
                raw_off16_descs[num_raw] = .{
                    .src_offset = ch.src_offset + raw_info.data_offset,
                    .size = raw_info.size,
                    .gpu_offset = @intCast(chunk_idx * 65536 + 32768),
                };
                num_raw += 1;
            }
        }
    }

    return .{ .num_lit = num_lit, .num_tok = num_tok, .num_off16hi = num_off16hi, .num_off16lo = num_off16lo, .num_raw_off16 = num_raw };
}

/// Parse a type 0 (memcpy) stream header, returning the data offset
/// (relative to chunk start) and the raw byte count.
const Type0Info = struct { data_offset: u32, size: u32 };
fn parseType0StreamInfo(chunk_src: []const u8, pos: u32) Type0Info {
    if (pos >= chunk_src.len) return .{ .data_offset = 0, .size = 0 };
    const first_byte = chunk_src[pos];
    if (first_byte >= 0x80) {
        if (pos + 2 > chunk_src.len) return .{ .data_offset = 0, .size = 0 };
        const sz: u32 = ((@as(u32, chunk_src[pos]) << 8) | @as(u32, chunk_src[pos + 1])) & 0xFFF;
        return .{ .data_offset = pos + 2, .size = sz };
    } else {
        if (pos + 3 > chunk_src.len) return .{ .data_offset = 0, .size = 0 };
        const sz: u32 = (@as(u32, chunk_src[pos]) << 16) | (@as(u32, chunk_src[pos + 1]) << 8) | @as(u32, chunk_src[pos + 2]);
        return .{ .data_offset = pos + 3, .size = sz };
    }
}

/// Skip past an entropy-coded stream header + payload, returning
/// the new position, or null if the data is truncated.
fn skipStreamHeader(chunk_src: []const u8, pos: u32) ?u32 {
    if (pos >= chunk_src.len) return null;
    const first_byte = chunk_src[pos];
    const ct = (first_byte >> 4) & 0x7;

    if (ct == 0) {
        // Type 0: memcpy, 2 or 3 byte header
        if (first_byte >= 0x80) {
            if (pos + 2 > chunk_src.len) return null;
            const sz: u32 = ((@as(u32, chunk_src[pos]) << 8) | @as(u32, chunk_src[pos + 1])) & 0xFFF;
            return pos + 2 + sz;
        } else {
            if (pos + 3 > chunk_src.len) return null;
            const sz: u32 = (@as(u32, chunk_src[pos]) << 16) | (@as(u32, chunk_src[pos + 1]) << 8) | @as(u32, chunk_src[pos + 2]);
            return pos + 3 + sz;
        }
    } else if (ct == 1 or ct == 2 or ct == 4) {
        // tANS (1) or Huffman (2, 4): 3 or 5 byte header + compressed payload
        if (first_byte >= 0x80) {
            if (pos + 3 > chunk_src.len) return null;
            const bits: u32 = (@as(u32, chunk_src[pos]) << 16) | (@as(u32, chunk_src[pos + 1]) << 8) | @as(u32, chunk_src[pos + 2]);
            const comp_size = bits & 0x3FF;
            return pos + 3 + comp_size;
        } else {
            if (pos + 5 > chunk_src.len) return null;
            const bits: u32 = (@as(u32, chunk_src[pos + 1]) << 24) | (@as(u32, chunk_src[pos + 2]) << 16) | (@as(u32, chunk_src[pos + 3]) << 8) | @as(u32, chunk_src[pos + 4]);
            const comp_size = bits & 0x3FFFF;
            return pos + 5 + comp_size;
        }
    }
    return null;
}

fn parseTansHeaderWithDstOffset(
    chunk_src: []const u8,
    lit_off: u32,
    src_offset_base: u32,
    dst_offset: usize,
    tans_descs_out: []TansDecChunkDesc,
    num_tans: *u32,
) bool {
    if (lit_off >= chunk_src.len) return false;
    const first_byte = chunk_src[lit_off];

    if (first_byte >= 0x80) {
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
                .dst_offset = @intCast(dst_offset),
                .dst_size = @intCast(tans_dst_size),
            };
            num_tans.* += 1;
            return true;
        }
    } else {
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
                .dst_offset = @intCast(dst_offset),
                .dst_size = @intCast(tans_dst_size),
            };
            num_tans.* += 1;
            return true;
        }
    }
    return false;
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

    // Allocate unified tANS scratch: 3 regions × 65536 bytes per chunk
    // Layout: [lit: N*64K] [tok: N*64K] [off16: N*64K]
    const per_chunk_scratch: usize = 65536;
    const tans_scratch_bytes = chunk_descs.len * per_chunk_scratch * 3;
    if (!ensureDeviceBuf(&d_tans_scratch, &d_tans_scratch_size, tans_scratch_bytes)) return error.BadMode;
    const tok_offset = chunk_descs.len * per_chunk_scratch;
    const off16_offset = chunk_descs.len * per_chunk_scratch * 2;
    d_tans_tok_scratch = d_tans_scratch + tok_offset;
    d_tans_off16_scratch = d_tans_scratch + off16_offset;

    if (compressed_block.len > 0)
        _ = h2d_fn(d_comp_persist, @ptrCast(compressed_block.ptr), compressed_block.len);
    _ = h2d_fn(d_descs_persist, @ptrCast(chunk_descs.ptr), desc_bytes);
    _ = sync_fn();

    // ── Pass 1: tANS decode (literals + tokens + off16) ─────────
    if (tans_kernel_fn != 0) {
        const max_tans = @min(chunk_descs.len, tans_host_buf.len);

        const scan = scanForTansChunks(
            chunk_descs,
            compressed_block,
            sub_chunk_cap,
            tans_host_buf[0..max_tans],
            tans_tok_host_buf[0..max_tans],
            tans_off16hi_host_buf[0..max_tans],
            tans_off16lo_host_buf[0..max_tans],
            &raw_off16_buf,
        );

        // Merge all tANS descriptors into one array, adjust offsets for unified scratch
        // Tok descriptors: add tok_offset to dst_offset
        // Off16 descriptors: add off16_offset to dst_offset
        var merged_buf: [4096 * 4]TansDecChunkDesc = undefined;
        var merged_count: u32 = 0;

        for (0..scan.num_lit) |i| {
            merged_buf[merged_count] = tans_host_buf[i];
            merged_count += 1;
        }
        for (0..scan.num_tok) |i| {
            var d = tans_tok_host_buf[i];
            d.dst_offset += @intCast(tok_offset);
            merged_buf[merged_count] = d;
            merged_count += 1;
        }
        for (0..scan.num_off16hi) |i| {
            var d = tans_off16hi_host_buf[i];
            d.dst_offset += @intCast(off16_offset);
            merged_buf[merged_count] = d;
            merged_count += 1;
        }
        for (0..scan.num_off16lo) |i| {
            var d = tans_off16lo_host_buf[i];
            d.dst_offset += @intCast(off16_offset);
            merged_buf[merged_count] = d;
            merged_count += 1;
        }

        // Dump descriptors + compressed data for standalone profiling
        if (merged_count > 0) {
            const dump_c = @cImport({ @cInclude("stdlib.h"); });
            if (dump_c.getenv("SLZ_DUMP_TANS") != null) {
                const cio = @cImport({ @cInclude("stdio.h"); });
                const fd = cio.fopen("c:\\tmp\\tans_descs.bin", "wb");
                if (fd != null) {
                    _ = cio.fwrite(&merged_buf, @sizeOf(TansDecChunkDesc), merged_count, fd);
                    _ = cio.fclose(fd);
                }
                const fc = cio.fopen("c:\\tmp\\tans_compressed.bin", "wb");
                if (fc != null) {
                    _ = cio.fwrite(compressed_block.ptr, 1, compressed_block.len, fc);
                    _ = cio.fclose(fc);
                }
                std.debug.print("Dumped {} tANS descs + {} bytes compressed to c:\\tmp\n", .{ merged_count, compressed_block.len });
            }
        }

        // Upload tANS descriptors to GPU (before timer)
        if (merged_count > 0) {
            const mdesc_bytes = merged_count * @sizeOf(TansDecChunkDesc);
            const mstatus_bytes = merged_count * @sizeOf(u32);
            if (!ensureDeviceBuf(&d_tans_descs_persist, &d_tans_descs_persist_size, mdesc_bytes))
                return error.BadMode;
            if (!ensureDeviceBuf(&d_tans_status_persist, &d_tans_status_persist_size, mstatus_bytes))
                return error.BadMode;
            _ = h2d_fn(d_tans_descs_persist, @ptrCast(@as([*]TansDecChunkDesc, &merged_buf)), mdesc_bytes);
            if (cuMemsetD8_fn) |memset_fn| _ = memset_fn(d_tans_status_persist, 0, mstatus_bytes);
        }

        // Upload raw (type 0) off16 sub-streams to GPU (before timer)
        for (0..scan.num_raw_off16) |ri| {
            const rd = raw_off16_buf[ri];
            if (rd.size > 0 and rd.src_offset + rd.size <= compressed_block.len) {
                _ = h2d_fn(d_tans_off16_scratch + rd.gpu_offset, @ptrCast(compressed_block.ptr + rd.src_offset), rd.size);
            }
        }

        _ = sync_fn();
    }

    // ── KERNEL TIMER: only pure GPU kernel time from here ──
    const t_before_kern = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

    // Launch tANS kernel
    if (tans_kernel_fn != 0 and d_tans_descs_persist_size > 0) {
        const tans_count = @as(u32, @intCast(d_tans_descs_persist_size / @sizeOf(TansDecChunkDesc)));
        if (tans_count > 0) {
            var tp_comp = d_comp_persist;
            var tp_scratch = d_tans_scratch;
            var tp_descs = d_tans_descs_persist;
            var tp_status = d_tans_status_persist;
            var tp_num = tans_count;
            var tans_params = [_]?*anyopaque{
                @ptrCast(&tp_comp), @ptrCast(&tp_scratch),
                @ptrCast(&tp_descs), @ptrCast(&tp_status), @ptrCast(&tp_num),
            };
            var tans_extra = [_]?*anyopaque{null};
            const tans_grid = (tans_count + 1) / 2;
            if (launch_fn(tans_kernel_fn, tans_grid, 1, 1, 32, 2, 1, 0, 0, &tans_params, &tans_extra) != CUDA_SUCCESS)
                return error.BadMode;
        }
    }

    if (sync_fn() != CUDA_SUCCESS) return error.BadMode;

    if (io) |io_val| {
        if (t_before_kern) |t_start| {
            last_tans_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
        }
    }

    // ── Pass 2: LZ decode ──────────────────────────────────────

    var p_comp = d_comp_persist;
    var p_descs_dev = d_descs_persist;
    var p_dst = d_output;
    var p_cpg = chunks_per_group;
    var p_total: u32 = @intCast(chunk_descs.len);
    var p_sc_cap = sub_chunk_cap;
    var p_tans_scratch = d_tans_scratch;
    var p_tans_tok_scratch = d_tans_tok_scratch;
    var p_tans_off16_scratch = d_tans_off16_scratch;

    var params = [_]?*anyopaque{
        @ptrCast(&p_comp),
        @ptrCast(&p_descs_dev),
        @ptrCast(&p_dst),
        @ptrCast(&p_cpg),
        @ptrCast(&p_total),
        @ptrCast(&p_sc_cap),
        @ptrCast(&p_tans_scratch),
        @ptrCast(&p_tans_tok_scratch),
        @ptrCast(&p_tans_off16_scratch),
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

// ── Helper: launch tANS kernel for a set of descriptors ───────
// When skip_post_sync is true, the kernel is launched without a trailing
// cuCtxSynchronize — the caller must sync before reading results.
fn launchTansKernel(
    h2d_fn: FnMemcpyHtoD,
    launch_fn: FnLaunchKernel,
    _: FnCtxSync,
    num_descs: u32,
    host_descs: [*]TansDecChunkDesc,
    dst_scratch: CUdeviceptr,
    d_descs: *CUdeviceptr,
    d_descs_size: *usize,
    d_status: *CUdeviceptr,
    d_status_size: *usize,
) !void {
    const desc_bytes = num_descs * @sizeOf(TansDecChunkDesc);
    const status_bytes = num_descs * @sizeOf(u32);

    if (!ensureDeviceBuf(d_descs, d_descs_size, desc_bytes))
        return error.BadMode;
    if (!ensureDeviceBuf(d_status, d_status_size, status_bytes))
        return error.BadMode;

    _ = h2d_fn(d_descs.*, @ptrCast(host_descs), desc_bytes);

    if (cuMemsetD8_fn) |memset_fn| {
        _ = memset_fn(d_status.*, 0, status_bytes);
    }

    var tp_comp = d_comp_persist;
    var tp_scratch = dst_scratch;
    var tp_descs = d_descs.*;
    var tp_status = d_status.*;
    var tp_num: u32 = num_descs;

    var tans_params = [_]?*anyopaque{
        @ptrCast(&tp_comp),
        @ptrCast(&tp_scratch),
        @ptrCast(&tp_descs),
        @ptrCast(&tp_status),
        @ptrCast(&tp_num),
    };
    var tans_extra = [_]?*anyopaque{null};

    const tans_grid = (num_descs + 1) / 2;
    if (launch_fn(tans_kernel_fn, tans_grid, 1, 1, 32, 2, 1, 0, 0, &tans_params, &tans_extra) != CUDA_SUCCESS)
        return error.BadMode;
}

// ── CPU off16 entropy decode + GPU upload ─────────────────────
// Walks each chunk looking for 0xFFFF off16 marker. When found,
// decodes hi+lo byte streams using the CPU entropy decoder,
// interleaves into u16 pairs, and uploads to tans_off16_scratch
// at the chunk's slot (chunk_idx * 65536).
fn cpuDecodeOff16Streams(
    chunk_descs: []const ChunkDesc,
    compressed_block: []const u8,
    h2d_fn: FnMemcpyHtoD,
    sync_fn: FnCtxSync,
) !void {
    const entropy = @import("../entropy/entropy_decoder.zig");
    const constants = @import("../../format/streamlz_constants.zig");
    const alloc = std.heap.page_allocator;

    // Allocate scratch buffer once, reuse across chunks
    const scratch_buf = alloc.alloc(u8, constants.scratch_size) catch return;
    defer alloc.free(scratch_buf);
    const scratch_ptr: [*]u8 = scratch_buf.ptr;
    const scratch_end: [*]u8 = scratch_ptr + scratch_buf.len;

    var any_uploaded = false;

    for (chunk_descs, 0..) |ch, chunk_idx| {
        if (ch.flags != 0) continue;
        if (ch.decomp_size == 0) continue;
        if (ch.src_offset >= compressed_block.len) continue;

        const chunk_end = @min(ch.src_offset + ch.comp_size, @as(u32, @intCast(compressed_block.len)));
        const chunk_src = compressed_block[ch.src_offset..chunk_end];
        if (chunk_src.len < 3) continue;

        const chunkhdr: u32 = (@as(u32, chunk_src[0]) << 16) |
            (@as(u32, chunk_src[1]) << 8) |
            @as(u32, chunk_src[2]);
        if ((chunkhdr & 0x800000) == 0) continue;

        // Walk to off16: skip sub-chunk header (3) + init bytes (8) + literal + token + cmd_stream2_offset
        var pos: u32 = 3 + 8; // sub-chunk hdr + init bytes
        if (pos >= chunk_src.len) continue;

        // Skip literal stream
        const lit_next = skipStreamHeader(chunk_src, pos);
        if (lit_next == null) continue;
        pos = lit_next.?;
        if (pos >= chunk_src.len) continue;

        // Skip token stream
        const tok_next = skipStreamHeader(chunk_src, pos);
        if (tok_next == null) continue;
        pos = tok_next.?;
        if (pos >= chunk_src.len) continue;

        // Skip cmd_stream2_offset if sub-chunk > 64KB
        if (ch.decomp_size > 0x10000) {
            if (pos + 2 > chunk_src.len) continue;
            pos += 2;
        }

        // Off16: read 2-byte count
        if (pos + 2 > chunk_src.len) continue;
        const off16_count: u32 = @as(u32, chunk_src[pos]) | (@as(u32, chunk_src[pos + 1]) << 8);
        if (off16_count != 0xFFFF) continue; // not entropy-coded
        pos += 2;

        // Entropy-coded off16: decode hi stream then lo stream on CPU
        const src_remaining = chunk_src[pos..];

        // Decode hi bytes
        const res_hi = entropy.highDecodeBytes(
            scratch_ptr,
            scratch_buf.len / 2,
            src_remaining,
            false,
            scratch_ptr,
            scratch_end,
        ) catch continue;

        const hi_ptr = res_hi.out_ptr;
        const count = res_hi.decoded_size;

        // Advance scratch if hi was written there
        var scratch_cur = scratch_ptr;
        if (@intFromPtr(res_hi.out_ptr) == @intFromPtr(scratch_ptr)) {
            scratch_cur += res_hi.decoded_size;
        }

        // Decode lo bytes
        const lo_src = src_remaining[res_hi.bytes_consumed..];
        const res_lo = entropy.highDecodeBytes(
            scratch_cur,
            scratch_buf.len / 2,
            lo_src,
            false,
            scratch_cur,
            scratch_end,
        ) catch continue;

        const lo_ptr = res_lo.out_ptr;
        if (res_lo.decoded_size != count) continue; // mismatch

        // Interleave hi+lo into u16 pairs in a host buffer
        const interleaved = alloc.alloc(u8, count * 2) catch continue;
        defer alloc.free(interleaved);

        for (0..count) |i| {
            interleaved[i * 2] = lo_ptr[i];
            interleaved[i * 2 + 1] = hi_ptr[i];
        }

        // Upload to GPU at chunk_idx * 65536 offset in off16 scratch
        const gpu_off = chunk_idx * 65536;
        _ = h2d_fn(d_tans_off16_scratch + gpu_off, @ptrCast(interleaved.ptr), count * 2);
        any_uploaded = true;
    }
    if (any_uploaded) _ = sync_fn();
}
