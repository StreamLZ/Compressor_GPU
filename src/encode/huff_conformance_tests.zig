//! Isolated kernel-level conformance tests for the Huffman chain —
//! backport of `srcVK/tests/huff_decode_conformance.zig` (BACKPORTS.md
//! D, wave 2).
//!
//! Drives the CUDA Huffman ENCODER kernels (slzHuffBuildTablesKernel +
//! slzHuffEncode4StreamKernel) on a known input to produce a BIL body,
//! then feeds those exact bytes as INPUTS to the DECODER kernels
//! (slzHuffBuildLutKernel + slzHuffDecode4StreamKernel) and asserts
//! byte-identity vs the original input.
//!
//! Why this exists (the VK lesson): a bitbufRefill hi/lo swap on the VK
//! side silently corrupted every huff_decode output and went undetected
//! because no isolated conformance test existed — e2e roundtrips mask
//! kernel-level bugs whenever encoder and decoder share the same wrong
//! assumption. The CUDA chain has the same blind spot; these cases are
//! the direct gate on the kernel I/O contract.
//!
//! The five cases match the VK suite case-for-case: pseudo-random 1 KiB
//! (smoke), repeating English 8 KiB (skewed-but-shallow tree), 64 KiB
//! random (full Phase-2 interleaved hot loop at the L1 sub-chunk cap),
//! 16 KiB Zipf-ish skew (code lengths spanning 2..11, uneven per-lane
//! byte counts), and 4097 bytes (src_size % 32 != 0 remainder lane).

const std = @import("std");
const testing = std.testing;

const cuda = @import("../decode/cuda_api.zig");
const cuda_ffi = @import("cuda_ffi.zig");
const enc_loader = @import("module_loader.zig");
const dec_loader = @import("../decode/module_loader.zig");
const encode_context = @import("encode_context.zig");
const decode_context = @import("../decode/decode_context.zig");
const descriptors = @import("../decode/descriptors.zig");
const gpu_roundtrip_tests = @import("gpu_roundtrip_tests.zig");

const HuffEncDesc = encode_context.HuffEncDesc;
const HuffDecChunkDesc = descriptors.HuffDecChunkDesc;
const CUDA_SUCCESS = cuda.CUDA_SUCCESS;

/// Mirror of HUFF_BODY_FIXED_BYTES in encode_huff.zig (private there).
/// Keep in sync with src/encode/encode_huff.zig:38.
const HUFF_BODY_FIXED_BYTES: u32 = 356;

/// Encoder lane count — mirrors HUFF_NUM_STREAMS in common/gpu_huffman.cuh.
const NUM_STREAMS: u32 = 32;

const HUFF_LUT_ENTRIES: u32 = @intCast(descriptors.HUFF_LUT_ENTRIES);

/// Mirror of encode_huff.zig::bilDstCap (private there). Keep in sync.
inline fn bilDstCap(count: u32) u32 {
    return (count * 2 + HUFF_BODY_FIXED_BYTES + 3) & ~@as(u32, 3);
}

/// RAII-ish device allocation for the test body: every buffer freed via
/// defer, mirrors the VK test's malloc/free chain.
fn devAlloc(ptr: *u64, bytes: usize) !void {
    const malloc_fn = cuda.cuMemAlloc_fn orelse {
        std.debug.print("huff-conformance gate: cuMemAlloc_fn null\n", .{});
        return error.SkipZigTest;
    };
    const rc = malloc_fn(ptr, bytes);
    if (rc != CUDA_SUCCESS) {
        std.debug.print("huff-conformance gate: cuMemAlloc({d}) rc={d}\n", .{ bytes, rc });
        return error.SkipZigTest;
    }
}

fn devFree(ptr: u64) void {
    if (cuda.cuMemFree_fn) |free_fn| _ = free_fn(ptr);
}

/// Core roundtrip: encoder build-tables + encode-4stream, then decoder
/// build-LUT + decode-4stream, on a single sub-chunk; byte-identity
/// assert. Direct kernel launches — deliberately NOT through the
/// encode/decode context orchestration, so a regression here points
/// at the kernel contract, not the host glue.
fn huffRoundtripOne(allocator: std.mem.Allocator, src: []const u8) !void {
    if (!enc_loader.init()) {
        std.debug.print("huff-conformance gate: enc init failed\n", .{});
        return error.SkipZigTest;
    }
    if (!dec_loader.init()) {
        std.debug.print("huff-conformance gate: dec init failed\n", .{});
        return error.SkipZigTest;
    }
    if (enc_loader.huff_tables_kernel_fn == 0) {
        std.debug.print("huff-conformance gate: huff_tables_kernel_fn == 0\n", .{});
        return error.SkipZigTest;
    }
    if (enc_loader.huff_encode_kernel_fn == 0) {
        std.debug.print("huff-conformance gate: huff_encode_kernel_fn == 0\n", .{});
        return error.SkipZigTest;
    }
    if (dec_loader.huff_build_fn == 0) {
        std.debug.print("huff-conformance gate: huff_build_fn == 0\n", .{});
        return error.SkipZigTest;
    }
    if (dec_loader.huff_decode_fn == 0) {
        std.debug.print("huff-conformance gate: huff_decode_fn == 0\n", .{});
        return error.SkipZigTest;
    }
    // The CUDA context is per-thread current; ptest workers are not the
    // thread that ran init(). Without this, every cuMemAlloc below
    // fails with CUDA_ERROR_INVALID_CONTEXT (201).
    if (!decode_context.bindContextToCallingThread()) {
        std.debug.print("huff-conformance gate: bindContext failed\n", .{});
        return error.SkipZigTest;
    }

    const launch_fn = cuda.cuLaunchKernel_fn orelse {
        std.debug.print("huff-conformance gate: cuLaunchKernel_fn null\n", .{});
        return error.SkipZigTest;
    };
    const h2d_fn = cuda.cuMemcpyHtoD_fn orelse {
        std.debug.print("huff-conformance gate: cuMemcpyHtoD_fn null\n", .{});
        return error.SkipZigTest;
    };
    const d2h_fn = cuda.cuMemcpyDtoH_fn orelse {
        std.debug.print("huff-conformance gate: cuMemcpyDtoH_fn null\n", .{});
        return error.SkipZigTest;
    };
    const sync_fn = cuda.cuCtxSynchronize_fn orelse {
        std.debug.print("huff-conformance gate: cuCtxSynchronize_fn null\n", .{});
        return error.SkipZigTest;
    };
    const memset_fn = cuda.cuMemsetD8_fn orelse {
        std.debug.print("huff-conformance gate: cuMemsetD8_fn null\n", .{});
        return error.SkipZigTest;
    };

    if (src.len == 0) return error.SkipZigTest;
    const src_size: u32 = @intCast(src.len);

    // ── Device buffers (all freed via defer) ───────────────────────
    var d_input: u64 = 0;
    try devAlloc(&d_input, src.len);
    defer devFree(d_input);

    var d_enc_descs: u64 = 0;
    try devAlloc(&d_enc_descs, @sizeOf(HuffEncDesc));
    defer devFree(d_enc_descs);

    var d_code_lengths: u64 = 0;
    try devAlloc(&d_code_lengths, 256);
    defer devFree(d_code_lengths);

    var d_codes: u64 = 0;
    try devAlloc(&d_codes, 256 * 4);
    defer devFree(d_codes);

    // Same per-lane scratch sizing math as gpuEncodeHuffImpl
    // (encode_huff.zig:111). KEEP IN SYNC.
    const scratch_per_stream: usize = std.mem.alignForward(
        usize,
        (@as(usize, src_size) / NUM_STREAMS + 64) * 2,
        16,
    );
    var d_scratch: u64 = 0;
    try devAlloc(&d_scratch, NUM_STREAMS * scratch_per_stream);
    defer devFree(d_scratch);

    const enc_out_cap = bilDstCap(src_size);
    var d_bil_out: u64 = 0;
    try devAlloc(&d_bil_out, enc_out_cap);
    defer devFree(d_bil_out);

    var d_enc_sizes: u64 = 0;
    try devAlloc(&d_enc_sizes, 4);
    defer devFree(d_enc_sizes);

    var d_dec_descs: u64 = 0;
    try devAlloc(&d_dec_descs, @sizeOf(HuffDecChunkDesc));
    defer devFree(d_dec_descs);

    var d_lut: u64 = 0;
    try devAlloc(&d_lut, HUFF_LUT_ENTRIES * 4);
    defer devFree(d_lut);

    var d_decoded: u64 = 0;
    try devAlloc(&d_decoded, src.len);
    defer devFree(d_decoded);

    // Counts block in the production layout scan_gpu.zig stages:
    // 6 u32 = [n_lit, n_tok, n_hi, n_lo, n_raw, n_merged]. The decode
    // kernels self-gate on n_merged (counts + 20) and classify block 0
    // as region lit via n_lit=1, n_tok=0.
    var d_counts: u64 = 0;
    try devAlloc(&d_counts, 6 * 4);
    defer devFree(d_counts);

    // ── H2D staging ────────────────────────────────────────────────
    if (h2d_fn(d_input, @ptrCast(src.ptr), src.len) != CUDA_SUCCESS) return error.CopyFailed;

    var enc_desc = HuffEncDesc{
        .src_offset = 0,
        .src_size = src_size,
        .src_stride = 1,
        .dst_offset = 0,
        .dst_capacity = enc_out_cap,
    };
    if (h2d_fn(d_enc_descs, @ptrCast(&enc_desc), @sizeOf(HuffEncDesc)) != CUDA_SUCCESS) return error.CopyFailed;

    var dec_desc = HuffDecChunkDesc{
        .in_offset = 0,
        .in_size = enc_out_cap,
        .out_offset = 0,
        .out_size = src_size,
        .lut_offset = 0,
    };
    if (h2d_fn(d_dec_descs, @ptrCast(&dec_desc), @sizeOf(HuffDecChunkDesc)) != CUDA_SUCCESS) return error.CopyFailed;

    const counts_host = [6]u32{ 1, 0, 0, 0, 0, 1 };
    if (h2d_fn(d_counts, @ptrCast(&counts_host), 6 * 4) != CUDA_SUCCESS) return error.CopyFailed;

    // Poison the device output so a partial decode shows as garbage
    // rather than silently matching stale zeros.
    if (memset_fn(d_decoded, 0xAA, src.len) != CUDA_SUCCESS) return error.CopyFailed;

    // ── Kernel 1+2: encoder build tables, then BIL encode ──────────
    // Param layout mirrors gpuEncodeHuffImpl (encode_huff.zig:137-171).
    {
        var p_src: u64 = d_input;
        var p_descs: u64 = d_enc_descs;
        var p_cl: u64 = d_code_lengths;
        var p_codes: u64 = d_codes;
        var p_stride: u32 = 256;
        var p_n: u32 = 1;
        var tbl_params = [_]?*anyopaque{
            @ptrCast(&p_src),   @ptrCast(&p_descs),  @ptrCast(&p_cl),
            @ptrCast(&p_codes), @ptrCast(&p_stride), @ptrCast(&p_n),
        };
        var extra = [_]?*anyopaque{null};
        if (launch_fn(enc_loader.huff_tables_kernel_fn, 1, 1, 1, 32, 1, 1, 0, 0, &tbl_params, &extra) != CUDA_SUCCESS)
            return error.KernelLaunchFailed;

        var p_scratch: u64 = d_scratch;
        var p_out: u64 = d_bil_out;
        var p_sizes: u64 = d_enc_sizes;
        var p_sps: u32 = @intCast(scratch_per_stream);
        var enc_params = [_]?*anyopaque{
            @ptrCast(&p_src),   @ptrCast(&p_descs),   @ptrCast(&p_cl),
            @ptrCast(&p_codes), @ptrCast(&p_scratch), @ptrCast(&p_out),
            @ptrCast(&p_sizes), @ptrCast(&p_sps),     @ptrCast(&p_stride),
            @ptrCast(&p_n),
        };
        if (launch_fn(enc_loader.huff_encode_kernel_fn, 1, 1, 1, 32, 1, 1, 0, 0, &enc_params, &extra) != CUDA_SUCCESS)
            return error.KernelLaunchFailed;
    }
    if (sync_fn() != CUDA_SUCCESS) return error.SyncFailed;

    // Encoder must have produced a non-empty body within capacity —
    // a zero here means the encode failed silently.
    var bil_size: u32 = 0;
    if (d2h_fn(@ptrCast(&bil_size), d_enc_sizes, 4) != CUDA_SUCCESS) return error.CopyFailed;
    try testing.expect(bil_size > 0);
    try testing.expect(bil_size <= enc_out_cap);

    // ── Kernel 3: decoder build LUT ────────────────────────────────
    // Param layout mirrors launchHuffKernels (decode_dispatch.zig:328).
    // n_blocks gate = counts + 20 (the n_merged slot).
    const d_n_huff: u64 = d_counts + 20;
    {
        var p_comp: u64 = d_bil_out;
        var p_descs: u64 = d_dec_descs;
        var p_lut: u64 = d_lut;
        var p_n: u64 = d_n_huff;
        var params = [_]?*anyopaque{
            @ptrCast(&p_comp), @ptrCast(&p_descs),
            @ptrCast(&p_lut),  @ptrCast(&p_n),
        };
        var extra = [_]?*anyopaque{null};
        if (launch_fn(dec_loader.huff_build_fn, 1, 1, 1, 32, 1, 1, 0, 0, &params, &extra) != CUDA_SUCCESS)
            return error.KernelLaunchFailed;
    }

    // ── Kernel 4: decoder decode 4-stream ──────────────────────────
    // A-024 layout: counts pointer + u64 region offsets. The single
    // block classifies as lit (n_lit=1), so both region offsets are
    // unused — pass 0.
    {
        var p_comp: u64 = d_bil_out;
        var p_descs: u64 = d_dec_descs;
        var p_lut: u64 = d_lut;
        var p_out: u64 = d_decoded;
        var p_n: u64 = d_n_huff;
        var p_counts: u64 = d_counts;
        var p_tok_off: u64 = 0;
        var p_off16_off: u64 = 0;
        var params = [_]?*anyopaque{
            @ptrCast(&p_comp), @ptrCast(&p_descs),
            @ptrCast(&p_lut),  @ptrCast(&p_out),
            @ptrCast(&p_n),
            @ptrCast(&p_counts),
            @ptrCast(&p_tok_off),
            @ptrCast(&p_off16_off),
        };
        var extra = [_]?*anyopaque{null};
        const shared_bytes: c_uint = HUFF_LUT_ENTRIES * @sizeOf(u32);
        if (launch_fn(dec_loader.huff_decode_fn, 1, 1, 1, 32, 1, 1, shared_bytes, 0, &params, &extra) != CUDA_SUCCESS)
            return error.KernelLaunchFailed;
    }
    if (sync_fn() != CUDA_SUCCESS) return error.SyncFailed;

    // ── D2H + byte-identity assert ─────────────────────────────────
    const decoded = try allocator.alloc(u8, src.len);
    defer allocator.free(decoded);
    @memset(decoded, 0x55);
    if (d2h_fn(@ptrCast(decoded.ptr), d_decoded, src.len) != CUDA_SUCCESS) return error.CopyFailed;
    if (sync_fn() != CUDA_SUCCESS) return error.SyncFailed;

    try testing.expectEqualSlices(u8, src, decoded);
}

// ── Test cases (mirror the VK suite case-for-case) ─────────────────
// All take the shared GPU-test lock: they launch kernels and allocate
// raw device buffers, which would race sibling GPU tests under the
// parallel runner.

test "huff conformance: pseudo-random 1 KiB roundtrip" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    const src = try testing.allocator.alloc(u8, 1024);
    defer testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xC0FFEE_F1F0);
    rng.random().bytes(src);
    try huffRoundtripOne(testing.allocator, src);
}

test "huff conformance: repeating English 8 KiB roundtrip" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    const len: usize = 8 * 1024;
    const src = try testing.allocator.alloc(u8, len);
    defer testing.allocator.free(src);
    const pattern = "the quick brown fox jumps over the lazy dog. ";
    for (src, 0..) |*b, i| b.* = pattern[i % pattern.len];
    try huffRoundtripOne(testing.allocator, src);
}

test "huff conformance: 64 KiB pseudo-random roundtrip" {
    // 64 KiB = the L1 sub-chunk cap; exercises the full Phase-2
    // interleaved hot loop plus the per-lane tail prefix sum.
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    const src = try testing.allocator.alloc(u8, 64 * 1024);
    defer testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xDEADBEEF_BADC0DE);
    rng.random().bytes(src);
    try huffRoundtripOne(testing.allocator, src);
}

test "huff conformance: skewed alphabet (wide code-length range)" {
    // Zipf-ish skew drives code lengths from 2 to the height limit;
    // lanes end up with very different per-stream bit counts, which
    // exercises both sides' per-lane prefix sums.
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    const len: usize = 16 * 1024;
    const src = try testing.allocator.alloc(u8, len);
    defer testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xABCD_1234);
    for (src) |*b| {
        const r = rng.random().int(u16);
        b.* = if (r < 32768) 'A' else if (r < 49152) 'B' else if (r < 57344) 'C' else @intCast(r & 0xFF);
    }
    try huffRoundtripOne(testing.allocator, src);
}

test "huff conformance: odd-length boundary (4097 bytes)" {
    // src_size % 32 != 0: lane 31 absorbs the remainder; non-trivial
    // BIL K with mixed full/partial words across lanes.
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    const len: usize = 4097;
    const src = try testing.allocator.alloc(u8, len);
    defer testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xFACE_F00D);
    rng.random().bytes(src);
    try huffRoundtripOne(testing.allocator, src);
}
