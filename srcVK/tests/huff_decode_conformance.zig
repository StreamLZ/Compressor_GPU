//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Isolated kernel-level conformance test for the Huffman decode chain
//! (slzHuffBuildLutKernel + slzHuffDecode4StreamKernel). Drives the VK
//! Huffman ENCODER kernels (slzHuffBuildTablesKernel +
//! slzHuffEncode4StreamKernel) on a known input to produce a BIL body +
//! per-symbol weights, then feeds those exact bytes as INPUTS to the VK
//! Huffman DECODER kernels and asserts byte-identity vs the original
//! input.
//!
//! Rationale (Phase 2A-decoder iter 3 review, workflow w5bqf1x6a):
//!   - The iter 3 adversarial review caught a CRITICAL_PORT_VIOLATION in
//!     bitbufRefill (hi/lo swap) that would silently corrupt every
//!     huff_decode output. Fix landed at ac6696f.
//!   - The bug went undetected because (a) L1 paths never reach
//!     huff_decode_4stream, (b) L2+ paths are blocked by the lz_decode
//!     general stub, and (c) no isolated conformance test exists.
//!   - This test exists so iter 4 (lz_decode general workhorse, ~600
//!     LOC) has a real correctness gate on the Huffman pre-decode chain
//!     to lean on. Without it, future bit-buffer math regressions could
//!     ship silently and produce garbage decode output only at L2+.
//!
//! The encoder kernels are proven byte-identical to CUDA at f9e84e3;
//! the decoder iter 3 fix should make the decoder also faithful. If
//! either side regresses, the roundtrip fails.

const std = @import("std");
const testing = std.testing;

const vk = @import("../decode/vulkan_api.zig");
const enc_driver = @import("../encode/driver.zig");
const dec_driver = @import("../decode/driver.zig");
const enc_module_loader = @import("../encode/module_loader.zig");
const dec_module_loader = @import("../decode/module_loader.zig");
const decode_context = @import("../decode/decode_context.zig");
const encode_context = @import("../encode/encode_context.zig");
const descriptors = @import("../decode/descriptors.zig");

const HuffEncDesc = encode_context.HuffEncDesc;
const HuffDecChunkDesc = descriptors.HuffDecChunkDesc;
const VkDeviceBuffer = vk.VkDeviceBuffer;
const VK_SUCCESS_RC = vk.VK_SUCCESS_RC;

// VK adaptation: mirror of HUFF_BODY_FIXED_BYTES in encode_huff.zig
// (which is private). Per-block fixed component of the BIL Huffman
// encoder dst capacity: 228 B header (weights 128 + sub-header 96 +
// K 4) + 96 B rounding tax (4 × N where N=32) + 32 B slack. Keep in
// sync with HUFF_BODY_FIXED_BYTES at srcVK/encode/encode_huff.zig:48.
const HUFF_BODY_FIXED_BYTES: u32 = 356;

// Encoder NUM_STREAMS — mirrors HUFF_NUM_STREAMS in gpu_huffman.glsl
// (= 32). KEEP IN SYNC with the encoder's `NUM_STREAMS` constant at
// srcVK/encode/encode_huff.zig.
const NUM_STREAMS: u32 = 32;

// Decoder LUT entry count — mirrors HUFF_LUT_ENTRIES in
// gpu_huffman.glsl (= 1024). Each LUT entry is a u32. Keep in sync
// with descriptors.HUFF_LUT_ENTRIES.
const HUFF_LUT_ENTRIES: u32 = @intCast(descriptors.HUFF_LUT_ENTRIES);

/// Mirror of encode_huff.zig::bilDstCap. Per-block dst slot size,
/// 4-byte aligned so cumulative `dst_offset` stays aligned for the
/// BIL kernel's per-row 4-byte stores.
inline fn bilDstCap(count: u32) u32 {
    return (count * 2 + HUFF_BODY_FIXED_BYTES + 3) & ~@as(u32, 3);
}

/// Core roundtrip helper. Drives the four Huffman kernels (encoder
/// build + encode, decoder build + decode) on a single sub-chunk and
/// asserts byte-identity. The test wraps this in per-case fixtures so
/// each test case gets fresh per-test contexts (matches the pattern in
/// l1_encode_roundtrip.zig — necessary because ptest_vk runs tests on
/// a worker pool against the shared driver state).
fn huffRoundtripOne(allocator: std.mem.Allocator, src: []const u8) !void {
    // Init both backends. The encode side chains into the decode side
    // (single VkInstance + VkDevice), so we only need to gate on the
    // encode init() — but verifying the decoder init separately keeps
    // the failure mode legible if the test is run in isolation.
    if (!enc_driver.init()) return error.SkipZigTest;
    if (!dec_driver.init()) return error.SkipZigTest;

    // Pipeline-fn handles required for the four-kernel roundtrip. If
    // any are unbuilt (missing SPV / shader-module create failure),
    // skip — there's no useful conformance signal to emit.
    if (enc_module_loader.huff_tables_kernel_fn == 0) return error.SkipZigTest;
    if (enc_module_loader.huff_encode_kernel_fn == 0) return error.SkipZigTest;
    if (dec_module_loader.huff_build_fn == 0) return error.SkipZigTest;
    if (dec_module_loader.huff_decode_fn == 0) return error.SkipZigTest;

    const launch_fn = vk.procs.launch_kernel orelse return error.SkipZigTest;
    const h2d_fn = vk.procs.h2d orelse return error.SkipZigTest;
    const d2h_fn = vk.procs.d2h orelse return error.SkipZigTest;
    const sync_fn = vk.procs.ctx_sync orelse return error.SkipZigTest;
    const malloc_fn = vk.procs.malloc_device orelse return error.SkipZigTest;
    const free_fn = vk.procs.free_device orelse return error.SkipZigTest;

    // Sanity: the BIL encoder requires src_size > 0 (empty descs early
    // out with no body emitted; out_sizes[0]=0). An empty-input test
    // wouldn't exercise the decoder at all.
    if (src.len == 0) return error.SkipZigTest;
    const src_size: u32 = @intCast(src.len);

    // ── Allocate device buffers ────────────────────────────────────
    // Encoder inputs/outputs + decoder inputs/outputs. All allocs are
    // freed via the `defer free_fn(...)` chain. Note: this test does
    // NOT use EncodeContext/DecodeContext — those carry orchestration
    // state irrelevant to the kernel conformance probe. Direct
    // allocations keep the test focused on the kernel I/O contract.

    // Encoder input (the bytes to compress).
    var d_input: VkDeviceBuffer = 0;
    if (malloc_fn(&d_input, src.len) != VK_SUCCESS_RC) return error.SkipZigTest;
    defer _ = free_fn(d_input);

    // Encoder descriptor list (one HuffEncDesc).
    var d_enc_descs: VkDeviceBuffer = 0;
    if (malloc_fn(&d_enc_descs, @sizeOf(HuffEncDesc)) != VK_SUCCESS_RC) return error.SkipZigTest;
    defer _ = free_fn(d_enc_descs);

    // Encoder code-length output (256 bytes — HUFF_ALPHABET u8).
    var d_code_lengths: VkDeviceBuffer = 0;
    if (malloc_fn(&d_code_lengths, 256) != VK_SUCCESS_RC) return error.SkipZigTest;
    defer _ = free_fn(d_code_lengths);

    // Encoder codes output (256 u32).
    var d_codes: VkDeviceBuffer = 0;
    if (malloc_fn(&d_codes, 256 * 4) != VK_SUCCESS_RC) return error.SkipZigTest;
    defer _ = free_fn(d_codes);

    // Encoder per-lane scratch slab. Same sizing math as gpuEncodeHuffImpl
    // at encode_huff.zig:108-115. KEEP IN SYNC with the encoder host
    // glue: any change to scratch_per_stream there must mirror here.
    const scratch_per_stream: usize = std.mem.alignForward(
        usize,
        (@as(usize, src_size) / NUM_STREAMS + 64) * 2,
        16,
    );
    const scratch_bytes: usize = NUM_STREAMS * scratch_per_stream;
    var d_scratch: VkDeviceBuffer = 0;
    if (malloc_fn(&d_scratch, scratch_bytes) != VK_SUCCESS_RC) return error.SkipZigTest;
    defer _ = free_fn(d_scratch);

    // Encoder BIL output. Sized by bilDstCap — same formula
    // encodeStreamAndPublish uses.
    const enc_out_cap = bilDstCap(src_size);
    var d_bil_out: VkDeviceBuffer = 0;
    if (malloc_fn(&d_bil_out, enc_out_cap) != VK_SUCCESS_RC) return error.SkipZigTest;
    defer _ = free_fn(d_bil_out);

    // Encoder per-block size (single u32 here for n=1).
    var d_enc_sizes: VkDeviceBuffer = 0;
    if (malloc_fn(&d_enc_sizes, 4) != VK_SUCCESS_RC) return error.SkipZigTest;
    defer _ = free_fn(d_enc_sizes);

    // Decoder descriptor list (one HuffDecChunkDesc, 20 bytes).
    var d_dec_descs: VkDeviceBuffer = 0;
    if (malloc_fn(&d_dec_descs, @sizeOf(HuffDecChunkDesc)) != VK_SUCCESS_RC) return error.SkipZigTest;
    defer _ = free_fn(d_dec_descs);

    // Decoder LUT output (1024 u32 = 4096 bytes).
    var d_lut: VkDeviceBuffer = 0;
    if (malloc_fn(&d_lut, HUFF_LUT_ENTRIES * 4) != VK_SUCCESS_RC) return error.SkipZigTest;
    defer _ = free_fn(d_lut);

    // Decoder n_blocks self-gate (single u32 = 1).
    var d_n_blocks: VkDeviceBuffer = 0;
    if (malloc_fn(&d_n_blocks, 4) != VK_SUCCESS_RC) return error.SkipZigTest;
    defer _ = free_fn(d_n_blocks);

    // Decoder output (decoded bytes — should match input).
    var d_decoded: VkDeviceBuffer = 0;
    if (malloc_fn(&d_decoded, src.len) != VK_SUCCESS_RC) return error.SkipZigTest;
    defer _ = free_fn(d_decoded);

    // A-024 (2026-06-10): the decode kernel partitions its merged-descs
    // block-id range into lit | tok | off16 regions via d_compact_counts
    // (256-byte stride per slot: n_lit at byte 0, n_tok at byte 256) and
    // only handles blocks whose region matches the region_select push
    // constant. This standalone probe has ONE desc which must classify
    // as region 0 (lit): n_lit = 1, n_tok = 0.
    var d_counts: VkDeviceBuffer = 0;
    if (malloc_fn(&d_counts, 512) != VK_SUCCESS_RC) return error.SkipZigTest;
    defer _ = free_fn(d_counts);

    // ── H2D uploads ────────────────────────────────────────────────
    // Encoder input bytes.
    if (h2d_fn(d_input, @ptrCast(src.ptr), src.len) != VK_SUCCESS_RC) return error.KernelLaunchFailed;

    // Single encoder descriptor: stride=1 (byte stream), full input.
    var enc_desc = HuffEncDesc{
        .src_offset = 0,
        .src_size = src_size,
        .src_stride = 1,
        .dst_offset = 0,
        .dst_capacity = enc_out_cap,
    };
    if (h2d_fn(d_enc_descs, @ptrCast(&enc_desc), @sizeOf(HuffEncDesc)) != VK_SUCCESS_RC) return error.KernelLaunchFailed;

    // Single decoder descriptor: in_offset=0 (BIL starts at byte 0 of
    // d_bil_out), out_offset=0, out_size = src_size, lut_offset=0 (one
    // 1024-entry LUT, indexed by u32 elements).
    //
    // Note in_size for the decoder is unused by huff_decode_4stream
    // (parses BIL header internally) but set it to enc_out_cap for
    // future-proofing.
    var dec_desc = HuffDecChunkDesc{
        .in_offset = 0,
        .in_size = enc_out_cap,
        .out_offset = 0,
        .out_size = src_size,
        .lut_offset = 0,
    };
    if (h2d_fn(d_dec_descs, @ptrCast(&dec_desc), @sizeOf(HuffDecChunkDesc)) != VK_SUCCESS_RC) return error.KernelLaunchFailed;

    var n_blocks: u32 = 1;
    if (h2d_fn(d_n_blocks, @ptrCast(&n_blocks), 4) != VK_SUCCESS_RC) return error.KernelLaunchFailed;

    // A-024 counts: n_lit = 1 at byte 0, n_tok = 0 at byte 256 (the
    // 256-byte COUNTS_STRIDE layout the dispatch uses).
    var counts_host = [_]u8{0} ** 512;
    counts_host[0] = 1;
    if (h2d_fn(d_counts, @ptrCast(&counts_host), 512) != VK_SUCCESS_RC) return error.KernelLaunchFailed;

    if (sync_fn() != VK_SUCCESS_RC) return error.SyncFailed;

    // ── Kernel 1: encoder build tables ─────────────────────────────
    // Binding order per huff_build_tables_kernel.comp:22-48 and
    // ENCODE_KERNELS.huff_tables (n_bindings=4, push_constant_size=8).
    // Layout: SSBOs [Input, Descs, CodeLengths, Codes] then push consts
    // [tables_stride: u32, n_blocks: u32].
    {
        var p_src: VkDeviceBuffer = d_input;
        var p_descs: VkDeviceBuffer = d_enc_descs;
        var p_cl: VkDeviceBuffer = d_code_lengths;
        var p_codes: VkDeviceBuffer = d_codes;
        var p_stride: u32 = 256;
        var p_n: u32 = 1;
        var params = [_]?*anyopaque{
            @ptrCast(&p_src), @ptrCast(&p_descs), @ptrCast(&p_cl), @ptrCast(&p_codes),
            @ptrCast(&p_stride), @ptrCast(&p_n),
        };
        var extra = [_]?*anyopaque{null};
        if (launch_fn(enc_module_loader.huff_tables_kernel_fn, 1, 1, 1, 32, 1, 1, 0, 0, &params, &extra, null) != VK_SUCCESS_RC) {
            return error.KernelLaunchFailed;
        }
    }
    if (sync_fn() != VK_SUCCESS_RC) return error.SyncFailed;

    // ── Kernel 2: encoder encode 4-stream ──────────────────────────
    // Binding order per huff_encode_4stream_kernel.comp:27-71 and
    // ENCODE_KERNELS.huff_encode (n_bindings=7, push_constant_size=12).
    // Layout: SSBOs [Input, Descs, CodeLengths, Codes, Scratch, Output,
    // OutSizes] then push consts [scratch_per_stream, tables_stride,
    // n_blocks].
    {
        var p_src: VkDeviceBuffer = d_input;
        var p_descs: VkDeviceBuffer = d_enc_descs;
        var p_cl: VkDeviceBuffer = d_code_lengths;
        var p_codes: VkDeviceBuffer = d_codes;
        var p_scratch: VkDeviceBuffer = d_scratch;
        var p_out: VkDeviceBuffer = d_bil_out;
        var p_sizes: VkDeviceBuffer = d_enc_sizes;
        var p_sps: u32 = @intCast(scratch_per_stream);
        var p_stride: u32 = 256;
        var p_n: u32 = 1;
        var params = [_]?*anyopaque{
            @ptrCast(&p_src),   @ptrCast(&p_descs),   @ptrCast(&p_cl),
            @ptrCast(&p_codes), @ptrCast(&p_scratch), @ptrCast(&p_out),
            @ptrCast(&p_sizes), @ptrCast(&p_sps),     @ptrCast(&p_stride),
            @ptrCast(&p_n),
        };
        var extra = [_]?*anyopaque{null};
        if (launch_fn(enc_module_loader.huff_encode_kernel_fn, 1, 1, 1, 32, 1, 1, 0, 0, &params, &extra, null) != VK_SUCCESS_RC) {
            return error.KernelLaunchFailed;
        }
    }
    if (sync_fn() != VK_SUCCESS_RC) return error.SyncFailed;

    // Validate the encoder actually wrote something — if out_size came
    // back zero the encode failed silently (degenerate alphabet or
    // empty input that slipped past the earlier guard).
    var bil_size: u32 = 0;
    if (d2h_fn(@ptrCast(&bil_size), d_enc_sizes, 4) != VK_SUCCESS_RC) return error.CopyFailed;
    try testing.expect(bil_size > 0);
    try testing.expect(bil_size <= enc_out_cap);

    // ── Kernel 3: decoder build LUT ────────────────────────────────
    // Binding order per huff_build_lut_kernel.comp:34-52 and
    // KERNEL_DECLS huff_build_fn (n_bindings=4, push_constant_size=0).
    // Layout: SSBOs [Comp, Descs, Luts, NBlocks] — no push consts.
    //
    // The encoder wrote a BIL body at offset 0 of d_bil_out. The
    // decoder's `in_offset` (descriptor field 0) is the absolute byte
    // offset within `comp` where the body's weights start — so feed
    // d_bil_out as the decoder's `comp` and set in_offset=0.
    {
        var p_comp: VkDeviceBuffer = d_bil_out;
        var p_descs: VkDeviceBuffer = d_dec_descs;
        var p_lut: VkDeviceBuffer = d_lut;
        var p_n: VkDeviceBuffer = d_n_blocks;
        var params = [_]?*anyopaque{
            @ptrCast(&p_comp), @ptrCast(&p_descs), @ptrCast(&p_lut), @ptrCast(&p_n),
        };
        var extra = [_]?*anyopaque{null};
        if (launch_fn(dec_module_loader.huff_build_fn, 1, 1, 1, 32, 1, 1, 0, 0, &params, &extra, null) != VK_SUCCESS_RC) {
            return error.KernelLaunchFailed;
        }
    }
    if (sync_fn() != VK_SUCCESS_RC) return error.SyncFailed;

    // ── Kernel 4: decoder decode 4-stream ──────────────────────────
    // Binding order per huff_decode_4stream_kernel.comp and KERNEL_DECLS
    // huff_decode_fn (A-024 2026-06-10: n_bindings=7,
    // push_constant_size=4). Layout: SSBOs [Comp, Descs, Luts, Output,
    // NBlocks, OutputU32, CompactCounts] then push const
    // [region_select: u32]. Binding 5 is a u32-aliased view of binding
    // 3's VkBuffer used for the Phase 2 hot loop's 4-byte store fast
    // path. The single desc classifies as region 0 (lit) via the counts
    // staged above, so one dispatch with region_select=0 suffices —
    // the production dispatch loops over all three regions.
    {
        var p_comp: VkDeviceBuffer = d_bil_out;
        var p_descs: VkDeviceBuffer = d_dec_descs;
        var p_lut: VkDeviceBuffer = d_lut;
        var p_out: VkDeviceBuffer = d_decoded;
        var p_n: VkDeviceBuffer = d_n_blocks;
        var p_out_u32: VkDeviceBuffer = d_decoded;
        var p_counts: VkDeviceBuffer = d_counts;
        var p_region: u32 = 0;
        var params = [_]?*anyopaque{
            @ptrCast(&p_comp),   @ptrCast(&p_descs), @ptrCast(&p_lut),
            @ptrCast(&p_out),    @ptrCast(&p_n),     @ptrCast(&p_out_u32),
            @ptrCast(&p_counts), @ptrCast(&p_region),
        };
        var extra = [_]?*anyopaque{null};
        if (launch_fn(dec_module_loader.huff_decode_fn, 1, 1, 1, 32, 1, 1, 0, 0, &params, &extra, null) != VK_SUCCESS_RC) {
            return error.KernelLaunchFailed;
        }
    }
    if (sync_fn() != VK_SUCCESS_RC) return error.SyncFailed;

    // ── D2H + assert ───────────────────────────────────────────────
    const decoded = try allocator.alloc(u8, src.len);
    defer allocator.free(decoded);
    @memset(decoded, 0xAA); // poison so a partial decode shows up as garbage
    if (d2h_fn(@ptrCast(decoded.ptr), d_decoded, src.len) != VK_SUCCESS_RC) return error.CopyFailed;
    if (sync_fn() != VK_SUCCESS_RC) return error.SyncFailed;

    try testing.expectEqualSlices(u8, src, decoded);
}

// ── Test cases ─────────────────────────────────────────────────────
//
// Each case targets a different region of the kernel input space.
// Smallest first (1 KiB) so a CI run with timeout-pressure surfaces
// the smoke gate first. The `[serial]` suffix forces the test onto
// the post-parallel serial phase of test_runner_parallel.zig — these
// tests allocate/free their own raw VkDeviceBuffers off the global
// allocator and would race with sibling-worker ensureBuf calls on
// the shared encode/decode contexts under the default parallel pool.

test "huff_decode conformance: pseudo-random 1 KiB roundtrip [serial]" {
    const src = try std.testing.allocator.alloc(u8, 1024);
    defer std.testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xC0FFEE_F1F0);
    rng.random().bytes(src);
    try huffRoundtripOne(std.testing.allocator, src);
}

test "huff_decode conformance: repeating English 8 KiB roundtrip [serial]" {
    const len: usize = 8 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    const pattern = "the quick brown fox jumps over the lazy dog. ";
    var i: usize = 0;
    while (i < len) : (i += 1) src[i] = pattern[i % pattern.len];
    try huffRoundtripOne(std.testing.allocator, src);
}

test "huff_decode conformance: 64 KiB pseudo-random roundtrip [serial]" {
    // 64 KiB = the L1 sub-chunk cap; exercises the full Phase-2 (interleaved)
    // hot loop plus per-lane tail prefix sum + 32-way output carve-up.
    const src = try std.testing.allocator.alloc(u8, 64 * 1024);
    defer std.testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xDEADBEEF_BADC0DE);
    rng.random().bytes(src);
    try huffRoundtripOne(std.testing.allocator, src);
}

test "huff_decode conformance: skewed alphabet (wide code-length range) [serial]" {
    // Skewed Zipf-ish distribution: a handful of high-frequency bytes +
    // many low-frequency bytes drives the code-length range from L=2
    // (top of the tree) to L=11 (the Kraft height limit). Tests that
    // lanes with very different per-stream byte counts roundtrip
    // identically — exercises the per-lane bytes prefix-sum on the
    // encode side and the per-lane tail-bytes prefix-sum on the
    // decode side.
    const len: usize = 16 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xABCD_1234);
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const r = rng.random().int(u16);
        // ~50% 'A', ~25% 'B', ~12.5% 'C', remainder uniform over 256.
        src[i] = if (r < 32768) 'A' else if (r < 49152) 'B' else if (r < 57344) 'C' else @intCast(r & 0xFF);
    }
    try huffRoundtripOne(std.testing.allocator, src);
}

test "huff_decode conformance: odd-length boundary (4097 bytes) [serial]" {
    // Tests the (src_size % NUM_STREAMS != 0) branch of the encoder's
    // per-lane slice carve-up where lane (N-1) absorbs the remainder.
    // Drives the corresponding (out_size % NUM_STREAMS != 0) branch in
    // huff_decode_4stream_kernel.comp::main (line 612-618), and a
    // non-trivial BIL K with mixed full/partial words across lanes.
    const len: usize = 4097;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xFACE_F00D);
    rng.random().bytes(src);
    try huffRoundtripOne(std.testing.allocator, src);
}
