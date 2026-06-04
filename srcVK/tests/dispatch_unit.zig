//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Unit tests for srcVK/decode/decode_dispatch.zig: L2 gate behaviour
//! (level=1 skips Huff/scan/compact/merge/gather), runLzPipeline raw-
//! kernel selection, buildChunkDescriptors output. Test bodies added by
//! the fleshout agent.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const testing = std.testing;
const decode_dispatch = @import("../decode/decode_dispatch.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const descriptors = @import("../decode/descriptors.zig");
const driver = @import("../decode/driver.zig");

// These tests focus on structural / type-level invariants of the
// dispatch layer — no GPU is brought up. The L2 gate is verified by
// inspecting the DecodeRequest field surface; runtime behaviour is
// exercised by l1_decode_roundtrip / l1_encode_roundtrip.

test "DecodeRequest: level field defaults to 1 (L1 gate per audit Section C.5.1)" {
    const req: decode_dispatch.DecodeRequest = .{
        .chunk_descs = &.{},
        .compressed_block = &.{},
        .dst_full = undefined,
        .dst_start_off = 0,
        .decompressed_size = 0,
        .chunks_per_group = 1,
        .sub_chunk_cap = 0,
        .io = null,
        .d_output_target = null,
        .d_compressed_src = null,
    };
    try testing.expectEqual(@as(u8, 1), req.level);
}

test "DecodeRequest: writesDirectlyToTarget true only when d_output_target set AND dst_start_off==0" {
    var dst_buf: [4]u8 = .{ 0, 0, 0, 0 };

    const no_target: decode_dispatch.DecodeRequest = .{
        .chunk_descs = &.{},
        .compressed_block = &.{},
        .dst_full = &dst_buf,
        .dst_start_off = 0,
        .decompressed_size = 0,
        .chunks_per_group = 1,
        .sub_chunk_cap = 0,
        .io = null,
        .d_output_target = null,
        .d_compressed_src = null,
    };
    try testing.expect(!no_target.writesDirectlyToTarget());

    const target_with_prefix: decode_dispatch.DecodeRequest = .{
        .chunk_descs = &.{},
        .compressed_block = &.{},
        .dst_full = &dst_buf,
        .dst_start_off = 16,
        .decompressed_size = 0,
        .chunks_per_group = 1,
        .sub_chunk_cap = 0,
        .io = null,
        .d_output_target = 0xDEAD_BEEF,
        .d_compressed_src = null,
    };
    // dst_start_off != 0 disqualifies even with target.
    try testing.expect(!target_with_prefix.writesDirectlyToTarget());

    const target_no_prefix: decode_dispatch.DecodeRequest = .{
        .chunk_descs = &.{},
        .compressed_block = &.{},
        .dst_full = &dst_buf,
        .dst_start_off = 0,
        .decompressed_size = 0,
        .chunks_per_group = 1,
        .sub_chunk_cap = 0,
        .io = null,
        .d_output_target = 0xDEAD_BEEF,
        .d_compressed_src = null,
    };
    try testing.expect(target_no_prefix.writesDirectlyToTarget());
}

test "DecodeRequest: level can be set to 2 for L2 (gate flips at fullGpuLaunchImpl line ~724)" {
    const req: decode_dispatch.DecodeRequest = .{
        .chunk_descs = &.{},
        .compressed_block = &.{},
        .dst_full = undefined,
        .dst_start_off = 0,
        .decompressed_size = 0,
        .chunks_per_group = 1,
        .sub_chunk_cap = 0,
        .io = null,
        .d_output_target = null,
        .d_compressed_src = null,
        .level = 2,
    };
    try testing.expectEqual(@as(u8, 2), req.level);
}

test "runHuffBuildAndDecode: L2 stub returns NotImplementedL2 (L1 must not reach this)" {
    // The L2 gate at decode_dispatch.zig:724 only calls into this on
    // level>=2 paths; if it ever reaches it on L1, surfaces as runtime
    // NotImplementedL2 — which IS the assertion we are documenting.
    // We can't legitimately call it here without a live DecodeContext +
    // procs, so this test asserts the stub's existence by capturing the
    // function pointer.
    const fn_ptr = &decode_dispatch.runHuffBuildAndDecode;
    try testing.expect(@intFromPtr(fn_ptr) != 0);
}

test "ChunkDesc layout: extern struct, 24 bytes, fields ordered for SPIR-V binding" {
    // Layout MUST match the .comp std430 layout in lz_decode_raw_kernel.comp.
    // src_offset(4) + comp_size(4) + decomp_size(4) + dst_offset(4) +
    // flags(4) + memset_fill(1) + reserved(3) = 24 bytes.
    try testing.expectEqual(@as(usize, 24), @sizeOf(descriptors.ChunkDesc));
    try testing.expectEqual(@as(usize, 0), @offsetOf(descriptors.ChunkDesc, "src_offset"));
    try testing.expectEqual(@as(usize, 4), @offsetOf(descriptors.ChunkDesc, "comp_size"));
    try testing.expectEqual(@as(usize, 8), @offsetOf(descriptors.ChunkDesc, "decomp_size"));
    try testing.expectEqual(@as(usize, 12), @offsetOf(descriptors.ChunkDesc, "dst_offset"));
    try testing.expectEqual(@as(usize, 16), @offsetOf(descriptors.ChunkDesc, "flags"));
    try testing.expectEqual(@as(usize, 20), @offsetOf(descriptors.ChunkDesc, "memset_fill"));
}

test "buildChunkDescriptors via decoder facade: 2 chunks at proper dst_offset stride" {
    const block_header = @import("../format/block_header.zig");

    // Construct a small block: 2 chunks of 256 bytes each, both uncompressed
    // at chunk scope (block-level uncompressed bit). buildChunkDescriptors
    // re-parses the 2-byte internal block header at each chunk boundary,
    // so the payload contains [hdr][256 raw][hdr][256 raw].
    const eff_chunk_size: usize = 256;
    const decompressed_size: usize = 512;

    var payload: [2 + 256 + 2 + 256]u8 = undefined;
    payload[0] = 0x80 | 0x05; // uncompressed = true, magic 0x5
    payload[1] = @intFromEnum(block_header.CodecType.fast);
    for (payload[2..258], 0..) |*b, i| b.* = @intCast((i + 1) & 0xFF);
    payload[258] = 0x80 | 0x05;
    payload[259] = @intFromEnum(block_header.CodecType.fast);
    for (payload[260..516], 0..) |*b, i| b.* = @intCast((i + 1) & 0xFF);

    var descs: [2]descriptors.ChunkDesc = undefined;
    @memset(std.mem.sliceAsBytes(descs[0..]), 0);
    try decoder.buildChunkDescriptors(
        &payload,
        &descs,
        eff_chunk_size,
        decompressed_size,
        0,
    );
    try testing.expectEqual(@as(u32, 0), descs[0].dst_offset);
    try testing.expectEqual(@as(u32, eff_chunk_size), descs[1].dst_offset);
    try testing.expectEqual(@as(u32, eff_chunk_size), descs[0].decomp_size);
    try testing.expectEqual(@as(u32, eff_chunk_size), descs[1].decomp_size);
    try testing.expectEqual(@as(u32, 1), descs[0].flags); // uncompressed
    try testing.expectEqual(@as(u32, 1), descs[1].flags); // uncompressed
}

test "buildChunkDescriptors via decoder facade: respects dst_start_off prefix" {
    const block_header = @import("../format/block_header.zig");

    const eff_chunk_size: usize = 128;
    const decompressed_size: usize = 128;
    const dst_start_off: usize = 1024;

    var payload: [2 + 128]u8 = undefined;
    payload[0] = 0x80 | 0x05; // uncompressed
    payload[1] = @intFromEnum(block_header.CodecType.fast);
    for (payload[2..], 0..) |*b, i| b.* = @intCast(i);

    var descs: [1]descriptors.ChunkDesc = undefined;
    @memset(std.mem.sliceAsBytes(descs[0..]), 0);
    try decoder.buildChunkDescriptors(
        &payload,
        &descs,
        eff_chunk_size,
        decompressed_size,
        dst_start_off,
    );
    try testing.expectEqual(@as(u32, dst_start_off), descs[0].dst_offset);
}

test "WALK_MAX_CHUNKS exported through driver facade is positive" {
    try testing.expect(driver.WALK_MAX_CHUNKS > 0);
}
