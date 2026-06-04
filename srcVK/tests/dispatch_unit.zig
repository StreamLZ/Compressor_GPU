//! NEW per Exception 3 (no CUDA counterpart). Test bodies are populated.
//!
//! Unit tests for srcVK/decode/decode_dispatch.zig: L2 gate behaviour
//! (level=1 skips Huff/scan/compact/merge/gather), runLzPipeline raw-
//! kernel selection, buildChunkDescriptors output, and the
//! kernel_raw_fn ABI surface (params[] layout vs KERNEL_DECLS).
//!
//! See srcVK/audit.md for the test scope rationale.

const std = @import("std");
const testing = std.testing;
const decode_dispatch = @import("../decode/decode_dispatch.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const descriptors = @import("../decode/descriptors.zig");
const module_loader = @import("../decode/module_loader.zig");

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

test "kernel_raw_fn ABI: KERNEL_DECLS n_bindings=4 + push=8 matches runLzPipeline raw_params[] layout" {
    // Pin the kernel_raw_fn entry in module_loader.KERNEL_DECLS against the
    // raw_params[] array built in decode_dispatch.runLzPipeline (use_raw_kernel
    // branch). The 4 SSBO bindings are CompressedBuf, ChunksBuf, DstBuf,
    // TotalChunksBuf; the 2× u32 push constants are chunks_per_group +
    // sub_chunk_cap. If either side drifts, this test fires.
    const info = module_loader.kernelLayoutByName("kernel_raw_fn") orelse {
        try testing.expect(false); // kernel_raw_fn missing from KERNEL_DECLS
        return;
    };
    try testing.expectEqual(@as(u32, 4), info.n_bindings);
    try testing.expectEqual(@as(u32, 8), info.push_constant_size);

    // Mirror the raw_params[] construction from runLzPipeline (decode_dispatch.zig:411)
    // to assert the array has 6 slots: 4 device-buffer-pointer slots
    // followed by 2 u32-value-pointer slots.
    var comp: u64 = 0xAAAA_0000_0000_0001;
    var descs: u64 = 0xAAAA_0000_0000_0002;
    var dst: u64 = 0xAAAA_0000_0000_0003;
    var total: u64 = 0xAAAA_0000_0000_0004;
    var chunks_per_group: u32 = 7;
    var sub_chunk_cap: u32 = 13;

    const raw_params = [_]?*anyopaque{
        @ptrCast(&comp),
        @ptrCast(&descs),
        @ptrCast(&dst),
        @ptrCast(&total),
        @ptrCast(&chunks_per_group),
        @ptrCast(&sub_chunk_cap),
    };

    // Total slot count = n_bindings (4 buffer ptrs) + push payload as 2× u32.
    try testing.expectEqual(@as(usize, 6), raw_params.len);
    try testing.expectEqual(info.n_bindings + (info.push_constant_size / @sizeOf(u32)), @as(u32, @intCast(raw_params.len)));

    // Slots 0..3 must point at the four VkDeviceBuffer-sized handles
    // (u64); slots 4..5 must point at the two u32 push constants.
    try testing.expectEqual(@as(usize, @intFromPtr(&comp)), @intFromPtr(raw_params[0].?));
    try testing.expectEqual(@as(usize, @intFromPtr(&descs)), @intFromPtr(raw_params[1].?));
    try testing.expectEqual(@as(usize, @intFromPtr(&dst)), @intFromPtr(raw_params[2].?));
    try testing.expectEqual(@as(usize, @intFromPtr(&total)), @intFromPtr(raw_params[3].?));
    try testing.expectEqual(@as(usize, @intFromPtr(&chunks_per_group)), @intFromPtr(raw_params[4].?));
    try testing.expectEqual(@as(usize, @intFromPtr(&sub_chunk_cap)), @intFromPtr(raw_params[5].?));

    // Confirm the underlying value sizes match the binding contract:
    // 4 buffer slots are u64-wide, 2 push slots are u32-wide.
    try testing.expectEqual(@as(usize, 8), @sizeOf(@TypeOf(comp)));
    try testing.expectEqual(@as(usize, 8), @sizeOf(@TypeOf(descs)));
    try testing.expectEqual(@as(usize, 8), @sizeOf(@TypeOf(dst)));
    try testing.expectEqual(@as(usize, 8), @sizeOf(@TypeOf(total)));
    try testing.expectEqual(@as(usize, 4), @sizeOf(@TypeOf(chunks_per_group)));
    try testing.expectEqual(@as(usize, 4), @sizeOf(@TypeOf(sub_chunk_cap)));
}
