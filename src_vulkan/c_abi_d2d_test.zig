//! Phase-2 (TODO A2) D2D test: exercises the BDA-based device-to-device
//! compress + decompress paths added in Phase 2. Wired as
//! `zig build vk-l1-d2d-test`.
//!
//! Scenario (per direction):
//!   1. Allocate a caller-owned VkBuffer with VK_BUFFER_USAGE_STORAGE +
//!      _SHADER_DEVICE_ADDRESS + transfer flags; back with
//!      HOST_VISIBLE+HOST_COHERENT memory so the test can fill / read
//!      bytes without staging.
//!   2. Fill the source buffer with input bytes from `assets/web.txt`.
//!   3. Call slzRegisterBuffer_vk to register both buffers on the
//!      handle. Pass d_base_address = NULL so the codec queries BDA.
//!   4. Query each buffer's BDA via slzBufferGetDeviceAddress_vk.
//!   5. Call slzCompress_vk with the BDA as `d_input` (and host buffer
//!      for `d_output`). Asserts byte-identical to the host-only
//!      reference output. (Compress-into-device buffer is also
//!      exercised via the host-bounce staging path the codec uses
//!      for output D2D.)
//!   6. Call slzDecompress_vk with the BDA as `d_output`. Asserts
//!      byte-identical to source.
//!
//! Prints PASS/FAIL per case + a summary line; CI scrapes the summary.

const std = @import("std");
const builtin = @import("builtin");

const vk_abi = @import("streamlz_gpu_vk.zig");
const driver = @import("driver.zig");
const vk = @import("vk_api.zig");

const CORPUS_PATH: []const u8 = "assets/web.txt";
const PREFIX_SIZE: usize = 128 * 1024; // 128 KiB — multi-chunk + small enough to be quick.

const Allocator = std.mem.Allocator;

const CaseResult = struct {
    name: []const u8,
    passed: bool,
    detail: []const u8,
};

fn loadCorpus(allocator: Allocator, io: std.Io) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, CORPUS_PATH, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const want: usize = @min(PREFIX_SIZE, @as(usize, @intCast(stat.size)));
    const buf = try allocator.alloc(u8, want);
    errdefer allocator.free(buf);
    const n = try file.readPositionalAll(io, buf, 0);
    if (n != want) return error.ShortRead;
    return buf;
}

/// Allocate a HOST_VISIBLE+HOST_COHERENT VkBuffer with the usage
/// flags needed for the D2D test: STORAGE_BUFFER (so the codec
/// binds it into descriptor sets), TRANSFER_SRC+TRANSFER_DST (so
/// staging copies in either direction work), SHADER_DEVICE_ADDRESS
/// (so vkGetBufferDeviceAddress returns non-zero). Returns the
/// VkBuffer + VkDeviceMemory handles + mapped pointer.
const TestBuf = struct {
    buf: vk.VkBuffer = null,
    mem: vk.VkDeviceMemory = null,
    mapped: ?[*]u8 = null,
};

fn createTestBuf(ctx: *driver.Context, size: usize) !TestBuf {
    // Lazy-resolve every fn we'll need. Most are already resolved
    // by the codec's first call but the test runs them up-front.
    const resolve = struct {
        fn fnDev(comptime T: type, dev: vk.VkDevice, name: [*:0]const u8) ?T {
            if (vk.vkGetDeviceProcAddr_fn) |gdpa| {
                if (gdpa(dev, name)) |raw| return @ptrCast(@alignCast(raw));
            }
            return null;
        }
    };
    if (vk.vkCreateBuffer_fn == null) vk.vkCreateBuffer_fn = resolve.fnDev(vk.FnCreateBuffer, ctx.dev, "vkCreateBuffer");
    if (vk.vkDestroyBuffer_fn == null) vk.vkDestroyBuffer_fn = resolve.fnDev(vk.FnDestroyBuffer, ctx.dev, "vkDestroyBuffer");
    if (vk.vkAllocateMemory_fn == null) vk.vkAllocateMemory_fn = resolve.fnDev(vk.FnAllocateMemory, ctx.dev, "vkAllocateMemory");
    if (vk.vkFreeMemory_fn == null) vk.vkFreeMemory_fn = resolve.fnDev(vk.FnFreeMemory, ctx.dev, "vkFreeMemory");
    if (vk.vkBindBufferMemory_fn == null) vk.vkBindBufferMemory_fn = resolve.fnDev(vk.FnBindBufferMemory, ctx.dev, "vkBindBufferMemory");
    if (vk.vkMapMemory_fn == null) vk.vkMapMemory_fn = resolve.fnDev(vk.FnMapMemory, ctx.dev, "vkMapMemory");
    if (vk.vkUnmapMemory_fn == null) vk.vkUnmapMemory_fn = resolve.fnDev(vk.FnUnmapMemory, ctx.dev, "vkUnmapMemory");
    if (vk.vkGetBufferMemoryRequirements_fn == null) vk.vkGetBufferMemoryRequirements_fn = resolve.fnDev(vk.FnGetBufferMemoryRequirements, ctx.dev, "vkGetBufferMemoryRequirements");
    if (vk.vkGetPhysicalDeviceMemoryProperties_fn == null) {
        if (vk.vkGetInstanceProcAddr_fn) |gipa| {
            if (gipa(driver.g_default.inst, "vkGetPhysicalDeviceMemoryProperties")) |raw| {
                vk.vkGetPhysicalDeviceMemoryProperties_fn = @ptrCast(@alignCast(raw));
            }
        }
    }

    const usage: vk.VkBufferUsageFlags = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    const bci: vk.VkBufferCreateInfo = .{
        .size = @max(@as(vk.VkDeviceSize, 4), size),
        .usage = usage,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
    };
    var buf: vk.VkBuffer = null;
    if (vk.vkCreateBuffer_fn.?(ctx.dev, &bci, null, &buf) != vk.VK_SUCCESS) return error.BufferCreateFailed;
    errdefer if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);

    var req: vk.VkMemoryRequirements = .{};
    vk.vkGetBufferMemoryRequirements_fn.?(ctx.dev, buf, &req);

    var mem_props: vk.VkPhysicalDeviceMemoryProperties = .{};
    vk.vkGetPhysicalDeviceMemoryProperties_fn.?(ctx.pd, &mem_props);

    const want = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    var mt_idx: u32 = std.math.maxInt(u32);
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        const supported = (req.memoryTypeBits & (@as(u32, 1) << @intCast(i))) != 0;
        if (supported and (mem_props.memoryTypes[i].propertyFlags & want) == want) {
            mt_idx = i;
            break;
        }
    }
    if (mt_idx == std.math.maxInt(u32)) return error.MemoryTypeNotFound;

    // BDA buffers need VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT chained
    // on the allocation. Without it, vkGetBufferDeviceAddress returns
    // 0 and the registry path bails with SLZ_ERROR_VK_FEATURE_MISSING.
    const alloc_flags: vk.VkMemoryAllocateFlagsInfo = .{
        .flags = vk.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    const mai: vk.VkMemoryAllocateInfo = .{
        .pNext = @ptrCast(&alloc_flags),
        .allocationSize = req.size,
        .memoryTypeIndex = mt_idx,
    };
    var mem: vk.VkDeviceMemory = null;
    if (vk.vkAllocateMemory_fn.?(ctx.dev, &mai, null, &mem) != vk.VK_SUCCESS) return error.AllocFailed;
    errdefer if (vk.vkFreeMemory_fn) |f| f(ctx.dev, mem, null);
    if (vk.vkBindBufferMemory_fn.?(ctx.dev, buf, mem, 0) != vk.VK_SUCCESS) return error.BindFailed;

    var raw: ?*anyopaque = null;
    if (vk.vkMapMemory_fn.?(ctx.dev, mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) != vk.VK_SUCCESS) return error.MapFailed;
    return .{ .buf = buf, .mem = mem, .mapped = @ptrCast(@alignCast(raw.?)) };
}

fn destroyTestBuf(ctx: *driver.Context, b: *TestBuf) void {
    if (b.mapped != null) {
        if (vk.vkUnmapMemory_fn) |u| u(ctx.dev, b.mem);
        b.mapped = null;
    }
    if (b.buf != null) {
        if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, b.buf, null);
        b.buf = null;
    }
    if (b.mem != null) {
        if (vk.vkFreeMemory_fn) |f| f(ctx.dev, b.mem, null);
        b.mem = null;
    }
}

pub fn main(process_init: std.process.Init) !void {
    const io = process_init.io;
    const allocator = std.heap.c_allocator;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    var results: std.ArrayList(CaseResult) = .empty;
    defer results.deinit(allocator);

    // ── Common: load corpus + create handle ──────────────────────────
    const src = loadCorpus(allocator, io) catch |err| {
        try w.print("D2D FAIL stage=load_corpus err={s}\n", .{@errorName(err)});
        return error.LoadFailed;
    };
    defer allocator.free(src);

    var handle: ?*vk_abi.VkContext = null;
    if (vk_abi.slzCreate_vk(&handle) != 0 or handle == null) {
        try w.writeAll("D2D FAIL stage=create\n");
        return error.CreateFailed;
    }
    defer vk_abi.slzDestroy_vk(handle);

    // Establish a host-pointer-path reference output for byte-equal
    // comparison below.
    const bound = vk_abi.slzCompressBound_vk(src.len);
    const ref_slz = try allocator.alloc(u8, bound);
    defer allocator.free(ref_slz);
    const ref_cz = vk_abi.slzCompressHost_vk(handle, src.ptr, src.len, ref_slz.ptr, ref_slz.len, .{ .level = 1 });
    if (ref_cz < 0) {
        try w.print("D2D FAIL stage=ref_compress rc={d}\n", .{ref_cz});
        return error.RefCompressFailed;
    }
    const ref_cz_bytes: usize = @intCast(ref_cz);

    const ctx = &driver.g_default;

    // ── Case 1: D2D encode input ─────────────────────────────────
    // Allocate src VkBuffer, fill with bytes, register, compress
    // with the BDA address as d_input. Output stays on host pointer.
    {
        var src_buf = try createTestBuf(ctx, src.len);
        defer destroyTestBuf(ctx, &src_buf);
        @memcpy(src_buf.mapped.?[0..src.len], src);

        // Register; null d_base_address triggers our internal BDA query.
        const reg_rc = vk_abi.slzRegisterBuffer_vk(handle, @ptrCast(src_buf.buf), null, src.len);
        if (reg_rc != 0) {
            try results.append(allocator, .{ .name = "encode_input_d2d", .passed = false, .detail = "register failed" });
        } else {
            const dev_addr = vk_abi.slzBufferGetDeviceAddress_vk(handle, @ptrCast(src_buf.buf));
            if (dev_addr == 0) {
                try results.append(allocator, .{ .name = "encode_input_d2d", .passed = false, .detail = "BDA returned 0" });
            } else {
                const dev_ptr: *const anyopaque = @ptrFromInt(dev_addr);
                const d2d_slz = try allocator.alloc(u8, bound);
                defer allocator.free(d2d_slz);
                const rc = vk_abi.slzCompress_vk(handle, dev_ptr, src.len, d2d_slz.ptr, d2d_slz.len, .{ .level = 1 });
                if (rc < 0) {
                    try results.append(allocator, .{ .name = "encode_input_d2d", .passed = false, .detail = "compress failed" });
                } else {
                    const d2d_cz: usize = @intCast(rc);
                    const ok = d2d_cz == ref_cz_bytes and std.mem.eql(u8, ref_slz[0..ref_cz_bytes], d2d_slz[0..d2d_cz]);
                    try results.append(allocator, .{
                        .name = "encode_input_d2d",
                        .passed = ok,
                        .detail = if (ok) "byte-equal vs host path" else "byte mismatch vs host path",
                    });
                }
            }
            _ = vk_abi.slzUnregisterBuffer_vk(handle, @ptrFromInt(dev_addr));
        }
    }

    // ── Case 2: D2D decode output ────────────────────────────────
    // Reference: decompress through host-pointer path.
    const ref_dst = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(ref_dst);
    @memset(ref_dst, 0);
    const ref_dz = vk_abi.slzDecompressHost_vk(handle, ref_slz.ptr, ref_cz_bytes, ref_dst.ptr, ref_dst.len, .{});
    if (ref_dz < 0) {
        try w.print("D2D FAIL stage=ref_decompress rc={d}\n", .{ref_dz});
        return error.RefDecompressFailed;
    }
    const ref_dz_bytes: usize = @intCast(ref_dz);

    {
        var dst_buf = try createTestBuf(ctx, src.len);
        defer destroyTestBuf(ctx, &dst_buf);

        const reg_rc = vk_abi.slzRegisterBuffer_vk(handle, @ptrCast(dst_buf.buf), null, src.len);
        if (reg_rc != 0) {
            try results.append(allocator, .{ .name = "decode_output_d2d", .passed = false, .detail = "register failed" });
        } else {
            const dev_addr = vk_abi.slzBufferGetDeviceAddress_vk(handle, @ptrCast(dst_buf.buf));
            if (dev_addr == 0) {
                try results.append(allocator, .{ .name = "decode_output_d2d", .passed = false, .detail = "BDA returned 0" });
            } else {
                const dev_out_ptr: *anyopaque = @ptrFromInt(dev_addr);
                const rc = vk_abi.slzDecompress_vk(
                    handle,
                    ref_slz.ptr,
                    ref_cz_bytes,
                    dev_out_ptr,
                    src.len,
                    .{},
                );
                if (rc < 0) {
                    try results.append(allocator, .{ .name = "decode_output_d2d", .passed = false, .detail = "decompress failed" });
                } else {
                    const decoded: usize = @intCast(rc);
                    // Bytes are in `dst_buf.mapped`; compare to ref.
                    const ok = decoded == ref_dz_bytes and
                        std.mem.eql(u8, src, dst_buf.mapped.?[0..src.len]);
                    try results.append(allocator, .{
                        .name = "decode_output_d2d",
                        .passed = ok,
                        .detail = if (ok) "byte-equal vs source" else "byte mismatch vs source",
                    });
                }
            }
            _ = vk_abi.slzUnregisterBuffer_vk(handle, @ptrFromInt(dev_addr));
        }
    }

    // ── Case 3: Round-trip D2D — encode input D2D, decode output D2D ──
    // Encode src VkBuffer → host slz; then host slz → dst VkBuffer.
    {
        var src_buf = try createTestBuf(ctx, src.len);
        defer destroyTestBuf(ctx, &src_buf);
        @memcpy(src_buf.mapped.?[0..src.len], src);
        var dst_buf = try createTestBuf(ctx, src.len);
        defer destroyTestBuf(ctx, &dst_buf);

        var ok_outer = true;
        const src_reg_rc = vk_abi.slzRegisterBuffer_vk(handle, @ptrCast(src_buf.buf), null, src.len);
        const dst_reg_rc = vk_abi.slzRegisterBuffer_vk(handle, @ptrCast(dst_buf.buf), null, src.len);
        if (src_reg_rc != 0 or dst_reg_rc != 0) ok_outer = false;
        const src_addr = vk_abi.slzBufferGetDeviceAddress_vk(handle, @ptrCast(src_buf.buf));
        const dst_addr = vk_abi.slzBufferGetDeviceAddress_vk(handle, @ptrCast(dst_buf.buf));
        if (src_addr == 0 or dst_addr == 0) ok_outer = false;

        if (!ok_outer) {
            try results.append(allocator, .{ .name = "roundtrip_d2d", .passed = false, .detail = "register/BDA failed" });
        } else {
            const slz_scratch = try allocator.alloc(u8, bound);
            defer allocator.free(slz_scratch);

            const src_ptr: *const anyopaque = @ptrFromInt(src_addr);
            const cz = vk_abi.slzCompress_vk(handle, src_ptr, src.len, slz_scratch.ptr, slz_scratch.len, .{ .level = 1 });

            const dst_ptr: *anyopaque = @ptrFromInt(dst_addr);
            const dz = vk_abi.slzDecompress_vk(handle, slz_scratch.ptr, @intCast(cz), dst_ptr, src.len, .{});

            const ok = cz >= 0 and dz >= 0 and @as(usize, @intCast(dz)) == src.len and
                std.mem.eql(u8, src, dst_buf.mapped.?[0..src.len]);
            try results.append(allocator, .{
                .name = "roundtrip_d2d",
                .passed = ok,
                .detail = if (ok) "byte-equal source -> compress D2D -> decompress D2D" else "round-trip mismatch",
            });
            _ = vk_abi.slzUnregisterBuffer_vk(handle, @ptrFromInt(src_addr));
            _ = vk_abi.slzUnregisterBuffer_vk(handle, @ptrFromInt(dst_addr));
        }
    }

    // ── Case 4: Scribble proof — verify the codec reads CURRENT device-
    // buffer contents, NOT the host bytes the caller passed at registration.
    //
    // Cluster F (F034) — cases 1 and 3 above byte-compare against a
    // reference whose source is the SAME bytes the test wrote into the
    // device buffer via the host map. If the codec silently dropped the
    // src_buffer_override and re-uploaded host bytes from `d_input` (the
    // BDA cast back to a pointer), the test would either crash on a bad
    // host dereference OR — worse — accidentally pass because the bytes
    // happened to be the same. The scribble proof closes that gap:
    //
    //   1. Fill the device buffer with bytes A (the corpus prefix),
    //      register it, compress → assert equals ref_compress(A).
    //   2. Overwrite the device buffer (host map is HOST_COHERENT so the
    //      next compress sees the new bytes immediately) with a
    //      RECOGNIZABLE pattern B (alternating 0xAA / 0x55), compress
    //      AGAIN → assert equals ref_compress(B).
    //   3. The same registered VkBuffer is used for both compresses. The
    //      codec MUST observe the buffer's CURRENT contents both times.
    //      A codec that secretly cached the address-A bytes from
    //      registration time, or that aliased the host pointer that
    //      `src` was originally backed by, would fail step 2.
    //
    // Asserts BOTH compresses match their respective host-path references
    // AND that the two D2D outputs differ — pattern B is wildly more
    // compressible than the natural-text corpus prefix, so equal outputs
    // would be a flagrant red flag for either the codec or the test.
    {
        var src_buf = try createTestBuf(ctx, src.len);
        defer destroyTestBuf(ctx, &src_buf);

        // Step 1: corpus-prefix bytes → device buffer; register; compress.
        @memcpy(src_buf.mapped.?[0..src.len], src);
        const reg_rc = vk_abi.slzRegisterBuffer_vk(handle, @ptrCast(src_buf.buf), null, src.len);
        if (reg_rc != 0) {
            try results.append(allocator, .{
                .name = "encode_input_d2d_scribble",
                .passed = false,
                .detail = "register failed",
            });
        } else {
            const dev_addr = vk_abi.slzBufferGetDeviceAddress_vk(handle, @ptrCast(src_buf.buf));
            if (dev_addr == 0) {
                try results.append(allocator, .{
                    .name = "encode_input_d2d_scribble",
                    .passed = false,
                    .detail = "BDA returned 0",
                });
            } else {
                const dev_ptr: *const anyopaque = @ptrFromInt(dev_addr);

                // ── Compress A: corpus bytes already loaded ──────────────
                const d2d_a = try allocator.alloc(u8, bound);
                defer allocator.free(d2d_a);
                const rc_a = vk_abi.slzCompress_vk(
                    handle,
                    dev_ptr,
                    src.len,
                    d2d_a.ptr,
                    d2d_a.len,
                    .{ .level = 1 },
                );

                // ── Scribble: overwrite the device buffer with the
                //    AA/55 pattern via the HOST_COHERENT mapping. The next
                //    compress through the SAME registered buffer should
                //    see the new bytes — `ref_slz`'s host bytes are gone.
                {
                    var i: usize = 0;
                    while (i < src.len) : (i += 1) {
                        src_buf.mapped.?[i] = if ((i & 1) == 0) 0xAA else 0x55;
                    }
                }
                // Build a host scratch carrying the SAME scribble pattern
                // so the reference path produces ref_compress(B).
                const scribble = try allocator.alloc(u8, src.len);
                defer allocator.free(scribble);
                {
                    var i: usize = 0;
                    while (i < src.len) : (i += 1) {
                        scribble[i] = if ((i & 1) == 0) 0xAA else 0x55;
                    }
                }
                const ref_b = try allocator.alloc(u8, bound);
                defer allocator.free(ref_b);
                const ref_b_rc = vk_abi.slzCompressHost_vk(
                    handle,
                    scribble.ptr,
                    scribble.len,
                    ref_b.ptr,
                    ref_b.len,
                    .{ .level = 1 },
                );

                // ── Compress B: device buffer now holds AA/55 ────────────
                const d2d_b = try allocator.alloc(u8, bound);
                defer allocator.free(d2d_b);
                const rc_b = vk_abi.slzCompress_vk(
                    handle,
                    dev_ptr,
                    src.len,
                    d2d_b.ptr,
                    d2d_b.len,
                    .{ .level = 1 },
                );

                const ok_each_compress = (rc_a >= 0) and (rc_b >= 0) and (ref_b_rc >= 0);
                const a_matches_ref = ok_each_compress and
                    @as(usize, @intCast(rc_a)) == ref_cz_bytes and
                    std.mem.eql(u8, ref_slz[0..ref_cz_bytes], d2d_a[0..@intCast(rc_a)]);
                const b_matches_ref = ok_each_compress and
                    @as(usize, @intCast(rc_b)) == @as(usize, @intCast(ref_b_rc)) and
                    std.mem.eql(u8, ref_b[0..@intCast(ref_b_rc)], d2d_b[0..@intCast(rc_b)]);
                // A != B is a sanity check on the scribble pattern itself
                // — alternating AA/55 must compress to something other
                // than the natural-text corpus prefix. If the two outputs
                // matched, the codec is broken (or the test's scribble
                // somehow coincided with the source, which is impossible
                // because the corpus is real bytes).
                const distinct = ok_each_compress and
                    @as(usize, @intCast(rc_a)) != @as(usize, @intCast(rc_b));

                const ok = a_matches_ref and b_matches_ref and distinct;
                try results.append(allocator, .{
                    .name = "encode_input_d2d_scribble",
                    .passed = ok,
                    .detail = if (ok)
                        "device-buffer contents drove output for both A (corpus) and B (AA/55 scribble)"
                    else if (!a_matches_ref)
                        "compress(A) on D2D buffer != ref_compress(corpus)"
                    else if (!b_matches_ref)
                        "compress(B) on scribbled D2D buffer != ref_compress(AA/55)"
                    else
                        "scribbled D2D output identical to corpus output (codec ignored scribble?)",
                });
            }
            _ = vk_abi.slzUnregisterBuffer_vk(handle, @ptrFromInt(dev_addr));
        }
    }

    // ── Report ──────────────────────────────────────────────────────
    var pass_count: u32 = 0;
    var fail_count: u32 = 0;
    for (results.items) |r| {
        if (r.passed) {
            pass_count += 1;
            try w.print("D2D PASS {s} ({s})\n", .{ r.name, r.detail });
        } else {
            fail_count += 1;
            try w.print("D2D FAIL {s} ({s})\n", .{ r.name, r.detail });
        }
    }
    try w.print(
        "D2D SUMMARY pass={d} fail={d} total={d}\n",
        .{ pass_count, fail_count, pass_count + fail_count },
    );
    if (fail_count > 0) {
        try w.flush();
        std.process.exit(1);
    }
    _ = builtin;
}
