//! Quick wall-clock + GPU-side perf measurement for the Vulkan L1 codec.
//!
//! Usage:
//!   vk_l1_perf_bench [corpus_path] [--device <N|substring>]
//!
//! Defaults to assets/web.txt and the first compute-capable device.
//!
//! Reports per-trial wall ns (QPC) AND per-trial GPU-side dispatch ns
//! (VkQueryPool TIMESTAMP) for both encode and decode, plus the device
//! name + tier so the same binary captures Intel and NVIDIA runs
//! comparably. The GPU - wall delta isolates per-call PCIe/host overhead
//! from device-side kernel time, which is the key signal for diagnosing
//! TODO P1 (NVIDIA decode ~5× slower than Intel at the wall-clock level).

const std = @import("std");

const driver = @import("driver.zig");
const device_mod = @import("device.zig");
const probe_mod = @import("probe.zig");
const l1_codec = @import("l1_codec.zig");
const spv_blobs = @import("spv_blobs");

const DEFAULT_CORPUS: []const u8 = "assets/web.txt";

const win32 = struct {
    extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.c) c_int;
    extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.c) c_int;
};

fn qpcNow() i64 {
    var c: i64 = 0;
    _ = win32.QueryPerformanceCounter(&c);
    return c;
}

fn qpcNs(from: i64, to: i64) u64 {
    var freq: i64 = 0;
    _ = win32.QueryPerformanceFrequency(&freq);
    if (freq <= 0) freq = 1;
    const delta = if (to > from) to - from else 0;
    const ns = @divTrunc(@as(i128, delta) * 1_000_000_000, @as(i128, freq));
    return @intCast(ns);
}

fn tierName(t: probe_mod.Tier) ?spv_blobs.Tier {
    return switch (t) {
        .tier1 => .tier1,
        .tier1_nv => .tier1_nv,
        .tier2 => .tier2,
        .unsupported => null,
    };
}

fn tierLabel(t: probe_mod.Tier) []const u8 {
    return switch (t) {
        .tier1 => "tier1",
        .tier1_nv => "tier1_nv",
        .tier2 => "tier2",
        .unsupported => "unsupported",
    };
}

pub fn main(process_init: std.process.Init) !void {
    const allocator = process_init.gpa;
    const io = process_init.io;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    // ── Parse argv: [corpus_path] [--device <N|name>] ────────────────
    // No clever flag library — pos-arg-first then a single optional
    // --device flag, in either order; anything we don't understand is
    // reported and the bench falls back to the default for that slot.
    var corpus_path: []const u8 = DEFAULT_CORPUS;
    var argv_it = try process_init.minimal.args.iterateAllocator(allocator);
    defer argv_it.deinit();
    _ = argv_it.next(); // skip exe name
    while (argv_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--device")) {
            const val = argv_it.next() orelse {
                try w.writeAll("error: --device needs an argument\n");
                return;
            };
            // Try integer first, fall back to substring match.
            if (std.fmt.parseInt(u32, val, 10)) |idx| {
                driver.setSelector(.{ .by_index = idx });
            } else |_| {
                driver.setSelector(.{ .by_name = val });
            }
        } else if (std.mem.startsWith(u8, arg, "--")) {
            try w.print("warning: ignoring unknown flag {s}\n", .{arg});
        } else {
            corpus_path = arg;
        }
    }

    // Load corpus.
    var file = std.Io.Dir.cwd().openFile(io, corpus_path, .{}) catch |err| {
        try w.print("error: cannot open {s}: {s}\n", .{ corpus_path, @errorName(err) });
        return;
    };
    defer file.close(io);
    const stat = try file.stat(io);
    const size: usize = @intCast(stat.size);
    const src = try allocator.alloc(u8, size);
    defer allocator.free(src);
    const n_read = try file.readPositionalAll(io, src, 0);
    if (n_read != size) {
        try w.print("error: short read on {s}\n", .{corpus_path});
        return;
    }

    try driver.ensureInit();
    defer driver.deinit();

    const ctx = &driver.g_default;
    const pr = probe_mod.probe(ctx.inst, ctx.pd);
    const tier_b = tierName(pr.tier) orelse {
        try w.writeAll("error: unsupported tier\n");
        return;
    };

    // Header: device + tier + corpus so multi-device runs are unambiguous.
    try w.print(
        "VK_L1_BENCH device=\"{s}\" tier={s} corpus={s} src_bytes={d} has_8bit_storage={}\n",
        .{ pr.device_name, tierLabel(pr.tier), corpus_path, size, ctx.has_8bit_storage },
    );

    const enc_spv_raw = spv_blobs.find("lz_encode", tier_b) orelse {
        try w.writeAll("error: no lz_encode spv\n");
        return;
    };
    const dec_spv_raw = spv_blobs.find("lz_decode", tier_b) orelse {
        try w.writeAll("error: no lz_decode spv\n");
        return;
    };
    // SPV needs 4-byte alignment for vkCreateShaderModule. @embedFile()
    // returns alignment-1 data; copy into aligned buffers.
    const enc_spv = try allocator.alignedAlloc(u8, .@"4", enc_spv_raw.len);
    defer allocator.free(enc_spv);
    @memcpy(enc_spv, enc_spv_raw);
    const dec_spv = try allocator.alignedAlloc(u8, .@"4", dec_spv_raw.len);
    defer allocator.free(dec_spv);
    @memcpy(dec_spv, dec_spv_raw);

    const dst = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dst);
    @memset(dst, 0);

    // Warmup: one encode + decode pass (pipeline create, JIT warm-up).
    {
        var enc = try l1_codec.encodeL1Sync(ctx, src, enc_spv);
        defer l1_codec.freeStreams(ctx, &enc.streams);
        try l1_codec.decodeL1Sync(ctx, enc.streams, src.len, dst[0..src.len], dec_spv);
    }

    // Measured encode: 3 trials, take min for both wall and GPU-side ns.
    var encode_ns_min: u64 = std.math.maxInt(u64);
    var encode_gpu_ns_min: u64 = std.math.maxInt(u64);
    var trial: u32 = 0;
    while (trial < 3) : (trial += 1) {
        const t0 = qpcNow();
        var enc = try l1_codec.encodeL1Sync(ctx, src, enc_spv);
        const t1 = qpcNow();
        defer l1_codec.freeStreams(ctx, &enc.streams);
        const dt = qpcNs(t0, t1);
        if (dt < encode_ns_min) encode_ns_min = dt;
        const gpu = l1_codec.last_encode_dispatch_ns;
        if (gpu < encode_gpu_ns_min) encode_gpu_ns_min = gpu;
    }

    // Measured decode: 3 trials, take min. Each trial uses the same
    // pre-encoded streams so noise is concentrated in the decode path.
    var enc_for_decode = try l1_codec.encodeL1Sync(ctx, src, enc_spv);
    defer l1_codec.freeStreams(ctx, &enc_for_decode.streams);
    var decode_ns_min: u64 = std.math.maxInt(u64);
    var decode_gpu_ns_min: u64 = std.math.maxInt(u64);
    // Per-phase min from the same set of trials (each min taken
    // independently — phase mins are a lower bound on the breakdown, not
    // a single coherent trial's split).
    var phase_alloc_min: u64 = std.math.maxInt(u64);
    var phase_memset_min: u64 = std.math.maxInt(u64);
    var phase_desc_min: u64 = std.math.maxInt(u64);
    var phase_disp_wall_min: u64 = std.math.maxInt(u64);
    var phase_read_min: u64 = std.math.maxInt(u64);
    trial = 0;
    while (trial < 3) : (trial += 1) {
        @memset(dst, 0);
        const t0 = qpcNow();
        try l1_codec.decodeL1Sync(ctx, enc_for_decode.streams, src.len, dst[0..src.len], dec_spv);
        const t1 = qpcNow();
        const dt = qpcNs(t0, t1);
        if (dt < decode_ns_min) decode_ns_min = dt;
        const gpu = l1_codec.last_decode_dispatch_ns;
        if (gpu < decode_gpu_ns_min) decode_gpu_ns_min = gpu;
        if (l1_codec.last_decode_dst_alloc_ns < phase_alloc_min) phase_alloc_min = l1_codec.last_decode_dst_alloc_ns;
        if (l1_codec.last_decode_dst_memset_ns < phase_memset_min) phase_memset_min = l1_codec.last_decode_dst_memset_ns;
        if (l1_codec.last_decode_descriptors_ns < phase_desc_min) phase_desc_min = l1_codec.last_decode_descriptors_ns;
        if (l1_codec.last_decode_dispatch_wall_ns < phase_disp_wall_min) phase_disp_wall_min = l1_codec.last_decode_dispatch_wall_ns;
        if (l1_codec.last_decode_readback_ns < phase_read_min) phase_read_min = l1_codec.last_decode_readback_ns;
    }

    const len_f = @as(f64, @floatFromInt(src.len));
    const enc_ns_per_byte: f64 = @as(f64, @floatFromInt(encode_ns_min)) / len_f;
    const dec_ns_per_byte: f64 = @as(f64, @floatFromInt(decode_ns_min)) / len_f;
    const enc_gpu_ns_per_byte: f64 = @as(f64, @floatFromInt(encode_gpu_ns_min)) / len_f;
    const dec_gpu_ns_per_byte: f64 = @as(f64, @floatFromInt(decode_gpu_ns_min)) / len_f;

    // Original line (preserved for log-scraping callers).
    try w.print(
        "VK_L1_PERF src_bytes={d} encode_ns={d} decode_ns={d} encode_ns_per_byte={d:.3} decode_ns_per_byte={d:.3}\n",
        .{ src.len, encode_ns_min, decode_ns_min, enc_ns_per_byte, dec_ns_per_byte },
    );
    // New line: GPU-side dispatch ns from VkQueryPool TIMESTAMP. The
    // difference (wall - gpu) is per-call host overhead (buffer create +
    // upload + descriptor build + fence wait + readback).
    try w.print(
        "VK_L1_GPU  src_bytes={d} encode_gpu_ns={d} decode_gpu_ns={d} encode_gpu_ns_per_byte={d:.3} decode_gpu_ns_per_byte={d:.3}\n",
        .{ src.len, encode_gpu_ns_min, decode_gpu_ns_min, enc_gpu_ns_per_byte, dec_gpu_ns_per_byte },
    );
    // Host-overhead split makes the PCIe-vs-algorithmic question
    // visually obvious without arithmetic in the caller.
    const enc_host_ns: u64 = if (encode_ns_min > encode_gpu_ns_min) encode_ns_min - encode_gpu_ns_min else 0;
    const dec_host_ns: u64 = if (decode_ns_min > decode_gpu_ns_min) decode_ns_min - decode_gpu_ns_min else 0;
    try w.print(
        "VK_L1_HOST src_bytes={d} encode_host_ns={d} decode_host_ns={d} encode_host_ns_per_byte={d:.3} decode_host_ns_per_byte={d:.3}\n",
        .{
            src.len,
            enc_host_ns,
            dec_host_ns,
            @as(f64, @floatFromInt(enc_host_ns)) / len_f,
            @as(f64, @floatFromInt(dec_host_ns)) / len_f,
        },
    );
    // Decode-phase breakdown — pinpoints which host step dominates.
    // alloc = vkCreateBuffer + vkAllocateMemory + vkMapMemory for dst.
    // memset = host @memset of the (mapped) dst buffer to zero.
    // desc = pipeline cache lookup + descriptor set alloc/write.
    // disp_wall = vkQueueSubmit + vkWaitForFences (kernel + sync drain).
    // read = host @memcpy from mapped dst -> caller's host buffer.
    try w.print(
        "VK_L1_DECODE_PHASES src_bytes={d} alloc_ns={d} memset_ns={d} desc_ns={d} disp_wall_ns={d} read_ns={d}" ++
            " alloc_ns_per_byte={d:.3} memset_ns_per_byte={d:.3} desc_ns_per_byte={d:.3} disp_wall_ns_per_byte={d:.3} read_ns_per_byte={d:.3}\n",
        .{
            src.len,
            phase_alloc_min,
            phase_memset_min,
            phase_desc_min,
            phase_disp_wall_min,
            phase_read_min,
            @as(f64, @floatFromInt(phase_alloc_min)) / len_f,
            @as(f64, @floatFromInt(phase_memset_min)) / len_f,
            @as(f64, @floatFromInt(phase_desc_min)) / len_f,
            @as(f64, @floatFromInt(phase_disp_wall_min)) / len_f,
            @as(f64, @floatFromInt(phase_read_min)) / len_f,
        },
    );
}
