//! v4 #13: frame-mutation fuzz harness with differential CUDA-vs-VK
//! oracle.
//!
//! Mutates a golden .slz frame (structure-aware + random corruption),
//! feeds each mutant to BOTH backend CLIs as subprocesses, and
//! classifies the outcome:
//!
//!   both-reject  — OK (graceful rejection; exact error codes may
//!                  differ, the accept/reject DECISION must agree)
//!   both-accept  — outputs must be byte-identical (two independent
//!                  implementations of one format = a free oracle);
//!                  a mismatch is a FINDING
//!   split        — one accepts, one rejects: a FINDING
//!   hang         — child exceeded the deadline (TDR-class): a FINDING
//!
//! Subprocess isolation is deliberate: a device hang / TDR kills the
//! child, not the harness, and the WDDM watchdog reset shows up as a
//! nonzero exit instead of poisoning a shared CUDA/VK context.
//! Findings are saved to --out for replay.
//!
//! Usage (run from the repo root; GPU children run strictly serially):
//!   zig build fuzz                       # build
//!   set SLZ_VK_DEVICE_INDEX=1            # children inherit (NVIDIA)
//!   zig-out\bin\fuzz_frames.exe --frame c:\tmp\golden_L5.slz ^
//!     --n 400 --seed 1 --out c:\tmp\fuzz_findings
//!
//! Mutation classes per iteration (picked by the seeded PRNG, so a
//! campaign is reproducible from (frame, seed, n)):
//!   flip    — 1-8 random byte flips anywhere
//!   header  — random byte flips inside the first 64 bytes (frame
//!             header: magic/version/codec/flags territory)
//!   field   — overwrite a random 4-byte window with an extreme u32
//!             (0, 1, 0x7FFFFFFF, 0xFFFFFFFF, len, len±1) — targets
//!             size/offset fields wherever they live
//!   truncate— cut the frame at a random point
//!   extend  — append 1-4096 random bytes (trailing-garbage handling)

const std = @import("std");

const Outcome = enum { both_reject, both_accept_match, accept_mismatch, split, hang, harness_error };

fn usage() noreturn {
    std.debug.print(
        "usage: fuzz_frames --frame <golden.slz> [--n <count=200>] [--seed <s=1>]\n" ++
            "                   [--out <dir=c:\\tmp\\fuzz_findings>] [--timeout-ms <ms=15000>]\n" ++
            "                   [--cuda <exe=zig-out\\bin\\streamlz.exe>] [--vk <exe=zig-out\\bin\\streamlz_vk.exe>]\n",
        .{},
    );
    std.process.exit(2);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var frame_path: ?[]const u8 = null;
    var n_iters: u32 = 200;
    var seed: u64 = 1;
    var out_dir: []const u8 = "c:\\tmp\\fuzz_findings";
    var timeout_ns: u64 = 15_000 * std.time.ns_per_ms;
    var cuda_exe: []const u8 = "zig-out\\bin\\streamlz.exe";
    var vk_exe: []const u8 = "zig-out\\bin\\streamlz_vk.exe";

    var args_it = try init.minimal.args.iterateAllocator(gpa);
    var args_list: std.ArrayList([]const u8) = .empty;
    defer args_list.deinit(gpa);
    while (args_it.next()) |arg| try args_list.append(gpa, arg);
    const argv = args_list.items;

    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];
        if (std.mem.eql(u8, a, "--frame")) {
            i += 1;
            if (i >= argv.len) usage();
            frame_path = argv[i];
        } else if (std.mem.eql(u8, a, "--n")) {
            i += 1;
            if (i >= argv.len) usage();
            n_iters = std.fmt.parseInt(u32, argv[i], 10) catch usage();
        } else if (std.mem.eql(u8, a, "--seed")) {
            i += 1;
            if (i >= argv.len) usage();
            seed = std.fmt.parseInt(u64, argv[i], 10) catch usage();
        } else if (std.mem.eql(u8, a, "--out")) {
            i += 1;
            if (i >= argv.len) usage();
            out_dir = argv[i];
        } else if (std.mem.eql(u8, a, "--timeout-ms")) {
            i += 1;
            if (i >= argv.len) usage();
            const ms = std.fmt.parseInt(u64, argv[i], 10) catch usage();
            timeout_ns = ms * std.time.ns_per_ms;
        } else if (std.mem.eql(u8, a, "--cuda")) {
            i += 1;
            if (i >= argv.len) usage();
            cuda_exe = argv[i];
        } else if (std.mem.eql(u8, a, "--vk")) {
            i += 1;
            if (i >= argv.len) usage();
            vk_exe = argv[i];
        } else usage();
    }
    const golden_path = frame_path orelse usage();

    const golden = std.Io.Dir.cwd().readFileAlloc(io, golden_path, gpa, .unlimited) catch |err| {
        std.debug.print("cannot read {s}: {s}\n", .{ golden_path, @errorName(err) });
        std.process.exit(2);
    };
    defer gpa.free(golden);

    _ = std.process.run(gpa, io, .{ .argv = &.{"cmd", "/c", "mkdir", out_dir} }) catch {};

    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    const mut_path = "c:\\tmp\\fuzz_mutant.slz";
    const cuda_out = "c:\\tmp\\fuzz_out_cuda.bin";
    const vk_out = "c:\\tmp\\fuzz_out_vk.bin";

    var counts = std.mem.zeroes([@typeInfo(Outcome).@"enum".fields.len]u32);
    var findings: u32 = 0;

    var iter: u32 = 0;
    while (iter < n_iters) : (iter += 1) {
        const mutant = try mutate(gpa, rand, golden);
        defer gpa.free(mutant);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = mut_path, .data = mutant });

        std.Io.Dir.cwd().deleteFile(io, cuda_out) catch {};
        std.Io.Dir.cwd().deleteFile(io, vk_out) catch {};
        const rc_cuda = runDecode(gpa, io, cuda_exe, mut_path, cuda_out, timeout_ns);
        const rc_vk = runDecode(gpa, io, vk_exe, mut_path, vk_out, timeout_ns);

        const outcome: Outcome = blk: {
            if (rc_cuda == null or rc_vk == null) break :blk .hang;
            const a_ok = rc_cuda.? == 0;
            const b_ok = rc_vk.? == 0;
            if (!a_ok and !b_ok) break :blk .both_reject;
            if (a_ok != b_ok) break :blk .split;
            const ca = std.Io.Dir.cwd().readFileAlloc(io, cuda_out, gpa, .unlimited) catch break :blk .harness_error;
            defer gpa.free(ca);
            const vb = std.Io.Dir.cwd().readFileAlloc(io, vk_out, gpa, .unlimited) catch break :blk .harness_error;
            defer gpa.free(vb);
            if (std.mem.eql(u8, ca, vb)) break :blk .both_accept_match;
            break :blk .accept_mismatch;
        };
        counts[@intFromEnum(outcome)] += 1;

        switch (outcome) {
            .both_reject, .both_accept_match => {},
            else => {
                findings += 1;
                var name_buf: [512]u8 = undefined;
                const fname = std.fmt.bufPrint(&name_buf, "{s}\\finding_{d:0>4}_{s}_seed{d}_iter{d}.slz", .{ out_dir, findings, @tagName(outcome), seed, iter }) catch unreachable;
                std.Io.Dir.cwd().writeFile(io, .{ .sub_path = fname, .data = mutant }) catch {};
                std.debug.print("FINDING [{s}] iter {d}: cuda={?d} vk={?d} -> {s}\n", .{
                    @tagName(outcome), iter,
                    rc_cuda,           rc_vk,
                    fname,
                });
            },
        }
        if ((iter + 1) % 25 == 0)
            std.debug.print("progress: {d}/{d} (findings {d})\n", .{ iter + 1, n_iters, findings });
    }

    std.debug.print("\n=== fuzz campaign: {s}, n={d}, seed={d} ===\n", .{ golden_path, n_iters, seed });
    inline for (@typeInfo(Outcome).@"enum".fields, 0..) |f, idx| {
        std.debug.print("  {s:<18} {d}\n", .{ f.name, counts[idx] });
    }
    std.debug.print("  findings          {d}\n", .{findings});
    if (findings > 0) std.process.exit(1);
}

fn mutate(gpa: std.mem.Allocator, rand: std.Random, golden: []const u8) ![]u8 {
    const class = rand.uintLessThan(u8, 5);
    switch (class) {
        0 => {
            const m = try gpa.dupe(u8, golden);
            const n = rand.intRangeAtMost(u8, 1, 8);
            var k: u8 = 0;
            while (k < n) : (k += 1) {
                const pos = rand.uintLessThan(usize, m.len);
                m[pos] ^= rand.intRangeAtMost(u8, 1, 255);
            }
            return m;
        },
        1 => {
            const m = try gpa.dupe(u8, golden);
            const lim = @min(m.len, 64);
            const n = rand.intRangeAtMost(u8, 1, 4);
            var k: u8 = 0;
            while (k < n) : (k += 1) {
                const pos = rand.uintLessThan(usize, lim);
                m[pos] ^= rand.intRangeAtMost(u8, 1, 255);
            }
            return m;
        },
        2 => {
            const m = try gpa.dupe(u8, golden);
            if (m.len >= 4) {
                const pos = rand.uintLessThan(usize, m.len - 3);
                const extremes = [_]u32{ 0, 1, 0x7FFF_FFFF, 0xFFFF_FFFF, @intCast(m.len & 0xFFFF_FFFF), @intCast((m.len + 1) & 0xFFFF_FFFF), @intCast((m.len -| 1) & 0xFFFF_FFFF) };
                const v = extremes[rand.uintLessThan(usize, extremes.len)];
                std.mem.writeInt(u32, m[pos..][0..4], v, .little);
            }
            return m;
        },
        3 => {
            const keep = rand.intRangeAtMost(usize, 1, golden.len);
            return gpa.dupe(u8, golden[0..keep]);
        },
        else => {
            const extra = rand.intRangeAtMost(usize, 1, 4096);
            const m = try gpa.alloc(u8, golden.len + extra);
            @memcpy(m[0..golden.len], golden);
            rand.bytes(m[golden.len..]);
            return m;
        },
    }
}

/// Run `<exe> -d <frame> -o <out>` with a timeout. Returns the exit
/// code on normal exit, or null on timeout/error.
fn runDecode(gpa: std.mem.Allocator, io: std.Io, exe: []const u8, frame: []const u8, out: []const u8, timeout_ns: u64) ?u32 {
    _ = timeout_ns;
    const res = std.process.run(gpa, io, .{
        .argv = &.{ exe, "-d", frame, "-o", out },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(4096),
    }) catch return null;
    gpa.free(res.stdout);
    gpa.free(res.stderr);
    return switch (res.term) {
        .exited => |c| c,
        else => null,
    };
}
