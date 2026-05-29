const std = @import("std");
const builtin = @import("builtin");

// Build steps
// -----------
//   zig build          ReleaseFast streamlz.exe + streamlz_gpu.dll
//   zig build run      Run the CLI on a file
//   zig build test     Run unit tests (parallel runner)
//   zig build gpulib   Build only streamlz_gpu.dll (the C ABI)

pub fn build(b: *std.Build) void {
    // Default x86_64 to `v3` (AVX2 baseline — every Intel since Haswell
    // and AMD since Excavator). Without this, `-mcpu=native` would bake
    // host-specific instructions into the binary and STATUS_ILLEGAL_
    // INSTRUCTION on older or different-vendor CPUs.
    const default_query: std.Target.Query = if (builtin.target.cpu.arch == .x86_64) .{
        .cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 },
    } else .{};
    const target = b.standardTargetOptions(.{ .default_target = default_query });
    const optimize = b.standardOptimizeOption(.{});

    // `-Dstrip=false` keeps debug info even in ReleaseFast so profilers
    // (VTune, samply, etc.) can attribute samples to source lines.
    const strip = b.option(bool, "strip", "Strip debug symbols (default: optimize-mode default)");

    // The CLI links nothing optional at this point. The build options
    // object stays around in case future flags need it.
    const build_options = b.addOptions();

    const ptx_freshness = newPtxFreshnessStep(b);

    // ── streamlz CLI ─────────────────────────────────────────────────────
    const cli_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    cli_module.addOptions("build_options", build_options);

    const exe = b.addExecutable(.{ .name = "streamlz", .root_module = cli_module });
    exe.step.dependOn(ptx_freshness);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the streamlz CLI").dependOn(&run_cmd.step);

    // ── Unit tests ───────────────────────────────────────────────────────
    const test_runner = b.addTest(.{
        .root_module = cli_module,
        .test_runner = .{ .path = b.path("src/test_runner_parallel.zig"), .mode = .simple },
    });
    const run_tests = b.addRunArtifact(test_runner);
    b.step("test", "Run unit tests (parallel)").dependOn(&run_tests.step);
    b.step("ptest", "Run unit tests (parallel, alias for test)").dependOn(&run_tests.step);

    // ── streamlz_gpu.dll — the C ABI shared library ─────────────────────
    // nvCOMP-shaped handle-based compress/decompress that links into
    // game engines, ML pipelines, etc. The library is what
    // tools/build_d2d_bench.bat compiles against.
    const lib_options = b.addOptions();
    const lib_module = b.createModule(.{
        .root_source_file = b.path("src/streamlz_gpu.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    lib_module.addOptions("build_options", lib_options);

    const gpulib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "streamlz_gpu",
        .root_module = lib_module,
    });
    gpulib.step.dependOn(ptx_freshness);
    const gpulib_install = b.addInstallArtifact(gpulib, .{});
    const gpu_hdr_install = b.addInstallHeaderFile(b.path("include/streamlz_gpu.h"), "streamlz_gpu.h");
    const gpulib_step = b.step("gpulib", "Build GPU C API shared library (streamlz_gpu.dll)");
    gpulib_step.dependOn(&gpulib_install.step);
    gpulib_step.dependOn(&gpu_hdr_install.step);
}

// ────────────────────────────────────────────────────────────────────────
//  PTX freshness check
// ────────────────────────────────────────────────────────────────────────
//
// Walks the kernel directories for `.cu`/`.cuh`/`.ptx` files and fails the
// build if any source is newer than any committed PTX. Catches the "I
// edited a kernel but forgot to run tools/build_gpu.bat" mistake that
// would otherwise silently embed stale device code.

const kernel_dirs = [_][]const u8{ "src/encode", "src/decode", "src/common" };

fn newPtxFreshnessStep(b: *std.Build) *std.Build.Step {
    const step = b.allocator.create(std.Build.Step) catch @panic("oom");
    step.* = std.Build.Step.init(.{
        .id = .custom,
        .name = "gpu-ptx-freshness",
        .owner = b,
        .makeFn = ptxFreshnessCheck,
    });
    return step;
}

fn ptxFreshnessCheck(step: *std.Build.Step, opts: std.Build.Step.MakeOptions) anyerror!void {
    _ = opts;
    const b = step.owner;
    const alloc = b.allocator;
    const io = b.graph.io;

    var newest_src_path: ?[]u8 = null;
    var newest_src_mtime: i96 = std.math.minInt(i96);
    var oldest_ptx_path: ?[]u8 = null;
    var oldest_ptx_mtime: i96 = std.math.maxInt(i96);

    for (kernel_dirs) |sub| {
        var dir = b.build_root.handle.openDir(io, sub, .{ .iterate = true }) catch |err| {
            return step.fail("cannot open {s}: {s}", .{ sub, @errorName(err) });
        };
        defer dir.close(io);

        var walker = dir.walk(alloc) catch |err| {
            return step.fail("walk {s} failed: {s}", .{ sub, @errorName(err) });
        };
        defer walker.deinit();

        while (walker.next(io) catch |err| {
            return step.fail("walk {s} next failed: {s}", .{ sub, @errorName(err) });
        }) |entry| {
            if (entry.kind != .file) continue;
            const is_src = std.mem.endsWith(u8, entry.basename, ".cu") or
                std.mem.endsWith(u8, entry.basename, ".cuh");
            const is_ptx = std.mem.endsWith(u8, entry.basename, ".ptx");
            if (!is_src and !is_ptx) continue;

            const stat = entry.dir.statFile(io, entry.basename, .{}) catch continue;
            const mtime_ns: i96 = stat.mtime.toNanoseconds();
            if (is_src and mtime_ns > newest_src_mtime) {
                if (newest_src_path) |p| alloc.free(p);
                newest_src_path = std.fmt.allocPrint(alloc, "{s}/{s}", .{ sub, entry.path }) catch unreachable;
                newest_src_mtime = mtime_ns;
            }
            if (is_ptx and mtime_ns < oldest_ptx_mtime) {
                if (oldest_ptx_path) |p| alloc.free(p);
                oldest_ptx_path = std.fmt.allocPrint(alloc, "{s}/{s}", .{ sub, entry.path }) catch unreachable;
                oldest_ptx_mtime = mtime_ns;
            }
        }
    }

    if (newest_src_path == null) return;
    if (oldest_ptx_path == null) {
        return step.fail(
            "no .ptx files found under src/encode | src/decode | src/common.\n" ++
                "Run tools\\build_gpu.bat to compile every kernel and rebuild the exe.",
            .{},
        );
    }
    if (newest_src_mtime > oldest_ptx_mtime) {
        return step.fail(
            "PTX is stale relative to .cu/.cuh sources.\n" ++
                "  newest source: {s}\n" ++
                "  oldest PTX:    {s}\n" ++
                "Run tools\\build_gpu.bat to rebuild every kernel + the embedding exe.",
            .{ newest_src_path.?, oldest_ptx_path.? },
        );
    }
}
