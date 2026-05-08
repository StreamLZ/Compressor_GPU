const std = @import("std");
const builtin = @import("builtin");

// Build steps
// -----------
//   zig build          Default ReleaseFast build + install
//   zig build run      Run the streamlz CLI
//   zig build test     Run unit tests
//   zig build safe     Build with ReleaseSafe (bounds + overflow checks)
//   zig build fuzz     Build fuzz_decompress harness (ReleaseSafe)
//                      Usage: zig-out/bin/fuzz-decompress <input-file>

pub fn build(b: *std.Build) void {
    // Default the x86_64 CPU model to `x86_64_v3` (AVX2 baseline,
    // covering all Intel since Haswell 2013 and all AMD since
    // Excavator 2015). Without this, Zig defaults to `-mcpu=native`
    // which produces a binary that uses host-CPU-specific instructions
    // (BMI2, ADX, AVX-VNNI, etc.) and crashes with STATUS_ILLEGAL_
    // INSTRUCTION (0xc000001d) on older or different-vendor CPUs.
    //
    // Override with `-Dcpu=native` for maximum local perf, or
    // `-Dcpu=baseline` for SSE2-only (most portable).
    const default_query: std.Target.Query = if (builtin.target.cpu.arch == .x86_64) .{
        .cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 },
    } else .{};
    const target = b.standardTargetOptions(.{ .default_target = default_query });
    const optimize = b.standardOptimizeOption(.{});

    // -Dstrip=false keeps debug info even in ReleaseFast so profilers
    // (VTune, samply, etc.) can attribute samples to source lines.
    const strip = b.option(bool, "strip", "Strip debug symbols (default: optimize-mode default)");

    const bench = b.option(bool, "bench", "Include zstd/lz4 vendor libs for -bc comparison benchmark (default: false)") orelse false;

    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    const gpu = b.option(bool, "gpu", "Enable GPU (CUDA) decompression via Driver API (default: false)") orelse false;

    const bench_option = b.addOptions();
    bench_option.addOption(bool, "enable_bench", bench);
    bench_option.addOption(bool, "gpu", gpu);
    root_module.addOptions("build_options", bench_option);

    const exe = b.addExecutable(.{
        .name = "streamlz",
        .root_module = root_module,
    });
    if (bench) addVendorLibs(b, exe);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the streamlz CLI");
    run_step.dependOn(&run_cmd.step);

    // Both `zig build test` and `zig build ptest` use the parallel test runner.
    const parallel_tests = b.addTest(.{
        .root_module = root_module,
        .test_runner = .{ .path = b.path("src/test_runner_parallel.zig"), .mode = .simple },
    });
    const run_parallel_tests = b.addRunArtifact(parallel_tests);
    const test_step = b.step("test", "Run unit tests (parallel)");
    test_step.dependOn(&run_parallel_tests.step);
    const ptest_step = b.step("ptest", "Run unit tests (parallel, alias for test)");
    ptest_step.dependOn(&run_parallel_tests.step);

    // ---- ReleaseSafe build step ----
    // Enables runtime safety checks (bounds, overflow) at moderate
    // performance cost. Useful for CI and testing against untrusted data.
    const safe_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .strip = strip,
        .link_libc = true,
    });
    safe_module.addOptions("build_options", bench_option);
    const safe_exe = b.addExecutable(.{
        .name = "streamlz-safe",
        .root_module = safe_module,
    });
    addVendorLibs(b, safe_exe);
    const safe_install = b.addInstallArtifact(safe_exe, .{});
    const safe_step = b.step("safe", "Build with ReleaseSafe (bounds + overflow checks)");
    safe_step.dependOn(&safe_install.step);

    // ---- Fuzz harness for the decompressor ----
    // Reads stdin as compressed input, calls decompressFramed, swallows
    // decode errors. Panics only on memory safety violations (which
    // ReleaseSafe catches). Feed with: afl-fuzz, honggfuzz, or manual
    // corpus via  `zig build fuzz && echo ... | ./zig-out/bin/fuzz-decompress`
    const streamlz_module = b.createModule(.{
        .root_source_file = b.path("src/streamlz.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .strip = strip,
        .link_libc = true,
    });
    const fuzz_module = b.createModule(.{
        .root_source_file = b.path("scripts/fuzz_decompress.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .strip = strip,
        .link_libc = true,
        .imports = &.{
            .{ .name = "streamlz", .module = streamlz_module },
        },
    });
    const fuzz_exe = b.addExecutable(.{
        .name = "fuzz-decompress",
        .root_module = fuzz_module,
    });
    const fuzz_install = b.addInstallArtifact(fuzz_exe, .{});
    const fuzz_step = b.step("fuzz", "Build fuzz_decompress harness (ReleaseSafe)");
    fuzz_step.dependOn(&fuzz_install.step);

    // ---- C API static library ----
    const lib_module = b.createModule(.{
        .root_source_file = b.path("src/capi.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "streamlz",
        .root_module = lib_module,
    });
    const lib_install = b.addInstallArtifact(lib, .{});
    const hdr_install = b.addInstallHeaderFile(b.path("include/streamlz.h"), "streamlz.h");
    const lib_step = b.step("lib", "Build C API static library (libstreamlz.a)");
    lib_step.dependOn(&lib_install.step);
    lib_step.dependOn(&hdr_install.step);
}

fn addVendorLibs(b: *std.Build, exe: *std.Build.Step.Compile) void {
    const target = exe.root_module.resolved_target.?;
    const optimize = exe.root_module.optimize.?;

    const c_flags = &.{ "-DXXH_NAMESPACE=ZSTD_", "-DZSTD_DISABLE_ASM", "-DZSTD_MULTITHREAD" };

    // ---- zstd (v1.5.7) — precompiled static library ----
    const zstd_mod = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
    zstd_mod.addIncludePath(b.path("vendor/zstd"));
    zstd_mod.addIncludePath(b.path("vendor/zstd/common"));
    zstd_mod.addCSourceFiles(.{ .files = &.{
        "vendor/zstd/common/debug.c",
        "vendor/zstd/common/entropy_common.c",
        "vendor/zstd/common/error_private.c",
        "vendor/zstd/common/fse_decompress.c",
        "vendor/zstd/common/pool.c",
        "vendor/zstd/common/threading.c",
        "vendor/zstd/common/xxhash.c",
        "vendor/zstd/common/zstd_common.c",
        "vendor/zstd/compress/fse_compress.c",
        "vendor/zstd/compress/hist.c",
        "vendor/zstd/compress/huf_compress.c",
        "vendor/zstd/compress/zstd_compress.c",
        "vendor/zstd/compress/zstd_compress_literals.c",
        "vendor/zstd/compress/zstd_compress_sequences.c",
        "vendor/zstd/compress/zstd_compress_superblock.c",
        "vendor/zstd/compress/zstd_double_fast.c",
        "vendor/zstd/compress/zstd_fast.c",
        "vendor/zstd/compress/zstd_lazy.c",
        "vendor/zstd/compress/zstd_ldm.c",
        "vendor/zstd/compress/zstd_opt.c",
        "vendor/zstd/compress/zstd_preSplit.c",
        "vendor/zstd/compress/zstdmt_compress.c",
        "vendor/zstd/decompress/huf_decompress.c",
        "vendor/zstd/decompress/zstd_ddict.c",
        "vendor/zstd/decompress/zstd_decompress.c",
        "vendor/zstd/decompress/zstd_decompress_block.c",
    }, .flags = c_flags });
    const zstd_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "zstd",
        .root_module = zstd_mod,
    });

    // ---- LZ4 (v1.10.0) — precompiled static library ----
    const lz4_mod = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
    lz4_mod.addIncludePath(b.path("vendor/lz4"));
    lz4_mod.addCSourceFiles(.{ .files = &.{
        "vendor/lz4/lz4.c",
        "vendor/lz4/lz4hc.c",
    }, .flags = &.{} });
    const lz4_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "lz4",
        .root_module = lz4_mod,
    });

    // Link the precompiled libs + expose include paths to the exe
    exe.root_module.addIncludePath(b.path("vendor/zstd"));
    exe.root_module.addIncludePath(b.path("vendor/zstd/common"));
    exe.root_module.addIncludePath(b.path("vendor/lz4"));
    exe.root_module.linkLibrary(zstd_lib);
    exe.root_module.linkLibrary(lz4_lib);
}
