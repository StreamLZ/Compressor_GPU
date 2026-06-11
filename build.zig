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
    // Default to ReleaseFast so bare `zig build` matches the header comment
    // above. The stock `standardOptimizeOption` defaults to `.Debug` even
    // when `preferred_optimize_mode` is set (it only consults `--release`
    // on the CLI), which means every bench/release artifact that forgot
    // `-Doptimize=ReleaseFast` silently shipped runtime safety checks.
    // We still honour an explicit `-Doptimize=...` so existing scripts
    // (`tools/bench_d2d.bat`, `tools/build_gpu.bat`, README examples,
    // `.claude/settings.json` permissions) keep working unchanged.
    const optimize: std.builtin.OptimizeMode = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Prioritize performance, safety, or binary size (default: ReleaseFast)",
    ) orelse .ReleaseFast;

    // `-Dstrip=false` keeps debug info even in ReleaseFast so profilers
    // (VTune, samply, etc.) can attribute samples to source lines.
    const strip = b.option(bool, "strip", "Strip debug symbols (default: optimize-mode default)");

    // The CLI links nothing optional at this point. The build options
    // object stays around in case future flags need it.
    const build_options = b.addOptions();

    const ptx_freshness = newPtxFreshnessStep(b);

    // `zig build ptx` — recompile exactly the stale .cu TUs (and all of
    // them after a shared-.cuh edit), replacing the manual
    // vcvarsall + nvcc + touch-the-sibling-PTXs workflow.
    b.step("ptx", "Recompile stale GPU kernels (.cu -> .ptx) via nvcc; no-op when fresh").dependOn(newPtxRebuildStep(b));

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

    // v4 #13: frame-mutation fuzz harness (differential CUDA-vs-VK).
    // Host-only tool - drives both backend CLIs as subprocesses.
    const fuzz_module = b.createModule(.{
        .root_source_file = b.path("tools/fuzz_frames.zig"),
        .target = target,
        .optimize = optimize,
    });
    const fuzz_exe = b.addExecutable(.{ .name = "fuzz_frames", .root_module = fuzz_module });
    b.installArtifact(fuzz_exe);
    b.step("fuzz", "Build the v4 #13 frame-mutation fuzz harness").dependOn(&b.addInstallArtifact(fuzz_exe, .{}).step);

    // v4 #11 gate-2: replay SLZ_TANS_GATE2_DUMP records through the
    // resurrected host tANS-32 encoder (measurement-only, host-only).
    const gate2_module = b.createModule(.{
        .root_source_file = b.path("tools/tans_gate2/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const gate2_exe = b.addExecutable(.{ .name = "tans_gate2", .root_module = gate2_module });
    b.step("tans_gate2", "Build the v4 #11 gate-2 tANS-vs-Huffman measurement tool").dependOn(&b.addInstallArtifact(gate2_exe, .{}).step);

    // ── Unit tests ───────────────────────────────────────────────────────
    const test_runner = b.addTest(.{
        .root_module = cli_module,
        .test_runner = .{ .path = b.path("src/test_runner_parallel.zig"), .mode = .simple },
    });
    const run_tests = b.addRunArtifact(test_runner);
    // cli_smoke_tests.zig spawns zig-out/bin/streamlz.exe as a child
    // process — install the production binary first so the smoke tests
    // never run against a stale exe (mirrors ptest_vk's dependency on
    // srcvk_install).
    run_tests.step.dependOn(b.getInstallStep());
    run_tests.has_side_effects = true;
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

    // ── Vulkan backend (srcVK/): streamlz_vk.exe ──────────────────────
    // The production Vulkan CLI, L1-L5 feature-parity with the CUDA
    // streamlz.exe. (The earlier src_vulkan/ + src_vk/ port trees and
    // the vendored /third_party/ VMA they used were deleted 2026-06-10;
    // srcVK is self-contained — VMA lives under srcVK/vma/.)
    //
    // SPV blobs are compiled from every srcVK/**/*.comp by
    // addSrcVkShaderSteps below and embedded via @embedFile at compile
    // time.
    const srcvk_shaders = addSrcVkShaderSteps(b);
    const srcvk_exe_module = b.createModule(.{
        .root_source_file = b.path("srcVK/main.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
        .link_libcpp = true,
    });
    srcvk_exe_module.addCSourceFile(.{
        .file = b.path("srcVK/vma/vk_mem_alloc_impl.cpp"),
        .flags = &.{
            "-std=c++17",
            "-Wno-everything",
        },
    });
    // VMA's vk_mem_alloc.h lives at srcVK/vma/vk_mem_alloc.h.
    // The @cInclude("vma/vk_mem_alloc.h") in srcVK/vma.zig resolves
    // against the srcVK/ include path.
    srcvk_exe_module.addIncludePath(b.path("srcVK"));
    if (b.graph.environ_map.get("VULKAN_SDK")) |sdk| {
        const include_path = b.fmt("{s}{c}Include", .{ sdk, std.fs.path.sep });
        srcvk_exe_module.addIncludePath(.{ .cwd_relative = include_path });
    }
    // Wire spv_blobs module into srcvk_exe_module so module_loader.zig can
    // @import("spv_blobs") for the @embedFile()'d kernel binaries.
    addSrcVkSpvBlobsImport(b, srcvk_exe_module, srcvk_shaders);
    const srcvk_exe = b.addExecutable(.{
        .name = "streamlz_vk",
        .root_module = srcvk_exe_module,
    });
    srcvk_exe.step.dependOn(srcvk_shaders.step);
    srcvk_exe.step.dependOn(srcvk_shaders.embed_dir_step);
    const srcvk_install = b.addInstallArtifact(srcvk_exe, .{});
    b.step("streamlz_vk", "Build the foundation-wave Vulkan CLI (streamlz_vk.exe, srcVK/ tree)").dependOn(&srcvk_install.step);

    // ── Vulkan port (Phase 13): srcVK/ parallel unit test runner ────────
    // Mirrors the CUDA `zig build ptest` shape: re-uses srcvk_exe_module
    // so the test binary sees the same imports as the production CLI,
    // and runs every `test {}` block aggregated from srcVK/main.zig
    // (which now pulls in the 8 NEW test files under srcVK/tests/).
    const srcvk_test = b.addTest(.{
        .root_module = srcvk_exe_module,
        .test_runner = .{ .path = b.path("srcVK/test_runner_parallel.zig"), .mode = .simple },
    });
    srcvk_test.step.dependOn(srcvk_shaders.step);
    srcvk_test.step.dependOn(srcvk_shaders.embed_dir_step);
    const run_srcvk_tests = b.addRunArtifact(srcvk_test);
    // The CLI smoke / cross-backend tests shell out to
    // zig-out/bin/streamlz_vk.exe, so the test runner must wait for the
    // production binary install before spawning the child processes.
    run_srcvk_tests.step.dependOn(&srcvk_install.step);
    run_srcvk_tests.has_side_effects = true;
    b.step("ptest_vk", "Run the srcVK/ parallel unit test suite (8 new test files + srcVK module aggregator)").dependOn(&run_srcvk_tests.step);
    b.step("test_vk", "Alias for ptest_vk").dependOn(&run_srcvk_tests.step);

}



fn resolveGlslc(b: *std.Build) []const u8 {
    // Prefer $VULKAN_SDK\Bin\glslc.exe on Windows, $VULKAN_SDK/bin/glslc on
    // POSIX. Fall back to bare "glslc" so PATH lookup handles it.
    if (b.graph.environ_map.get("VULKAN_SDK")) |sdk| {
        const exe_name = if (builtin.target.os.tag == .windows) "glslc.exe" else "glslc";
        const subdir = if (builtin.target.os.tag == .windows) "Bin" else "bin";
        return b.fmt("{s}{c}{s}{c}{s}", .{ sdk, std.fs.path.sep, subdir, std.fs.path.sep, exe_name });
    }
    return "glslc";
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
                "Run `zig build ptx` (stale-only nvcc rebuild), then re-run this build.\n" ++
                "(tools\\build_gpu.bat does the full rebuild with cuobjdump res-usage.)",
            .{ newest_src_path.?, oldest_ptx_path.? },
        );
    }
}

// ────────────────────────────────────────────────────────────────────────
//  `zig build ptx` — stale-only nvcc rebuild
// ────────────────────────────────────────────────────────────────────────
//
// E-section QoL item (BACKPORTS.md): the manual workflow was
// vcvarsall → nvcc for the edited TU → touch the four sibling .ptx so
// the freshness gate stops tripping on a shared-.cuh edit. This step
// replaces all of it: a TU is rebuilt when its .ptx is older than its
// .cu or ANY .cuh under the kernel dirs (the same global conservatism
// the gate uses, so a header edit rebuilds every TU and the gate then
// passes with no touch dance). Fresh TUs are skipped, so the step is a
// fast no-op when nothing changed. PTX only — for cubin + res-usage
// printouts use tools/build_gpu.bat.
//
// The nvcc invocations run from a generated .bat (one vcvarsall call,
// `if errorlevel 1` after each nvcc) because passing a compound
// command line through cmd.exe /c as a single argv element gets
// re-quoted by the spawn layer and breaks on the spaces in the
// vcvarsall path.

const vcvarsall_path = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvarsall.bat";

const ptx_tus = [_]struct { cu: []const u8, ptx: []const u8 }{
    .{ .cu = "src/decode/lz_kernel.cu", .ptx = "src/decode/lz_kernel.ptx" },
    .{ .cu = "src/decode/huffman_kernel.cu", .ptx = "src/decode/huffman_kernel.ptx" },
    .{ .cu = "src/encode/lz_kernel.cu", .ptx = "src/encode/lz_kernel.ptx" },
    .{ .cu = "src/encode/huffman_kernel.cu", .ptx = "src/encode/huffman_kernel.ptx" },
    .{ .cu = "src/encode/assemble_kernel.cu", .ptx = "src/encode/assemble_kernel.ptx" },
};

fn newPtxRebuildStep(b: *std.Build) *std.Build.Step {
    const step = b.allocator.create(std.Build.Step) catch @panic("oom");
    step.* = std.Build.Step.init(.{
        .id = .custom,
        .name = "gpu-ptx-rebuild",
        .owner = b,
        .makeFn = ptxRebuild,
    });
    return step;
}

fn ptxRebuild(step: *std.Build.Step, opts: std.Build.Step.MakeOptions) anyerror!void {
    _ = opts;
    const b = step.owner;
    const alloc = b.allocator;
    const io = b.graph.io;

    // Newest header mtime across the kernel dirs. Any .cuh edit makes
    // every TU stale — same conservatism as the freshness gate, which
    // is what lets this step fully replace the manual touch dance.
    var newest_cuh_mtime: i96 = std.math.minInt(i96);
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
            if (!std.mem.endsWith(u8, entry.basename, ".cuh")) continue;
            const stat = entry.dir.statFile(io, entry.basename, .{}) catch continue;
            const mtime_ns: i96 = stat.mtime.toNanoseconds();
            if (mtime_ns > newest_cuh_mtime) newest_cuh_mtime = mtime_ns;
        }
    }

    var stale = std.ArrayListUnmanaged(usize).empty;
    defer stale.deinit(alloc);
    for (ptx_tus, 0..) |tu, i| {
        const cu_stat = b.build_root.handle.statFile(io, tu.cu, .{}) catch |err| {
            return step.fail("cannot stat {s}: {s}", .{ tu.cu, @errorName(err) });
        };
        const cu_mtime: i96 = cu_stat.mtime.toNanoseconds();
        const newest_dep = @max(cu_mtime, newest_cuh_mtime);
        const ptx_stat = b.build_root.handle.statFile(io, tu.ptx, .{}) catch {
            try stale.append(alloc, i); // missing .ptx → rebuild
            continue;
        };
        if (ptx_stat.mtime.toNanoseconds() < newest_dep) try stale.append(alloc, i);
    }

    if (stale.items.len == 0) {
        std.debug.print("ptx: all {d} kernels fresh, nothing to do\n", .{ptx_tus.len});
        return;
    }

    var bat = std.ArrayListUnmanaged(u8).empty;
    defer bat.deinit(alloc);
    try bat.appendSlice(alloc, "@echo off\r\ncall \"" ++ vcvarsall_path ++ "\" x64 >nul 2>&1\r\n");
    for (stale.items) |i| {
        const tu = ptx_tus[i];
        std.debug.print("ptx: rebuilding {s}\n", .{tu.ptx});
        const cu_abs = b.pathFromRoot(tu.cu);
        const ptx_abs = b.pathFromRoot(tu.ptx);
        try bat.appendSlice(alloc, "echo nvcc ");
        try bat.appendSlice(alloc, tu.ptx);
        try bat.appendSlice(alloc, "\r\nnvcc -ptx -o \"");
        try bat.appendSlice(alloc, ptx_abs);
        try bat.appendSlice(alloc, "\" \"");
        try bat.appendSlice(alloc, cu_abs);
        // -std=c++17: libcu++ (<cuda/pipeline>, v4 #15 mbarrier ring)
        // refuses to compile under MSVC's default C++14.
        try bat.appendSlice(alloc, "\" -arch=sm_89 -O3 -std=c++17\r\nif errorlevel 1 exit /b 1\r\n");
    }

    b.cache_root.handle.writeFile(io, .{
        .sub_path = "ptx_rebuild.bat",
        .data = bat.items,
    }) catch |err| {
        return step.fail("cannot write ptx_rebuild.bat: {s}", .{@errorName(err)});
    };
    const bat_path = b.cache_root.join(alloc, &.{"ptx_rebuild.bat"}) catch @panic("oom");

    const result = std.process.run(alloc, io, .{
        .argv = &.{ "cmd.exe", "/c", bat_path },
    }) catch |err| {
        return step.fail("spawning nvcc batch failed: {s}", .{@errorName(err)});
    };
    defer alloc.free(result.stdout);
    defer alloc.free(result.stderr);
    const code: u32 = switch (result.term) {
        .exited => |c| c,
        else => 1,
    };
    if (code != 0) {
        return step.fail(
            "nvcc rebuild failed (exit {d}).\nstdout:\n{s}\nstderr:\n{s}",
            .{ code, result.stdout, result.stderr },
        );
    }
    std.debug.print("ptx: rebuilt {d} kernel(s); run `zig build` to re-embed\n", .{stale.items.len});
}

// ────────────────────────────────────────────────────────────────────────
//  srcVK/ SPV blob compilation
// ────────────────────────────────────────────────────────────────────────
//
// Compiles every .comp under srcVK/{decode,encode}/ to a sibling .spv
// blob via glslc. Surfaces a single composite step the srcVK exe step
// depends on so the SPV files exist before linking. The skeleton
// shaders only contain `void main() {}` so the compilation is a
// formality at Step 2; the fleshout agent expands the bodies.

const srcvk_kernels = [_][]const u8{
    // decode (13)
    "decode/lz_decode_raw_kernel",
    "decode/lz_decode_raw_pipelined_kernel",
    "decode/lz_decode_kernel",
    "decode/prefix_sum_chunks_kernel",
    "decode/gather_raw_off16_kernel",
    "decode/walk_frame_kernel",
    "decode/compact_huff_descs_kernel",
    "decode/compact_raw_descs_kernel",
    "decode/compact_all_descs_kernel",
    "decode/merge_huff_descs_kernel",
    "decode/merge_huff_descs_par_kernel",
    "decode/scan_parse_kernel",
    "decode/huff_build_lut_kernel",
    "decode/huff_decode_4stream_kernel",
    // encode (6)
    "encode/lz_encode_kernel",
    "encode/huff_build_tables_kernel",
    "encode/huff_encode_4stream_kernel",
    "encode/assemble_measure_kernel",
    "encode/assemble_write_kernel",
    "encode/frame_assemble_kernel",
};

const SrcVkShaders = struct {
    step: *std.Build.Step,
    embed_dir: std.Build.LazyPath,
    embed_dir_step: *std.Build.Step,
    spv_blobs_zig: std.Build.LazyPath,
};

fn addSrcVkShaderSteps(b: *std.Build) SrcVkShaders {
    const glslc = resolveGlslc(b);
    const step = b.step("srcvk-shaders", "Compile every srcVK/**/*.comp to SPIR-V");

    // Aggregate every .spv blob into one WriteFiles directory so module_loader
    // can @embedFile() the bare filenames at compile time.
    const spv_wf = b.addWriteFiles();

    for (srcvk_kernels) |kernel| {
        const src_path = b.fmt("srcVK/{s}.comp", .{kernel});
        const basename = std.fs.path.basename(kernel);
        const out_name = b.fmt("{s}.spv", .{basename});
        const dep_name = b.fmt("{s}.d", .{basename});

        const cmd = b.addSystemCommand(&.{glslc});
        cmd.addArg("-fshader-stage=compute");
        cmd.addArg("-O");
        cmd.addArg("--target-env=vulkan1.3");
        cmd.addArg(b.fmt("-I{s}", .{b.pathFromRoot("srcVK/common")}));
        cmd.addArg(b.fmt("-I{s}", .{b.pathFromRoot("srcVK/decode")}));
        cmd.addArg(b.fmt("-I{s}", .{b.pathFromRoot("srcVK/encode")}));
        // Closes A-012: glslc -MD writes a Make-style depfile listing every
        // .glsl included by the .comp; addDepFileOutputArg has Zig parse it
        // after the run so editing any included header invalidates the cache
        // and forces a SPV rebuild. Retires the tools/build_vk.bat workaround.
        cmd.addArg("-MD");
        cmd.addArg("-MF");
        _ = cmd.addDepFileOutputArg(dep_name);
        cmd.addArg("-o");
        const spv_lp = cmd.addOutputFileArg(out_name);
        cmd.addFileArg(b.path(src_path));

        // Gather into the embed dir under a flat filename so spv_blobs.zig
        // resolves the bare names via @embedFile().
        _ = spv_wf.addCopyFile(spv_lp, out_name);

        const install = b.addInstallFileWithDir(spv_lp, .prefix, b.fmt("srcvk_shaders/{s}", .{out_name}));
        step.dependOn(&install.step);
    }

    // Copy spv_blobs.zig alongside the .spv files so its @embedFile("...spv")
    // calls find the colocated blobs at compile time.
    const blobs_in_dir = spv_wf.addCopyFile(b.path("srcVK/spv_blobs.zig"), "spv_blobs.zig");
    return .{
        .step = step,
        .embed_dir = spv_wf.getDirectory(),
        .embed_dir_step = &spv_wf.step,
        .spv_blobs_zig = blobs_in_dir,
    };
}

/// Wire the embedded `spv_blobs` module into `m` so calls to
/// `@import("spv_blobs")` resolve to the colocated-in-embed_dir copy of
/// `srcVK/spv_blobs.zig`, with the build-time dependency wired so the
/// .spv files exist before this module's compile step runs.
fn addSrcVkSpvBlobsImport(b: *std.Build, m: *std.Build.Module, vk_shaders: SrcVkShaders) void {
    const spv_mod = b.createModule(.{
        .root_source_file = vk_shaders.spv_blobs_zig,
    });
    m.addImport("spv_blobs", spv_mod);
}
