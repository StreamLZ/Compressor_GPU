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

    // ── Vulkan port (M1): hello_vulkan smoke test ───────────────────────
    // Sibling tree at src_vulkan/ — additive only; never touch existing
    // CUDA targets above. Builds an executable that loads vulkan-1.dll,
    // creates an instance + logical device, prints one line of identifying
    // info, and exits. Run with `zig build hello-vulkan`.
    const hello_vk_module = b.createModule(.{
        .root_source_file = b.path("src_vulkan/hello_vulkan_test.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    const hello_vk = b.addExecutable(.{ .name = "hello_vulkan", .root_module = hello_vk_module });
    b.installArtifact(hello_vk);
    const run_hello_vk = b.addRunArtifact(hello_vk);
    run_hello_vk.step.dependOn(b.getInstallStep());
    b.step("hello-vulkan", "Run the M1 Vulkan loader smoke test").dependOn(&run_hello_vk.step);

    // ── Vulkan port (M2): vendor + feature probe → tier classifier ──────
    // Enumerates every physical device the loader reports and prints a
    // one-line summary per device (vendor, api, subgroup, tier verdict,
    // feature bits). Reused later by the M-late init path to pick the
    // device + tier at slzCreate_vk time.
    const probe_module = b.createModule(.{
        .root_source_file = b.path("src_vulkan/probe_main.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    const probe_exe = b.addExecutable(.{ .name = "vk_probe", .root_module = probe_module });
    b.installArtifact(probe_exe);
    const run_probe = b.addRunArtifact(probe_exe);
    run_probe.step.dependOn(b.getInstallStep());
    b.step("vk-probe", "Run the M2 Vulkan feature + tier probe").dependOn(&run_probe.step);

    // ── Vulkan port (M6): SPIR-V shader compilation ─────────────────────
    // 17 .comp shells × 3 variants (tier1, tier1_nv, tier2) = 51 .spv blobs
    // emitted under zig-out/shaders/ AND aggregated into a WriteFiles
    // directory exposed as `vk_shaders.embed_dir`. Driven by glslc from
    // $VULKAN_SDK or PATH. Modules that import `src_vulkan/spv_blobs.zig`
    // must add `vk_shaders.embed_dir` via `addEmbedPath` and depend on
    // `vk_shaders.embed_dir_step` so the .spv files exist at zig compile
    // time and @embedFile() resolves the bare filenames.
    const vk_shaders = addVulkanShaderSteps(b);

    // ── Vulkan port (M5): pure-Zig unit tests for src_vulkan/ ───────────
    // Holds the algorithm-only modules that need no Vulkan device to
    // exercise — e.g. hash_bin_pack.zig (2 GiB VkBuffer split planner).
    // Decoupled from the main `test` step so a CUDA-only dev box never
    // accidentally pulls Vulkan modules into its test run.
    const vk_test_module = b.createModule(.{
        .root_source_file = b.path("src_vulkan/hash_bin_pack.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    const vk_test_runner = b.addTest(.{ .root_module = vk_test_module });
    const run_vk_tests = b.addRunArtifact(vk_test_runner);
    b.step("vk-test", "Run pure-Zig unit tests under src_vulkan/").dependOn(&run_vk_tests.step);

    // ── Vulkan port (M8b): descriptors cache bookkeeping tests ──────────
    // Pure-Zig tests on the LRU cache structure and PipelineKey.eql.
    // Decoupled into its own step (vs folding into vk-test) because the
    // module pulls in vk_api.zig + driver.zig transitively, which the
    // hash_bin_pack module deliberately avoids. Running both as separate
    // test executables keeps the dependency graphs visible.
    const vk_descriptors_module = b.createModule(.{
        .root_source_file = b.path("src_vulkan/descriptors.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    const vk_descriptors_test_runner = b.addTest(.{ .root_module = vk_descriptors_module });
    const run_vk_descriptors_tests = b.addRunArtifact(vk_descriptors_test_runner);
    b.step("vk-descriptors-test", "Run pure-Zig descriptors cache unit tests").dependOn(&run_vk_descriptors_tests.step);

    // ── Vulkan port (M4): streamlz_vk static + shared C ABI ─────────────
    // Sibling of the CUDA `streamlz_gpu.dll`. Exports 16 `_vk`-suffixed
    // symbols from src_vulkan/streamlz_gpu_vk.zig. The .dll is what game
    // engines / ML pipelines link against; the .lib is the import library
    // on Windows + a convenience static archive on POSIX.
    //
    // Root at the repo top (`streamlz_vk_lib_root.zig`) so the wire-format
    // module's `../src/format/...` imports resolve inside the package.
    const vk_lib_module = b.createModule(.{
        .root_source_file = b.path("streamlz_vk_lib_root.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    // SPV blobs are baked in at compile time via the `spv_blobs` module
    // (rooted in the WriteFiles dir alongside the .spv files). Modules
    // that import src_vulkan/spv_blobs.zig (transitively via slz1_codec.zig)
    // must call addSpvBlobsImport so the @import("spv_blobs") name
    // resolves to the colocated-in-embed_dir copy.
    addSpvBlobsImport(vk_lib_module, vk_shaders);

    const streamlz_vk_static = b.addLibrary(.{
        .linkage = .static,
        .name = "streamlz_vk",
        .root_module = vk_lib_module,
    });
    streamlz_vk_static.step.dependOn(vk_shaders.embed_dir_step);
    b.installArtifact(streamlz_vk_static);

    const streamlz_vk_shared = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "streamlz_vk",
        .root_module = vk_lib_module,
    });
    streamlz_vk_shared.step.dependOn(vk_shaders.embed_dir_step);
    const streamlz_vk_install = b.addInstallArtifact(streamlz_vk_shared, .{});
    const streamlz_vk_hdr = b.addInstallHeaderFile(
        b.path("include/streamlz_gpu_vk.h"),
        "streamlz_gpu_vk.h",
    );
    const vk_lib_step = b.step("vk-lib", "Build Vulkan C ABI shared library (streamlz_vk.dll)");
    vk_lib_step.dependOn(&streamlz_vk_install.step);
    vk_lib_step.dependOn(&streamlz_vk_hdr.step);

    // ── Vulkan port (M4): smoke test for the C ABI ──────────────────────
    // Calls slzCreate_vk + slzDestroy_vk directly via Zig import (no DLL
    // load); prints "vk smoke OK" on success. Wired as `zig build vk-smoke`.
    const vk_smoke_module = b.createModule(.{
        .root_source_file = b.path("streamlz_vk_smoke_test_root.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    addSpvBlobsImport(vk_smoke_module, vk_shaders);
    const vk_smoke_exe = b.addExecutable(.{ .name = "vk_smoke", .root_module = vk_smoke_module });
    vk_smoke_exe.step.dependOn(vk_shaders.embed_dir_step);
    b.installArtifact(vk_smoke_exe);
    const run_vk_smoke = b.addRunArtifact(vk_smoke_exe);
    run_vk_smoke.step.dependOn(b.getInstallStep());
    b.step("vk-smoke", "Run the M4 Vulkan C ABI smoke test").dependOn(&run_vk_smoke.step);

    // ── Vulkan port (M8c): end-to-end dispatch test ─────────────────────
    // The substrate-completion gate: bring up the loader + instance + device,
    // probe the tier, load the lz_encode .spv shell, create a 4 KiB host-
    // visible + device-local buffer, build the pipeline + descriptor set
    // via descriptors.zig, dispatch via dispatch.zig, read back data[0] and
    // assert == 0xDEADBEEF. Wired as `zig build vk-dispatch-test`.
    //
    // The .spv shells are loaded at RUNTIME from `zig-out/shaders/` (the
    // dir `vk-shaders` deposits them into). This sidesteps the M4-pending
    // build-system @embedFile wiring for spv_blobs.zig — the dispatch test
    // is an interactive tool, not a library, so a runtime open() is fine.
    // The hard dependency on `vk-shaders` below guarantees the files exist
    // before the test runs.
    const vk_dispatch_test_module = b.createModule(.{
        .root_source_file = b.path("src_vulkan/dispatch_test.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    const vk_dispatch_test_exe = b.addExecutable(.{
        .name = "vk_dispatch_test",
        .root_module = vk_dispatch_test_module,
    });
    b.installArtifact(vk_dispatch_test_exe);
    const run_vk_dispatch_test = b.addRunArtifact(vk_dispatch_test_exe);
    run_vk_dispatch_test.step.dependOn(b.getInstallStep());
    // Hard dependency: the vk-shaders top-level step deposits every
    // <kernel>.<variant>.spv under zig-out/shaders/ — the test opens the
    // lz_encode.{tier}.spv at runtime, so the files must exist.
    const vk_shaders_top = b.top_level_steps.get("vk-shaders").?;
    run_vk_dispatch_test.step.dependOn(&vk_shaders_top.step);
    b.step("vk-dispatch-test", "Run the M8c end-to-end dispatch test").dependOn(&run_vk_dispatch_test.step);

    // ── Vulkan port (M5): __match_any_sync emulation microbenchmark ─────
    // Stands up the loader → probe → pipeline → 100-dispatch timing loop
    // around the match_any_bench.comp shader and prints ns/call against
    // the foundation-R1 CUDA reference (~1 ns). Same runtime-load-from-
    // zig-out/shaders/ pattern as vk-dispatch-test, so the hard dependency
    // on vk-shaders ensures the .spv files exist before the bench runs.
    const vk_match_any_module = b.createModule(.{
        .root_source_file = b.path("src_vulkan/match_any_bench.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    const vk_match_any_exe = b.addExecutable(.{
        .name = "vk_match_any_bench",
        .root_module = vk_match_any_module,
    });
    b.installArtifact(vk_match_any_exe);
    const run_vk_match_any = b.addRunArtifact(vk_match_any_exe);
    run_vk_match_any.step.dependOn(b.getInstallStep());
    run_vk_match_any.step.dependOn(&vk_shaders_top.step);
    b.step("vk-match-any-bench", "Run the M5 __match_any_sync emulation microbenchmark").dependOn(&run_vk_match_any.step);

    // ── Vulkan port (L1 codec, phase 4): round-trip test ────────────────
    // The phase-1 L1 codec (greedy LZ77) round-trip gate: Vulkan-encode
    // bytes → Vulkan-decode the streams → byte-equal to input. Loads
    // lz_encode.<tier>.spv and lz_decode.<tier>.spv from `zig-out/shaders/`
    // at runtime (same shape as vk-dispatch-test and vk-match-any-bench),
    // so the hard dependency on `vk-shaders` ensures the .spv files
    // exist before the test runs. Wired as `zig build vk-l1-test`.
    const vk_l1_test_module = b.createModule(.{
        .root_source_file = b.path("src_vulkan/l1_codec_test.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    const vk_l1_test_exe = b.addExecutable(.{
        .name = "vk_l1_test",
        .root_module = vk_l1_test_module,
    });
    b.installArtifact(vk_l1_test_exe);
    const run_vk_l1_test = b.addRunArtifact(vk_l1_test_exe);
    run_vk_l1_test.step.dependOn(b.getInstallStep());
    run_vk_l1_test.step.dependOn(&vk_shaders_top.step);
    b.step("vk-l1-test", "Run the phase-4 L1 codec round-trip test").dependOn(&run_vk_l1_test.step);

    // ── Vulkan port (CLI): streamlz_vk.exe — sibling of streamlz.exe ────
    // L1-only, level=1 CLI for end users: `streamlz_vk -c f -o f.slz`,
    // `streamlz_vk -d f.slz -o f.out`. Linked statically against the L1
    // codec + wire-format modules. SPV blobs are loaded at runtime from
    // `zig-out/shaders/` (same pattern as the test exes), so the binary
    // must be invoked with cwd set to a directory containing
    // `zig-out/shaders/*.spv` — typically the repo root during dev.
    const cli_vk_module = b.createModule(.{
        .root_source_file = b.path("cli_vk_root.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    // SPV blobs are baked into streamlz_vk.exe via the embedded spv_blobs
    // module (see addSpvBlobsImport) so the binary runs from any working
    // directory with no zig-out/shaders dep at runtime. The
    // embed_dir_step dependency makes Zig wait for glslc to finish before
    // compiling spv_blobs.zig.
    addSpvBlobsImport(cli_vk_module, vk_shaders);
    const cli_vk_exe = b.addExecutable(.{ .name = "streamlz_vk", .root_module = cli_vk_module });
    cli_vk_exe.step.dependOn(vk_shaders.embed_dir_step);
    const cli_vk_install = b.addInstallArtifact(cli_vk_exe, .{});
    b.step("streamlz_vk", "Build the Vulkan-backend CLI (streamlz_vk.exe)").dependOn(&cli_vk_install.step);

    // ── Vulkan port (CLI test): self-roundtrip + cross-backend ───────────
    // Spawns streamlz_vk.exe and streamlz.exe (the CUDA build) via
    // std.process.Child to verify four directions of round-trip on
    // assets/web.txt: VK→VK, CUDA→CUDA(skipped — already covered by
    // existing CUDA tests), VK→CUDA, CUDA→VK. Reports PASS/FAIL lines.
    const cli_vk_test_module = b.createModule(.{
        .root_source_file = b.path("cli_vk_test_root.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    const cli_vk_test_exe = b.addExecutable(.{
        .name = "vk_cli_test",
        .root_module = cli_vk_test_module,
    });
    b.installArtifact(cli_vk_test_exe);
    const run_cli_vk_test = b.addRunArtifact(cli_vk_test_exe);
    run_cli_vk_test.step.dependOn(b.getInstallStep());
    run_cli_vk_test.step.dependOn(&cli_vk_install.step);
    run_cli_vk_test.step.dependOn(&vk_shaders_top.step);
    b.step("vk-cli-test", "Run the streamlz_vk.exe CLI self + cross-backend round-trip test").dependOn(&run_cli_vk_test.step);

    // ── Vulkan port (C ABI roundtrip): end-to-end through the public ABI ─
    // Calls slzCreate_vk + slzCompressHost_vk + slzDecompressHost_vk +
    // slzDestroy_vk on a 1 MiB prefix of `assets/web.txt` and asserts a
    // byte-equal round trip. Same SPV-blobs-from-zig-out/shaders runtime
    // dependency as vk-l1-test (the SPV files are loaded by slz1_codec
    // through std.Io). Wired as `zig build vk-abi-test`.
    const vk_abi_test_module = b.createModule(.{
        .root_source_file = b.path("c_abi_test_root.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    addSpvBlobsImport(vk_abi_test_module, vk_shaders);
    const vk_abi_test_exe = b.addExecutable(.{
        .name = "vk_abi_test",
        .root_module = vk_abi_test_module,
    });
    vk_abi_test_exe.step.dependOn(vk_shaders.embed_dir_step);
    b.installArtifact(vk_abi_test_exe);
    const run_vk_abi_test = b.addRunArtifact(vk_abi_test_exe);
    run_vk_abi_test.step.dependOn(b.getInstallStep());
    run_vk_abi_test.step.dependOn(&vk_shaders_top.step);
    b.step("vk-abi-test", "Run the C ABI end-to-end round-trip test").dependOn(&run_vk_abi_test.step);

    // ── Vulkan port (L1 wire-format): SLZ1 wrap/unwrap conformance ─────
    // Wraps the L1 codec's raw streams into a real .slz file (CPU-side)
    // and round-trips both directions against the CUDA encoder/decoder
    // shipping in zig-out/bin/streamlz.exe.  Shells out to streamlz.exe
    // for the CUDA half; the Vulkan half rides the same SPV blobs the
    // l1-test loads.  Wired as `zig build vk-wire-format-test`.
    // Root at the repo top so `wire_format.zig`'s `../src/format/...`
    // imports stay inside the module's package boundary.  Same pattern
    // as `tests_root.zig` above.
    const vk_wire_test_module = b.createModule(.{
        .root_source_file = b.path("wire_format_test_root.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    addSpvBlobsImport(vk_wire_test_module, vk_shaders);
    const vk_wire_test_exe = b.addExecutable(.{
        .name = "vk_wire_format_test",
        .root_module = vk_wire_test_module,
    });
    vk_wire_test_exe.step.dependOn(vk_shaders.embed_dir_step);
    b.installArtifact(vk_wire_test_exe);
    const run_vk_wire_test = b.addRunArtifact(vk_wire_test_exe);
    run_vk_wire_test.step.dependOn(b.getInstallStep());
    run_vk_wire_test.step.dependOn(&vk_shaders_top.step);
    b.step("vk-wire-format-test", "Run the L1 SLZ1 wire-format wrap/unwrap test").dependOn(&run_vk_wire_test.step);

    // ── Vulkan port: perf measurement bench (l1_perf_bench) ───────────
    // Standalone runner used to capture before/after wall-clock numbers
    // for the decoder fast-batch (piece 2) and encoder warp-parallel match
    // extension (piece 3) restorations. Reads assets/web.txt, encode +
    // decode through the production Vulkan modules, prints ns/byte lines.
    const vk_perf_bench_module = b.createModule(.{
        .root_source_file = b.path("src_vulkan/l1_perf_bench.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    addSpvBlobsImport(vk_perf_bench_module, vk_shaders);
    const vk_perf_bench_exe = b.addExecutable(.{
        .name = "vk_l1_perf_bench",
        .root_module = vk_perf_bench_module,
    });
    vk_perf_bench_exe.step.dependOn(vk_shaders.embed_dir_step);
    b.installArtifact(vk_perf_bench_exe);
    const run_vk_perf_bench = b.addRunArtifact(vk_perf_bench_exe);
    run_vk_perf_bench.step.dependOn(b.getInstallStep());
    run_vk_perf_bench.step.dependOn(&vk_shaders_top.step);
    b.step("vk-perf-bench", "Run the Vulkan L1 encode/decode wall-clock perf bench").dependOn(&run_vk_perf_bench.step);

    // ── Vulkan port (M9): cross-backend conformance harness ────────────
    // 4-direction (CUDA↔CUDA, CUDA→VK, VK→CUDA, VK↔VK) × 5 levels ×
    // 3 corpora matrix. Only D1 runs real codec today; the remaining
    // three directions return `.skipped_unimplemented` until the matching
    // Vulkan kernels land in Waves 1 + 2.
    //
    // The harness reuses the production encoder/decoder modules via
    // explicit `addImport` (instead of a relative `@import` chain) so
    // tests/cross_backend_tests.zig can live outside src/ without
    // duplicating the module roots.
    // Root the test module at `tests_root.zig` (repo root) so its package
    // boundary covers both `src/` and `tests/` — the conformance harness
    // file under `tests/` then `@import`s the production encoder /
    // decoder via `../src/...` relative paths without tripping Zig 0.16's
    // "file outside module path" check. tests_root.zig is a 4-line
    // forwarder; the real test code stays in tests/cross_backend_tests.zig.
    const conformance_module = b.createModule(.{
        .root_source_file = b.path("tests_root.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });

    const conformance_test = b.addTest(.{
        .root_module = conformance_module,
        .test_runner = .{ .path = b.path("src/test_runner_parallel.zig"), .mode = .simple },
    });
    conformance_test.step.dependOn(ptx_freshness);
    const run_conformance = b.addRunArtifact(conformance_test);
    // Surface stdout/stderr live so the "conformance: X pass / Y fail / Z
    // skipped" summary is visible during the run, not buried in a log.
    run_conformance.has_side_effects = true;
    b.step(
        "test-conformance",
        "Run the M9 cross-backend conformance matrix (CUDA↔CUDA real, VK cells stubbed)",
    ).dependOn(&run_conformance.step);
    // Also fold into the main `zig build test` step so the dashboard runs
    // on every dev iteration — additive (does not replace the parallel
    // `run_tests` step above).
    b.top_level_steps.get("test").?.step.dependOn(&run_conformance.step);
    b.top_level_steps.get("ptest").?.step.dependOn(&run_conformance.step);
}


pub const VulkanShaders = struct {
    /// Top-level step: `zig build vk-shaders` installs all 51 .spv blobs
    /// under zig-out/shaders/.
    step: *std.Build.Step,
    /// Directory LazyPath containing every <kernel>.<variant>.spv as a
    /// flat list, plus a copy of `spv_blobs.zig`. Used as the root
    /// directory for the spv_blobs Zig module — @embedFile() inside
    /// spv_blobs.zig resolves the bare .spv names against this dir.
    embed_dir: std.Build.LazyPath,
    /// WriteFile step that materializes the embed directory. Consumers
    /// MUST add this as an explicit `step.dependOn` so the .spv files
    /// exist before the compile step that reads them runs.
    embed_dir_step: *std.Build.Step,
    /// LazyPath to the copy of spv_blobs.zig that lives inside `embed_dir`.
    /// Use this as `root_source_file` when calling `addAnonymousImport` so
    /// the module's package root encompasses both spv_blobs.zig and the
    /// sibling .spv files.
    spv_blobs_zig: std.Build.LazyPath,
};

/// Wire the embedded `spv_blobs` module into `m` so calls to
/// `@import("spv_blobs")` resolve to the colocated-in-embed_dir copy of
/// `src_vulkan/spv_blobs.zig`. Also wires the build-time dependency so
/// the .spv files are produced before this module's compile step runs.
fn addSpvBlobsImport(m: *std.Build.Module, vk_shaders: VulkanShaders) void {
    const b = m.owner;
    const spv_mod = b.createModule(.{
        .root_source_file = vk_shaders.spv_blobs_zig,
    });
    m.addImport("spv_blobs", spv_mod);
}

// ────────────────────────────────────────────────────────────────────────
//  Vulkan SPIR-V build rules (M6)
// ────────────────────────────────────────────────────────────────────────

const vk_kernels = [_][]const u8{
    // encode (6)
    "lz_encode",
    "huff_build_tables",
    "huff_encode_4stream",
    "assemble_measure",
    "assemble_write",
    "frame_assemble",
    // decode (11)
    "walk_frame",
    "prefix_sum_chunks",
    "scan_parse",
    "compact_huff_descs",
    "compact_raw_descs",
    "gather_raw_off16",
    "merge_huff_descs",
    "huff_build_lut",
    "huff_decode_4stream",
    "lz_decode",
    "lz_decode_raw",
    // M5 microbench (not a production kernel, but rides the same glslc pipeline).
    "match_any_bench",
};

const VkVariant = struct {
    name: []const u8, // "tier1" | "tier1_nv" | "tier2"
    target_env: []const u8, // "vulkan1.3" | "vulkan1.2"
    defines: []const []const u8, // extra -D macros
};

const vk_variants = [_]VkVariant{
    .{ .name = "tier1", .target_env = "vulkan1.3", .defines = &.{"-DTIER1"} },
    .{ .name = "tier1_nv", .target_env = "vulkan1.3", .defines = &.{ "-DTIER1", "-DTIER1_NV" } },
    .{ .name = "tier2", .target_env = "vulkan1.2", .defines = &.{"-DTIER2"} },
};

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

fn addVulkanShaderSteps(b: *std.Build) VulkanShaders {
    const glslc = resolveGlslc(b);
    const vk_shaders_step = b.step("vk-shaders", "Compile Vulkan compute shaders to SPIR-V (17 × 3 = 51 blobs)");
    // Single WriteFile aggregates every .spv into one generated directory.
    // The same dir also receives a copy of src_vulkan/spv_blobs.zig so its
    // bare-filename @embedFile() calls resolve against sibling .spv files;
    // consumers import the colocated spv_blobs.zig as a Module rooted in
    // this generated dir (see vk_spv_blobs_module helper below).
    const spv_wf = b.addWriteFiles();

    for (vk_kernels) |kernel| {
        for (vk_variants) |variant| {
            const src_path = b.fmt("src_vulkan/shaders/{s}.comp", .{kernel});
            const out_name = b.fmt("{s}.{s}.spv", .{ kernel, variant.name });

            const cmd = b.addSystemCommand(&.{glslc});
            cmd.addArg("-fshader-stage=compute");
            cmd.addArg("-O");
            cmd.addArg(b.fmt("--target-env={s}", .{variant.target_env}));
            for (variant.defines) |def| cmd.addArg(def);
            cmd.addArg("-o");
            const spv_lp = cmd.addOutputFileArg(out_name);
            cmd.addFileArg(b.path(src_path));

            // Gather into the embed dir under a flat filename.
            _ = spv_wf.addCopyFile(spv_lp, out_name);

            // Also install for inspection — zig-out/shaders/<name>.spv is the
            // documented artifact location.
            const install = b.addInstallFileWithDir(spv_lp, .prefix, b.fmt("shaders/{s}", .{out_name}));
            vk_shaders_step.dependOn(&install.step);
        }
    }
    // Copy spv_blobs.zig alongside the .spv files so its @embedFile("...spv")
    // calls find the colocated blobs at compile time.
    const blobs_in_dir = spv_wf.addCopyFile(b.path("src_vulkan/spv_blobs.zig"), "spv_blobs.zig");
    return .{
        .step = vk_shaders_step,
        .embed_dir = spv_wf.getDirectory(),
        .embed_dir_step = &spv_wf.step,
        .spv_blobs_zig = blobs_in_dir,
    };
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
