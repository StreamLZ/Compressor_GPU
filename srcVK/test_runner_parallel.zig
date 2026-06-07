//! 1:1 port of src/test_runner_parallel.zig.
//!
//! Parallel test runner for the VK port. Runs tests across multiple
//! threads for faster CI/dev feedback. Used by `zig build ptest_vk`.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const builtin = @import("builtin");

const TestFn = std.builtin.TestFn;

var global_ok: std.atomic.Value(usize) = .init(0);
var global_skip: std.atomic.Value(usize) = .init(0);
var global_fail: std.atomic.Value(usize) = .init(0);
var next_test: std.atomic.Value(usize) = .init(0);
var test_fns: []const TestFn = undefined;

// VK adaptation: serial markers. Tests whose name ends in:
//   [serial_first]  → run BEFORE the parallel batch starts, in the main
//                     thread one at a time. Used for assertions that
//                     depend on module-loader/global state being still
//                     in its default-null condition.
//   [serial]        → run AFTER the parallel batch joins, in the main
//                     thread one at a time. Used for tests that spawn
//                     external streamlz_vk.exe / streamlz.exe processes
//                     (cross-process Vulkan contention) and for any
//                     test that absolutely must not be interleaved with
//                     other in-process GPU users.
//
// Any test without one of these suffixes runs in the parallel pool.
const SERIAL_FIRST_SUFFIX = "[serial_first]";
const SERIAL_SUFFIX = "[serial]";

fn endsWith(haystack: []const u8, needle: []const u8) bool {
    if (haystack.len < needle.len) return false;
    return std.mem.eql(u8, haystack[haystack.len - needle.len ..], needle);
}

fn runOne(test_fn: TestFn) void {
    const result = test_fn.func();
    if (result) |_| {
        _ = global_ok.fetchAdd(1, .monotonic);
    } else |err| {
        if (err == error.SkipZigTest) {
            _ = global_skip.fetchAdd(1, .monotonic);
        } else {
            _ = global_fail.fetchAdd(1, .monotonic);
            std.debug.print("FAIL: {s} ({s})\n", .{ test_fn.name, @errorName(err) });
        }
    }
}

fn workerFn() void {
    while (true) {
        const i = next_test.fetchAdd(1, .monotonic);
        if (i >= test_fns.len) return;
        runOne(test_fns[i]);
    }
}

/// CUDA reference: src/test_runner_parallel.zig:37-end. Parallel test
/// runner entry point — collects builtin.test_functions, dispatches to
/// worker threads, reports the aggregated result.
pub fn main() void {
    // VK adaptation: partition tests by serial marker. The parallel
    // pool is the bulk; serial_first runs before; serial runs after.
    var first_buf: [256]TestFn = undefined;
    var parallel_buf: [1024]TestFn = undefined;
    var last_buf: [256]TestFn = undefined;
    var n_first: usize = 0;
    var n_parallel: usize = 0;
    var n_last: usize = 0;
    for (builtin.test_functions) |t| {
        if (endsWith(t.name, SERIAL_FIRST_SUFFIX)) {
            if (n_first < first_buf.len) {
                first_buf[n_first] = t;
                n_first += 1;
            }
        } else if (endsWith(t.name, SERIAL_SUFFIX)) {
            if (n_last < last_buf.len) {
                last_buf[n_last] = t;
                n_last += 1;
            }
        } else {
            if (n_parallel < parallel_buf.len) {
                parallel_buf[n_parallel] = t;
                n_parallel += 1;
            }
        }
    }

    // 16 is the cap on parallel workers — past that, GPU contention
    // (limited streams + serialized vkQueueSubmit) erases parallelism
    // benefits while inflating peak memory. The stack array uses the
    // same constant to match.
    const MAX_WORKERS = 16;
    const cpu_count = std.Thread.getCpuCount() catch 4;
    // VK adaptation: honour SLZ_VK_TEST_THREADS (1..MAX_WORKERS) so the
    // test runner can be coerced to single-threaded for race diagnosis.
    var worker_count: usize = @min(cpu_count, MAX_WORKERS);
    if (std.c.getenv("SLZ_VK_TEST_THREADS")) |env_c| {
        const env_str = std.mem.span(env_c);
        const env_n = std.fmt.parseInt(usize, env_str, 10) catch worker_count;
        worker_count = @min(@max(env_n, 1), MAX_WORKERS);
    }

    std.debug.print("Running {d} tests on {d} threads ({d} first-serial, {d} parallel, {d} last-serial)...\n", .{
        builtin.test_functions.len,
        worker_count,
        n_first,
        n_parallel,
        n_last,
    });

    // Phase 1: serial_first — run in the main thread BEFORE any worker
    // touches global driver state.
    for (first_buf[0..n_first]) |t| runOne(t);

    // Phase 2: parallel pool.
    test_fns = parallel_buf[0..n_parallel];
    next_test.store(0, .monotonic);

    var threads: [MAX_WORKERS]?std.Thread = @splat(null);
    for (0..worker_count) |wi| {
        threads[wi] = std.Thread.spawn(.{}, workerFn, .{}) catch null;
    }
    for (&threads) |*t| {
        if (t.*) |thread| thread.join();
    }

    // Phase 3: serial — run after parallel join, in main thread. These
    // include tests that spawn child processes (cli_smoke, cross_backend)
    // and tests that absolutely cannot share GPU state with siblings.
    for (last_buf[0..n_last]) |t| runOne(t);

    const ok = global_ok.load(.monotonic);
    const skip = global_skip.load(.monotonic);
    const fail = global_fail.load(.monotonic);

    std.debug.print("\n{d} passed, {d} skipped, {d} failed ({d} total)\n", .{ ok, skip, fail, builtin.test_functions.len });

    if (fail > 0) std.process.exit(1);
}
