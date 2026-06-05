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

fn workerFn() void {
    while (true) {
        const i = next_test.fetchAdd(1, .monotonic);
        if (i >= test_fns.len) return;

        const test_fn = test_fns[i];
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
}

/// CUDA reference: src/test_runner_parallel.zig:37-end. Parallel test
/// runner entry point — collects builtin.test_functions, dispatches to
/// worker threads, reports the aggregated result.
pub fn main() void {
    test_fns = builtin.test_functions;

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

    std.debug.print("Running {d} tests on {d} threads...\n", .{ test_fns.len, worker_count });

    var threads: [MAX_WORKERS]?std.Thread = @splat(null);
    for (0..worker_count) |wi| {
        threads[wi] = std.Thread.spawn(.{}, workerFn, .{}) catch null;
    }
    for (&threads) |*t| {
        if (t.*) |thread| thread.join();
    }

    const ok = global_ok.load(.monotonic);
    const skip = global_skip.load(.monotonic);
    const fail = global_fail.load(.monotonic);

    std.debug.print("\n{d} passed, {d} skipped, {d} failed ({d} total)\n", .{ ok, skip, fail, test_fns.len });

    if (fail > 0) std.process.exit(1);
}
