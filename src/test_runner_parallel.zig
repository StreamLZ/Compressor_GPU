//! Parallel test runner for StreamLZ.
//! Runs tests across multiple threads for faster CI/dev feedback.
//! Use: `zig build ptest`

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

pub fn main() void {
    test_fns = builtin.test_functions;

    const cpu_count = std.Thread.getCpuCount() catch 4;
    const worker_count: usize = @min(cpu_count, 16);

    std.debug.print("Running {d} tests on {d} threads...\n", .{ test_fns.len, worker_count });

    var threads: [16]?std.Thread = @splat(null);
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
