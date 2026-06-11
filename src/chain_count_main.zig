//! v4 #14 gate-0 readback: run one L5 encode, then read the
//! SLZ_COUNT_CHAIN device globals from the encode module and print
//! the chain parser's work shape. Requires lz_chain_parser.cuh built
//! with SLZ_COUNT_CHAIN=1.
//!
//! Usage: chaincount <input-file>

const std = @import("std");
const encoder = @import("encode/streamlz_encoder.zig");
const gpu_encoder = @import("encode/driver.zig");
const enc_ml = @import("encode/module_loader.zig");
const dec_cuda = @import("decode/cuda_api.zig");

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;

    var args_it = try init.minimal.args.iterateAllocator(gpa);
    defer args_it.deinit();
    _ = args_it.next();
    const path = args_it.next() orelse {
        std.debug.print("usage: chaincount <input-file>\n", .{});
        return 1;
    };

    const src = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(src);
    const frame = try gpa.alloc(u8, encoder.compressBound(src.len));
    defer gpa.free(frame);

    const n = try encoder.compressFramed(gpa, src, frame, .{ .level = 5 }, &gpu_encoder.g_default);
    std.debug.print("encoded {d} -> {d}\n", .{ src.len, n });

    const get_global = dec_cuda.cuModuleGetGlobal_fn orelse {
        std.debug.print("cuModuleGetGlobal unavailable\n", .{});
        return 1;
    };
    const d2h = dec_cuda.cuMemcpyDtoH_fn orelse return 1;

    const names = [_][:0]const u8{ "g_slz_chain_calls", "g_slz_chain_cand", "g_slz_extend_bytes" };
    var vals: [3]u64 = .{ 0, 0, 0 };
    for (names, 0..) |name, i| {
        var dptr: usize = 0;
        var sz: usize = 0;
        if (get_global(&dptr, &sz, enc_ml.module, name.ptr) != 0) {
            std.debug.print("global {s}: not found (SLZ_COUNT_CHAIN off?)\n", .{name});
            return 1;
        }
        _ = d2h(@ptrCast(&vals[i]), dptr, 8);
    }
    const calls = vals[0];
    const cand = vals[1];
    const ext = vals[2];
    std.debug.print(
        "chain parser shape (L5, {d} bytes):\n" ++
            "  findMatchChain calls: {d}  ({d:.2} per input byte)\n" ++
            "  chain candidates:     {d}  ({d:.2} per call)\n" ++
            "  extend byte-compares: {d}  ({d:.2} per call)\n",
        .{
            src.len,
            calls,
            @as(f64, @floatFromInt(calls)) / @as(f64, @floatFromInt(src.len)),
            cand,
            @as(f64, @floatFromInt(cand)) / @as(f64, @floatFromInt(@max(calls, 1))),
            ext,
            @as(f64, @floatFromInt(ext)) / @as(f64, @floatFromInt(@max(calls, 1))),
        },
    );
    return 0;
}
