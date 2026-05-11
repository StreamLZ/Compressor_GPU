const std = @import("std");
const tans = @import("decode/entropy/tans_decoder.zig");

fn readFile(path: [*:0]const u8) ![]u8 {
    const c = @cImport({ @cInclude("stdio.h"); });
    const f = c.fopen(path, "rb") orelse return error.FileNotFound;
    defer _ = c.fclose(f);
    _ = c.fseek(f, 0, c.SEEK_END);
    const size: usize = @intCast(c.ftell(f));
    _ = c.fseek(f, 0, c.SEEK_SET);
    const buf = try std.heap.page_allocator.alloc(u8, size);
    _ = c.fread(buf.ptr, 1, size, f);
    return buf;
}

pub fn main() !void {
    const encoded = try readFile("c:\\tmp\\gpu_tans_encoded.bin");
    const original = try readFile("c:\\tmp\\gpu_tans_original.bin");
    std.debug.print("Encoded: {} bytes, Original: {} bytes\n", .{ encoded.len, original.len });

    var decoded: [65536]u8 = undefined;
    var scratch: [65536]u8 = undefined;

    // Also CPU-encode the same data for comparison
    const tans_enc = @import("encode/entropy/tans_encoder.zig");
    const hist_mod = @import("encode/entropy/byte_histogram.zig");
    var histo: hist_mod.ByteHistogram = .{};
    for (original) |b| histo.count[b] += 1;
    var cpu_dst: [8192]u8 = undefined;
    var cpu_cost: f32 = std.math.inf(f32);
    const cpu_n = tans_enc.encodeArrayU8Tans(std.heap.page_allocator, &cpu_dst, original, &histo, 1.0, &cpu_cost) catch |e| {
        std.debug.print("CPU tANS encode FAILED: {s}\n", .{@errorName(e)});
        return;
    };
    std.debug.print("CPU tANS encoded: {} bytes\n", .{cpu_n});
    std.debug.print("GPU bytes[0..16]:", .{});
    for (0..@min(16, encoded.len)) |i| std.debug.print(" {X:0>2}", .{encoded[i]});
    std.debug.print("\nCPU bytes[0..16]:", .{});
    for (0..@min(16, cpu_n)) |i| std.debug.print(" {X:0>2}", .{cpu_dst[i]});
    std.debug.print("\n", .{});

    // Find first byte difference and dump context
    for (0..@min(cpu_n, encoded.len)) |i| {
        if (cpu_dst[i] != encoded[i]) {
            std.debug.print("First byte diff at {}: CPU=0x{X:0>2} GPU=0x{X:0>2}\n", .{ i, cpu_dst[i], encoded[i] });
            const start = if (i >= 8) i - 8 else 0;
            const end = @min(i + 8, @min(cpu_n, encoded.len));
            std.debug.print("CPU[{}..{}]:", .{ start, end });
            for (start..end) |j| std.debug.print(" {X:0>2}", .{cpu_dst[j]});
            std.debug.print("\nGPU[{}..{}]:", .{ start, end });
            for (start..end) |j| std.debug.print(" {X:0>2}", .{encoded[j]});
            std.debug.print("\n", .{});
            break;
        }
    }

    const n = tans.highDecodeTans(
        encoded.ptr, encoded.len,
        &decoded, original.len,
        &scratch, scratch[scratch.len..].ptr,
    ) catch |e| {
        std.debug.print("CPU tANS decode of GPU data FAILED: {s}\n", .{@errorName(e)});
        // Try decoding CPU-encoded data instead
        const n2 = tans.highDecodeTans(
            &cpu_dst, cpu_n,
            &decoded, original.len,
            &scratch, scratch[scratch.len..].ptr,
        ) catch |e2| {
            std.debug.print("CPU tANS decode of CPU data ALSO FAILED: {s}\n", .{@errorName(e2)});
            return;
        };
        _ = n2;
        std.debug.print("CPU decode of CPU data: OK\n", .{});
        return;
    };
    _ = n;

    var mismatches: usize = 0;
    for (0..original.len) |i| {
        if (decoded[i] != original[i]) {
            if (mismatches < 5)
                std.debug.print("  mismatch at {}: expected 0x{X:0>2} got 0x{X:0>2}\n", .{ i, original[i], decoded[i] });
            mismatches += 1;
        }
    }
    if (mismatches == 0)
        std.debug.print("CPU decode of GPU-encoded tANS: BYTE-EXACT!\n", .{})
    else
        std.debug.print("CPU decode: {} mismatches\n", .{mismatches});
}
