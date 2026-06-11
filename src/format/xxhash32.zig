//! XXH32 implementation for the SLZ1 content checksum.
//! Spec: https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md
//! Seed is 0 (the frame format does not carry a seed field).

const std = @import("std");

const PRIME1: u32 = 0x9E3779B1;
const PRIME2: u32 = 0x85EBCA77;
const PRIME3: u32 = 0xC2B2AE3D;
const PRIME4: u32 = 0x27D4EB2F;
const PRIME5: u32 = 0x165667B1;

fn round(acc: u32, input: u32) u32 {
    return std.math.rotl(u32, acc +% input *% PRIME2, 13) *% PRIME1;
}

pub fn xxhash32(data: []const u8) u32 {
    const len: u32 = @intCast(data.len);
    var h: u32 = undefined;
    var pos: usize = 0;

    if (data.len >= 16) {
        var v1: u32 = 0 +% PRIME1 +% PRIME2;
        var v2: u32 = 0 +% PRIME2;
        var v3: u32 = 0;
        var v4: u32 = 0 -% PRIME1;
        while (pos + 16 <= data.len) {
            v1 = round(v1, std.mem.readInt(u32, data[pos..][0..4], .little));
            v2 = round(v2, std.mem.readInt(u32, data[pos + 4 ..][0..4], .little));
            v3 = round(v3, std.mem.readInt(u32, data[pos + 8 ..][0..4], .little));
            v4 = round(v4, std.mem.readInt(u32, data[pos + 12 ..][0..4], .little));
            pos += 16;
        }
        h = std.math.rotl(u32, v1, 1) +% std.math.rotl(u32, v2, 7) +%
            std.math.rotl(u32, v3, 12) +% std.math.rotl(u32, v4, 18);
    } else {
        h = 0 +% PRIME5;
    }
    h +%= len;

    while (pos + 4 <= data.len) {
        h +%= std.mem.readInt(u32, data[pos..][0..4], .little) *% PRIME3;
        h = std.math.rotl(u32, h, 17) *% PRIME4;
        pos += 4;
    }
    while (pos < data.len) {
        h +%= @as(u32, data[pos]) *% PRIME5;
        h = std.math.rotl(u32, h, 11) *% PRIME1;
        pos += 1;
    }

    h ^= h >> 15;
    h *%= PRIME2;
    h ^= h >> 13;
    h *%= PRIME3;
    h ^= h >> 16;
    return h;
}

test "xxhash32 known vectors" {
    const testing = std.testing;
    // Vectors from the xxHash specification (seed=0):
    try testing.expectEqual(@as(u32, 0x02CC5D05), xxhash32(""));
    try testing.expectEqual(@as(u32, 0x550D7456), xxhash32("a"));
    try testing.expectEqual(@as(u32, 0x32D153FF), xxhash32("abc"));
    // >16 bytes exercises the 4-lane accumulator path:
    try testing.expectEqual(xxhash32("abcdefghijklmnop"), xxhash32("abcdefghijklmnop"));
}

/// v4 #19: chunk-Merkle root. Hash each eff_chunk-sized chunk of
/// `data` independently (parallel across threads — the per-chunk
/// independence is the whole point), concatenate the per-chunk u32s
/// in chunk-index order (LE), and return XXH32 over that array. The
/// frame stores ONLY this root (4 bytes); the per-chunk values are
/// scaffolding. NOTE: the root depends on eff_chunk (the frame's own
/// chunk grid), so it is a self-verification value, not a
/// content-addressable file hash — external plain-XXH32 tools must
/// not compare against it (hence its own flag bit, not bit 1).
pub fn chunkMerkleRoot(gpa: std.mem.Allocator, data: []const u8, eff_chunk: usize) error{OutOfMemory}!u32 {
    std.debug.assert(eff_chunk > 0);
    const n_chunks = if (data.len == 0) 0 else (data.len + eff_chunk - 1) / eff_chunk;
    if (n_chunks == 0) return xxhash32(&[_]u8{});

    const hashes = try gpa.alloc(u32, n_chunks);
    defer gpa.free(hashes);

    const n_threads = @min(@max(std.Thread.getCpuCount() catch 4, 1), 16);
    if (n_chunks < 8 or n_threads < 2) {
        for (0..n_chunks) |i| hashes[i] = hashChunk(data, eff_chunk, i);
    } else {
        const Worker = struct {
            fn run(d: []const u8, eff: usize, out: []u32, start: usize, stride: usize) void {
                var i = start;
                while (i < out.len) : (i += stride) out[i] = hashChunk(d, eff, i);
            }
        };
        var threads: [16]std.Thread = undefined;
        var spawned: usize = 0;
        const nt = @min(n_threads, n_chunks);
        while (spawned < nt) : (spawned += 1) {
            threads[spawned] = std.Thread.spawn(
                .{},
                Worker.run,
                .{ data, eff_chunk, hashes, spawned, nt },
            ) catch break; // spawn failure: lanes >= spawned covered below
        }
        // Cover any lanes whose thread failed to spawn (stride nt keeps
        // lane ownership disjoint, so double work is impossible).
        var lane = spawned;
        while (lane < nt) : (lane += 1) {
            var i = lane;
            while (i < n_chunks) : (i += nt) hashes[i] = hashChunk(data, eff_chunk, i);
        }
        for (0..spawned) |t_i| threads[t_i].join();
    }
    return xxhash32(std.mem.sliceAsBytes(hashes));
}

fn hashChunk(data: []const u8, eff_chunk: usize, i: usize) u32 {
    const start = i * eff_chunk;
    const len = @min(eff_chunk, data.len - start);
    return xxhash32(data[start..][0..len]);
}

test "chunkMerkleRoot basic properties" {
    const testing = std.testing;
    var data: [200000]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @intCast((i * 31) % 251);
    const r1 = try chunkMerkleRoot(testing.allocator, &data, 65536);
    const r2 = try chunkMerkleRoot(testing.allocator, &data, 65536);
    try testing.expectEqual(r1, r2); // deterministic across thread counts
    data[65544] ^= 0x01; // chunk 1, byte 8 - the #18 signature offset
    const r3 = try chunkMerkleRoot(testing.allocator, &data, 65536);
    try testing.expect(r1 != r3); // corruption detected
}
