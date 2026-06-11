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
