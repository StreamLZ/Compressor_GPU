//! Runtime CPU cache size detection.
//! Terminology: "sc" / "SC" = "self-contained" throughout this module.
//!
//! Returns L1d, L2, and L3 cache sizes in bytes. Uses CPUID on x86-64
//! (zero OS dependencies), sysctlbyname on macOS/ARM, sysfs on Linux/ARM.
//! Falls back to conservative defaults if detection fails.

const std = @import("std");
const builtin = @import("builtin");

pub const CacheSizes = struct {
    l1d: usize,
    l2: usize,
    l3: usize,
};

const default_sizes: CacheSizes = .{
    .l1d = 32 * 1024,
    .l2 = 256 * 1024,
    .l3 = 8 * 1024 * 1024,
};

pub fn detect() CacheSizes {
    if (comptime builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .x86) {
        return detectCpuid();
    }
    if (comptime builtin.os.tag == .macos or builtin.os.tag == .ios) {
        return detectDarwin();
    }
    if (comptime builtin.os.tag == .linux) {
        return detectLinuxSysfs();
    }
    return default_sizes;
}

// ────────────────────────────────────────────────────────────
//  x86/x86-64: CPUID leaf 0x04 (Intel) / 0x8000001D (AMD)
// ────────────────────────────────────────────────────────────

fn cpuid(leaf: u32, sub: u32) struct { eax: u32, ebx: u32, ecx: u32, edx: u32 } {
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [_] "={eax}" (eax),
          [_] "={ebx}" (ebx),
          [_] "={ecx}" (ecx),
          [_] "={edx}" (edx),
        : [_] "{eax}" (leaf),
          [_] "{ecx}" (sub),
    );
    return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

fn parseCpuidLeaf(leaf_id: u32) CacheSizes {
    var result = CacheSizes{ .l1d = 0, .l2 = 0, .l3 = 0 };
    var sub: u32 = 0;
    while (sub < 32) : (sub += 1) {
        const r = cpuid(leaf_id, sub);
        const cache_type = r.eax & 0x1f;
        if (cache_type == 0) break;
        const level = (r.eax >> 5) & 0x7;
        const ways = (r.ebx >> 22) + 1;
        const partitions = ((r.ebx >> 12) & 0x3ff) + 1;
        const line_size = (r.ebx & 0xfff) + 1;
        const sets = r.ecx + 1;
        const size = ways * partitions * line_size * sets;
        switch (level) {
            1 => if (cache_type == 1) {
                result.l1d = size;
            },
            2 => {
                result.l2 = size;
            },
            3 => {
                result.l3 = size;
            },
            else => {},
        }
    }
    return result;
}

fn detectCpuid() CacheSizes {
    const result = parseCpuidLeaf(0x04);
    if (result.l1d > 0) return result;
    const amd = parseCpuidLeaf(0x8000001D);
    if (amd.l1d > 0) return amd;
    return default_sizes;
}

// ────────────────────────────────────────────────────────────
//  macOS / iOS: sysctlbyname
// ────────────────────────────────────────────────────────────

fn darwinSysctl(comptime name: [:0]const u8) ?usize {
    if (comptime builtin.os.tag != .macos and builtin.os.tag != .ios) return null;
    var value: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname(name.ptr, @ptrCast(&value), &len, null, 0) == 0 and value > 0) {
        return @intCast(value);
    }
    return null;
}

fn detectDarwin() CacheSizes {
    return .{
        .l1d = darwinSysctl("hw.l1dcachesize") orelse
            darwinSysctl("hw.perflevel0.l1dcachesize") orelse
            default_sizes.l1d,
        .l2 = darwinSysctl("hw.l2cachesize") orelse
            darwinSysctl("hw.perflevel0.l2cachesize") orelse
            default_sizes.l2,
        .l3 = darwinSysctl("hw.l3cachesize") orelse 0,
    };
}

// ────────────────────────────────────────────────────────────
//  Linux: /sys/devices/system/cpu/cpu0/cache/
// ────────────────────────────────────────────────────────────

fn readSysfsInt(comptime path: []const u8) ?usize {
    const file = std.fs.openFileAbsolute(path, .{}) catch return null;
    defer file.close();
    var buf: [32]u8 = undefined;
    const n = file.readAll(&buf) catch return null;
    const trimmed = std.mem.trimRight(u8, buf[0..n], "\n\r ");
    if (trimmed.len == 0) return null;
    const last = trimmed[trimmed.len - 1];
    const multiplier: usize = if (last == 'K') 1024 else if (last == 'M') 1024 * 1024 else 1;
    const num_str = if (last == 'K' or last == 'M') trimmed[0 .. trimmed.len - 1] else trimmed;
    const val = std.fmt.parseInt(usize, num_str, 10) catch return null;
    return val * multiplier;
}

fn detectLinuxSysfs() CacheSizes {
    var result = CacheSizes{ .l1d = 0, .l2 = 0, .l3 = 0 };
    const base = "/sys/devices/system/cpu/cpu0/cache/index";
    inline for (.{ "0", "1", "2", "3", "4", "5" }) |idx| {
        const size = readSysfsInt(base ++ idx ++ "/size") orelse continue;
        const level_val = readSysfsInt(base ++ idx ++ "/level") orelse continue;
        // Check type: read first byte, 'D'=Data, 'I'=Instruction, 'U'=Unified
        const type_file = std.fs.openFileAbsolute(base ++ idx ++ "/type", .{}) catch continue;
        defer type_file.close();
        var tbuf: [16]u8 = undefined;
        const tn = type_file.readAll(&tbuf) catch continue;
        if (tn == 0) continue;
        const is_data = tbuf[0] == 'D';
        const is_unified = tbuf[0] == 'U';
        switch (level_val) {
            1 => if (is_data) {
                result.l1d = size;
            },
            2 => if (is_data or is_unified) {
                result.l2 = size;
            },
            3 => if (is_unified) {
                result.l3 = size;
            },
            else => {},
        }
    }
    if (result.l1d == 0) return default_sizes;
    return result;
}

// ────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────

const testing = std.testing;

test "detect returns plausible values" {
    const c = detect();
    try testing.expect(c.l1d >= 8 * 1024);
    try testing.expect(c.l1d <= 512 * 1024);
    try testing.expect(c.l2 >= 32 * 1024);
    try testing.expect(c.l2 <= 64 * 1024 * 1024);
}

test "detect L1d is smaller than L2" {
    const c = detect();
    if (c.l2 > 0) {
        try testing.expect(c.l1d < c.l2);
    }
}
