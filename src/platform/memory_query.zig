//! Platform-specific memory query and thread-budget calculation.

const std = @import("std");
const builtin = @import("builtin");
const is_windows = builtin.os.tag == .windows;

/// Estimated memory consumption per compression worker thread (40 MB).
pub const per_thread_memory_estimate: u64 = 40 * 1024 * 1024;
/// Percentage of total physical RAM used as the thread-budget ceiling.
pub const memory_budget_pct: u64 = 60;

const win32 = if (is_windows) struct {
    const MemoryStatusEx = extern struct {
        dwLength: u32,
        dwMemoryLoad: u32,
        ullTotalPhys: u64,
        ullAvailPhys: u64,
        ullTotalPageFile: u64,
        ullAvailPageFile: u64,
        ullTotalVirtual: u64,
        ullAvailVirtual: u64,
        ullAvailExtendedVirtual: u64,
    };

    extern "kernel32" fn GlobalMemoryStatusEx(lpBuffer: *MemoryStatusEx) callconv(.winapi) std.os.windows.BOOL;

    const SYSTEM_CPU_SET_INFORMATION = extern struct {
        Size: u32,
        Type: u32,
        Id: u32,
        Group: u16,
        LogicalProcessorIndex: u8,
        CoreIndex: u8,
        LastLevelCacheIndex: u8,
        NumaNodeIndex: u8,
        EfficiencyClass: u8,
        AllFlags: u8,
        Reserved: u32 = 0,
        AllocationTag: u64 = 0,
    };

    extern "kernel32" fn GetSystemCpuSetInformation(
        Information: ?[*]SYSTEM_CPU_SET_INFORMATION,
        BufferLength: u32,
        ReturnedLength: *u32,
        Process: ?std.os.windows.HANDLE,
        Flags: u32,
    ) callconv(.winapi) std.os.windows.BOOL;

    extern "kernel32" fn SetThreadSelectedCpuSets(
        Thread: std.os.windows.HANDLE,
        CpuSetIds: [*]const u32,
        CpuSetIdCount: u32,
    ) callconv(.winapi) std.os.windows.BOOL;

    extern "kernel32" fn GetCurrentThread() callconv(.winapi) std.os.windows.HANDLE;
} else struct {};

/// Query total physical RAM from the OS; returns 0 on unsupported platforms.
pub fn totalAvailableMemoryBytes() u64 {
    const os = builtin.os.tag;
    if (os == .windows) {
        var ms: win32.MemoryStatusEx = .{
            .dwLength = @sizeOf(win32.MemoryStatusEx),
            .dwMemoryLoad = 0,
            .ullTotalPhys = 0,
            .ullAvailPhys = 0,
            .ullTotalPageFile = 0,
            .ullAvailPageFile = 0,
            .ullTotalVirtual = 0,
            .ullAvailVirtual = 0,
            .ullAvailExtendedVirtual = 0,
        };
        if (win32.GlobalMemoryStatusEx(&ms) == .FALSE) return 0;
        return ms.ullTotalPhys;
    } else if (os == .linux) {
        var info: std.os.linux.Sysinfo = undefined;
        if (std.os.linux.sysinfo(&info) != 0) return 0;
        return @as(u64, info.totalram) * @as(u64, info.mem_unit);
    } else if (os == .macos or os == .ios) {
        var mem: u64 = 0;
        var size: usize = @sizeOf(u64);
        const rc = std.posix.system.sysctlbyname("hw.memsize", @ptrCast(&mem), &size, null, 0);
        if (rc != 0) return 0;
        return mem;
    } else {
        return 0;
    }
}

/// Compute the maximum worker thread count given source size and available RAM.
pub fn calculateMaxThreads(src_len: usize) u32 {
    const cpu: u32 = @intCast(std.Thread.getCpuCount() catch 1);
    const total_memory: u64 = totalAvailableMemoryBytes();
    if (total_memory == 0) return @max(@as(u32, 1), cpu);

    const memory_budget: u64 = (total_memory * memory_budget_pct) / 100;
    const shared_mem: u64 = src_len;
    if (memory_budget <= shared_mem) return 1;
    const available_for_threads: u64 = memory_budget - shared_mem;

    const max_by_memory: u64 = available_for_threads / per_thread_memory_estimate;
    if (max_by_memory == 0) return 1;
    const max_by_memory_u32: u32 = @intCast(@min(max_by_memory, @as(u64, cpu)));
    return @max(@as(u32, 1), max_by_memory_u32);
}

/// CPU topology snapshot: P-core / E-core counts and IDs for hybrid processors.
pub const CoreInfo = struct {
    p_core_count: u32,
    e_core_count: u32,
    p_core_ids: [64]u32 = .{0} ** 64,
    e_core_ids: [64]u32 = .{0} ** 64,
    total_cores: u32,
    is_hybrid: bool,
};

/// Detect P-core and E-core topology on the current system (Windows hybrid-aware).
pub fn detectCores() CoreInfo {
    if (!is_windows) {
        const cpu: u32 = @intCast(std.Thread.getCpuCount() catch 1);
        return .{ .p_core_count = cpu, .e_core_count = 0, .total_cores = cpu, .is_hybrid = false };
    }

    var needed: u32 = 0;
    _ = win32.GetSystemCpuSetInformation(null, 0, &needed, null, 0);
    if (needed == 0) {
        const cpu: u32 = @intCast(std.Thread.getCpuCount() catch 1);
        return .{ .p_core_count = cpu, .e_core_count = 0, .total_cores = cpu, .is_hybrid = false };
    }

    var buf: [256]win32.SYSTEM_CPU_SET_INFORMATION = undefined;
    const buf_bytes: u32 = @intCast(@min(needed, @sizeOf(@TypeOf(buf))));
    var returned: u32 = 0;
    if (win32.GetSystemCpuSetInformation(&buf, buf_bytes, &returned, null, 0) == .FALSE) {
        const cpu: u32 = @intCast(std.Thread.getCpuCount() catch 1);
        return .{ .p_core_count = cpu, .e_core_count = 0, .total_cores = cpu, .is_hybrid = false };
    }

    const count = returned / @sizeOf(win32.SYSTEM_CPU_SET_INFORMATION);
    var max_efficiency: u8 = 0;
    for (buf[0..count]) |info| {
        if (info.EfficiencyClass > max_efficiency) max_efficiency = info.EfficiencyClass;
    }

    var result: CoreInfo = .{
        .p_core_count = 0,
        .e_core_count = 0,
        .total_cores = @intCast(count),
        .is_hybrid = max_efficiency > 0,
    };

    for (buf[0..count]) |info| {
        if (info.EfficiencyClass == max_efficiency and result.p_core_count < 64) {
            result.p_core_ids[result.p_core_count] = info.Id;
            result.p_core_count += 1;
        } else if (result.e_core_count < 64) {
            result.e_core_ids[result.e_core_count] = info.Id;
            result.e_core_count += 1;
        }
    }

    if (result.p_core_count == 0) {
        result.p_core_count = @intCast(count);
        result.is_hybrid = false;
    }

    return result;
}

/// Pin the calling thread to the given CPU-set IDs (Windows only; no-op elsewhere).
pub fn pinCurrentThreadToCpuSet(ids: []const u32) void {
    if (!is_windows or ids.len == 0) return;
    _ = win32.SetThreadSelectedCpuSets(
        win32.GetCurrentThread(),
        ids.ptr,
        @intCast(ids.len),
    );
}

const testing = std.testing;

test "calculateMaxThreads: returns at least 1" {
    const n = calculateMaxThreads(0);
    try testing.expect(n >= 1);
    const m = calculateMaxThreads(1024 * 1024);
    try testing.expect(m >= 1);
}

test "calculateMaxThreads: doesn't exceed CPU count" {
    const cpu: u32 = @intCast(std.Thread.getCpuCount() catch 1);
    const n = calculateMaxThreads(1024);
    try testing.expect(n <= cpu);
}

test "calculateMaxThreads: scales down with huge src_len" {
    const n = calculateMaxThreads(std.math.maxInt(usize) / 2);
    try testing.expect(n >= 1);
}

test "totalAvailableMemoryBytes: returns a plausible value on Windows" {
    if (!is_windows) return;
    const mem = totalAvailableMemoryBytes();
    try testing.expect(mem >= 1 * 1024 * 1024 * 1024);
}
