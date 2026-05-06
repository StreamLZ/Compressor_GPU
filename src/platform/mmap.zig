//! Cross-platform memory-mapped file I/O.
//!
//! Provides `mapFileRead` / `mapFileReadWrite` for zero-copy file access.
//! On Windows uses CreateFileMappingW + MapViewOfFile.
//! On POSIX uses std.posix.mmap.

const std = @import("std");
const builtin = @import("builtin");
const is_windows = builtin.os.tag == .windows;

const win32 = if (is_windows) struct {
    const HANDLE = std.os.windows.HANDLE;
    const DWORD = u32;
    const BOOL = std.os.windows.BOOL;

    extern "kernel32" fn CreateFileMappingW(
        hFile: HANDLE,
        lpAttributes: ?*anyopaque,
        flProtect: DWORD,
        dwMaxSizeHigh: DWORD,
        dwMaxSizeLow: DWORD,
        lpName: ?[*:0]const u16,
    ) callconv(.winapi) ?HANDLE;

    extern "kernel32" fn MapViewOfFile(
        hMap: HANDLE,
        dwAccess: DWORD,
        dwOffHi: DWORD,
        dwOffLo: DWORD,
        dwBytes: usize,
    ) callconv(.winapi) ?*anyopaque;

    extern "kernel32" fn UnmapViewOfFile(lpBase: *const anyopaque) callconv(.winapi) BOOL;
    extern "kernel32" fn CloseHandle(h: HANDLE) callconv(.winapi) BOOL;

    const PAGE_READONLY: DWORD = 0x02;
    const PAGE_READWRITE: DWORD = 0x04;
    const FILE_MAP_READ: DWORD = 0x04;
    const FILE_MAP_WRITE: DWORD = 0x02;
} else struct {};

/// A memory-mapped view of a file (read-only or read-write).
pub const MappedFile = struct {
    /// Base pointer to the mapped region.
    ptr: [*]u8,
    /// Length of the mapped region in bytes.
    len: usize,
    /// OS mapping handle (Windows only; void on POSIX).
    map_handle: if (is_windows) ?std.os.windows.HANDLE else void,

    /// Return the mapped region as a const byte slice.
    pub fn sliceConst(self: MappedFile) []const u8 {
        return self.ptr[0..self.len];
    }

    /// Return the mapped region as a mutable byte slice.
    pub fn slice(self: MappedFile) []u8 {
        return self.ptr[0..self.len];
    }

    /// Release the memory mapping and close the OS handle.
    pub fn unmap(self: *MappedFile) void {
        if (is_windows) {
            _ = win32.UnmapViewOfFile(@ptrCast(self.ptr));
            if (self.map_handle) |h| _ = win32.CloseHandle(h);
        } else {
            const aligned: [*]align(std.mem.page_size) u8 = @alignCast(self.ptr);
            std.posix.munmap(aligned[0..self.len]);
        }
    }
};

/// Map an open file as read-only; returns null on failure.
pub fn mapFileRead(file: std.Io.File, size: usize) ?MappedFile {
    if (is_windows) {
        const map_h = win32.CreateFileMappingW(file.handle, null, win32.PAGE_READONLY, 0, 0, null) orelse return null;
        const view = win32.MapViewOfFile(map_h, win32.FILE_MAP_READ, 0, 0, 0) orelse {
            _ = win32.CloseHandle(map_h);
            return null;
        };
        return .{
            .ptr = @ptrCast(view),
            .len = size,
            .map_handle = map_h,
        };
    } else {
        const result = std.posix.mmap(null, size, std.posix.PROT.READ, .{ .TYPE = .SHARED }, file.handle, 0);
        const ptr = result catch return null;
        return .{
            .ptr = @ptrCast(ptr),
            .len = size,
            .map_handle = {},
        };
    }
}

/// Map an open file as read-write; returns null on failure.
pub fn mapFileReadWrite(file: std.Io.File, size: usize) ?MappedFile {
    if (is_windows) {
        const size_hi: win32.DWORD = @intCast(size >> 32);
        const size_lo: win32.DWORD = @intCast(size & 0xFFFFFFFF);
        const map_h = win32.CreateFileMappingW(file.handle, null, win32.PAGE_READWRITE, size_hi, size_lo, null) orelse return null;
        const view = win32.MapViewOfFile(map_h, win32.FILE_MAP_READ | win32.FILE_MAP_WRITE, 0, 0, 0) orelse {
            _ = win32.CloseHandle(map_h);
            return null;
        };
        return .{
            .ptr = @ptrCast(view),
            .len = size,
            .map_handle = map_h,
        };
    } else {
        const result = std.posix.mmap(null, size, std.posix.PROT.READ | std.posix.PROT.WRITE, .{ .TYPE = .SHARED }, file.handle, 0);
        const ptr = result catch return null;
        return .{
            .ptr = @ptrCast(ptr),
            .len = size,
            .map_handle = {},
        };
    }
}
