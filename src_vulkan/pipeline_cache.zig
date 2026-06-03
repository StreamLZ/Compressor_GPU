//! VkPipelineCache disk persistence — M7.
//!
//! Pipeline objects in Vulkan are notoriously expensive to compile (SPIR-V
//! → ISA, link, optimize) but the Vulkan API lets the application persist
//! the cumulative cache across runs by serializing the driver-managed blob
//! to disk. This module owns the on-disk side of that lifecycle:
//!
//!   • `cacheDir`     → resolves the per-OS cache directory and mkdirs it.
//!   • `cachePath`    → joins dir + "<hex(uuid)>.<tier>.bin" filename.
//!   • `loadOrCreate` → tries to read the blob; if absent or unreadable,
//!                      creates an empty in-memory cache. Never fatal.
//!   • `save`         → drains `vkGetPipelineCacheData` and writes the blob
//!                      atomically-enough for our use case (overwrite + on
//!                      Windows, retry on SHARING_VIOLATION up to 3×).
//!
//! Persistence is best-effort. The driver embeds a 32-byte header at the
//! start of the blob (vendorID/deviceID/pipelineCacheUUID); on the next
//! run, if any of those changed (driver upgrade, hardware swap), the
//! driver silently discards our seed and returns an empty cache — that is
//! the correct behavior, not an error worth bubbling to the user.
//!
//! M7 scope: serialization only. Cache-size cap and "invalidate-all-cached-
//! command-buffers when cache changes" live in M8b (descriptor factory).

const std = @import("std");
const builtin = @import("builtin");

const vk = @import("vk_api.zig");

// ── Constants ────────────────────────────────────────────────────
const SUBDIR_NAME: []const u8 = "streamlz_vk";

// Cap the driver-emitted blob at 256 MiB. Real-world pipeline caches are
// tens of MiB at the high end (~5-20 MiB on NVIDIA, ~30-50 MiB on AMD for
// heavy AAA workloads); 256 MiB is "the driver returned something insane,
// don't try to write it" rather than a tuning knob. Bytes above the cap
// are dropped — the next run rebuilds those pipelines.
const MAX_CACHE_BYTES: usize = 256 * 1024 * 1024;

// Windows SHARING_VIOLATION retry policy. The expected case is "another
// streamlz_vk process is currently saving the same .bin"; backoff is
// fixed at 100ms because the contending writer's write-and-close is
// microseconds, so any longer wait is gravy.
const WIN_SAVE_RETRY_ATTEMPTS: u8 = 3;
const WIN_SAVE_RETRY_BACKOFF_MS: u64 = 100;

// Win32 Sleep — kernel32 already linked elsewhere under src_vulkan/. Use
// the millisecond-resolution Sleep rather than std.Thread.sleep because
// we don't want to pull a Thread dependency into this library file.
extern "kernel32" fn Sleep(dwMilliseconds: u32) callconv(.c) void;

// ── Errors ───────────────────────────────────────────────────────
pub const PipelineCacheError = error{
    /// VkDevice is null or `vkCreatePipelineCache_fn` was never resolved.
    LoaderNotReady,
    /// vkCreatePipelineCache returned a non-VK_SUCCESS VkResult.
    CreateFailed,
    /// vkGetPipelineCacheData reported an error other than VK_INCOMPLETE.
    GetDataFailed,
    /// The driver emitted a blob larger than MAX_CACHE_BYTES. Dropped on the floor.
    CacheTooLarge,
    /// All Win32 retries for SHARING_VIOLATION exhausted.
    SaveContention,
    /// Caller's allocator exhausted while reading the cache file
    /// (forwarded from `readFileAlloc` instead of silently degrading
    /// to "no cache"; that would just defer the failure to the
    /// caller's next allocation, masking the real cause).
    OutOfMemory,
    /// Unexpected OS error reading the on-disk cache blob. Forwarded
    /// instead of swallowed so the caller sees the actual cause.
    Unexpected,
};

// ── cacheDir ─────────────────────────────────────────────────────
/// Returns the per-platform cache directory and ensures it exists.
///
/// Layout:
///   Windows: %LOCALAPPDATA%\streamlz_vk\        (fallback %TEMP%\streamlz_vk\)
///   Linux:   $XDG_CACHE_HOME/streamlz_vk/       (fallback $HOME/.cache/streamlz_vk/)
///   macOS:   $HOME/Library/Caches/streamlz_vk/
///
/// Caller owns the returned slice and must free it with `allocator`.
///
/// `io` is required because Zig 0.16's filesystem ops are vtable-routed
/// (no more `std.fs.cwd()` without an `io`). Downstream M8b passes the
/// process-init's io handle through.
pub fn cacheDir(allocator: std.mem.Allocator, io: std.Io) ![]u8 {
    const dir = try resolveCacheDirPath(allocator);
    errdefer allocator.free(dir);

    // mkdir -p semantics: createDirPath silently succeeds if the path
    // already exists as a directory. We accept that — losing the race to
    // another process that just created it is fine.
    std.Io.Dir.cwd().createDirPath(io, dir) catch |err| switch (err) {
        // Treat permission errors as soft failures so the caller can fall
        // back to an in-memory-only cache (no persistence). We still
        // return the path so the caller can log it.
        error.AccessDenied,
        error.PermissionDenied,
        error.ReadOnlyFileSystem,
        => {},
        else => return err,
    };
    return dir;
}

fn resolveCacheDirPath(allocator: std.mem.Allocator) ![]u8 {
    switch (builtin.os.tag) {
        .windows => {
            if (envOwned(allocator, "LOCALAPPDATA")) |base| {
                defer allocator.free(base);
                return std.fs.path.join(allocator, &.{ base, SUBDIR_NAME });
            }
            // Fallback: %TEMP% is always set on Windows.
            if (envOwned(allocator, "TEMP")) |base| {
                defer allocator.free(base);
                return std.fs.path.join(allocator, &.{ base, SUBDIR_NAME });
            }
            // Last resort: current directory.
            return std.fs.path.join(allocator, &.{ ".", SUBDIR_NAME });
        },
        .macos => {
            if (envOwned(allocator, "HOME")) |home| {
                defer allocator.free(home);
                return std.fs.path.join(allocator, &.{ home, "Library", "Caches", SUBDIR_NAME });
            }
            return std.fs.path.join(allocator, &.{ ".", SUBDIR_NAME });
        },
        else => {
            // Linux / Android / other XDG-style platforms.
            if (envOwned(allocator, "XDG_CACHE_HOME")) |base| {
                defer allocator.free(base);
                return std.fs.path.join(allocator, &.{ base, SUBDIR_NAME });
            }
            if (envOwned(allocator, "HOME")) |home| {
                defer allocator.free(home);
                return std.fs.path.join(allocator, &.{ home, ".cache", SUBDIR_NAME });
            }
            return std.fs.path.join(allocator, &.{ ".", SUBDIR_NAME });
        },
    }
}

/// Read an env var via libc getenv (matches the rest of src_vulkan/ — see
/// instance.zig SLZ_VK_VALIDATION). Returns an allocator-owned copy so the
/// caller can free uniformly regardless of source. Returns null when unset
/// OR empty (empty %LOCALAPPDATA% is not a usable base).
fn envOwned(allocator: std.mem.Allocator, name: [*:0]const u8) ?[]u8 {
    const raw = std.c.getenv(name) orelse return null;
    const s = std.mem.span(raw);
    if (s.len == 0) return null;
    return allocator.dupe(u8, s) catch null;
}

// ── cachePath ────────────────────────────────────────────────────
/// Composes the cache filename: `<dir>/<hex(uuid)>.<tier_tag>.bin`.
///
/// `device_uuid` is the 16-byte pipelineCacheUUID from
/// VkPhysicalDeviceProperties — guarantees a per-device cache file so a
/// laptop with switchable graphics doesn't clobber its own caches when the
/// active GPU changes. `tier_tag` is one of "tier1" / "tier1_nv" / "tier2"
/// (matches the SPV subdirectory naming in src_vulkan/spv/).
///
/// Caller owns the returned slice.
pub fn cachePath(
    allocator: std.mem.Allocator,
    dir: []const u8,
    device_uuid: [16]u8,
    tier_tag: []const u8,
) ![]u8 {
    // bytesToHex returns a stack [32]u8 — copy it into the formatted path.
    const hex = std.fmt.bytesToHex(device_uuid, .lower);
    const filename = try std.fmt.allocPrint(allocator, "{s}.{s}.bin", .{ hex, tier_tag });
    defer allocator.free(filename);
    return std.fs.path.join(allocator, &.{ dir, filename });
}

// ── loadOrCreate ─────────────────────────────────────────────────
/// Creates a VkPipelineCache, seeded from the on-disk blob at `path` if
/// it exists and is readable. Missing / unreadable files are NOT errors —
/// the function falls back to creating an empty cache.
///
/// Driver header validation: if `path` exists but the blob's embedded
/// vendorID/deviceID/pipelineCacheUUID doesn't match the current device,
/// the driver silently returns an empty cache via VK_SUCCESS. That is the
/// spec-defined behavior and the file gets overwritten on the next save.
pub fn loadOrCreate(dev: vk.VkDevice, io: std.Io, allocator: std.mem.Allocator, path: []const u8) PipelineCacheError!vk.VkPipelineCache {
    if (dev == null) return error.LoaderNotReady;
    const create = vk.vkCreatePipelineCache_fn orelse return error.LoaderNotReady;

    // Try to slurp the existing blob. Use the cap as the read limit so a
    // corrupted multi-GB file doesn't OOM us at start-up.
    //
    // Error policy: the "expected on first run / cache eviction"
    // variants degrade gracefully to an empty cache (returns
    // null → fall through to vkCreatePipelineCache with size=0).
    // OOM is a hard error — it indicates the caller's allocator is
    // exhausted; silently degrading would just defer the failure to
    // the next `try allocator.alloc` and confuse the cause. Other
    // unexpected variants (Unexpected from the OS layer, etc.) also
    // propagate so the caller sees the actual cause.
    const data_opt: ?[]u8 = std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        @enumFromInt(MAX_CACHE_BYTES),
    ) catch |err| switch (err) {
        error.FileNotFound,
        error.AccessDenied,
        error.PermissionDenied,
        error.IsDir,
        error.NotDir,
        error.StreamTooLong,
        => null,
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.Unexpected,
    };
    defer if (data_opt) |d| allocator.free(d);

    const ci: vk.VkPipelineCacheCreateInfo = if (data_opt) |d| .{
        .flags = 0,
        .initialDataSize = d.len,
        .pInitialData = @ptrCast(d.ptr),
    } else .{
        .flags = 0,
        .initialDataSize = 0,
        .pInitialData = null,
    };

    var cache: vk.VkPipelineCache = null;
    const r = create(dev, &ci, null, &cache);
    if (r != vk.VK_SUCCESS) return error.CreateFailed;
    return cache;
}

// ── save ─────────────────────────────────────────────────────────
/// Drains the driver's pipeline cache to `path`. Idempotent in the sense
/// that calling it repeatedly is safe; the previous file is overwritten.
///
/// On Windows, retries up to `WIN_SAVE_RETRY_ATTEMPTS` times on
/// SHARING_VIOLATION (mapped by the Zig stdlib to `error.AccessDenied`),
/// backing off `WIN_SAVE_RETRY_BACKOFF_MS` between attempts. Two
/// streamlz_vk processes racing to save the same .bin is the realistic
/// scenario (multi-tenant servers, parallel test runs); the loser drops
/// its blob and that is fine — the cache is monotone, the winner's blob
/// is a strict superset within seconds.
pub fn save(dev: vk.VkDevice, io: std.Io, allocator: std.mem.Allocator, cache: vk.VkPipelineCache, path: []const u8) PipelineCacheError!void {
    if (dev == null or cache == null) return error.LoaderNotReady;
    const get_data = vk.vkGetPipelineCacheData_fn orelse return error.LoaderNotReady;

    // Two-call idiom. First call discovers size, second copies bytes.
    var size: usize = 0;
    if (get_data(dev, cache, &size, null) != vk.VK_SUCCESS) return error.GetDataFailed;
    if (size == 0) return; // Nothing to persist (no pipelines compiled this run).
    if (size > MAX_CACHE_BYTES) return error.CacheTooLarge;

    const buf = allocator.alloc(u8, size) catch return error.GetDataFailed;
    defer allocator.free(buf);

    var actual = size;
    const r = get_data(dev, cache, &actual, @ptrCast(buf.ptr));
    // VK_INCOMPLETE means the driver wrote `actual` bytes (≤ size). The
    // spec says the partial result is still a valid cache blob (the
    // driver truncates at a pipeline boundary), so persist it as-is.
    if (r != vk.VK_SUCCESS and r != vk.VK_INCOMPLETE) return error.GetDataFailed;
    if (actual == 0) return;

    const payload = buf[0..actual];

    var attempt: u8 = 0;
    while (true) : (attempt += 1) {
        const write_result = std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = path,
            .data = payload,
            .flags = .{ .truncate = true },
        });
        if (write_result) |_| return;
        const err = write_result catch |e| e;

        // SHARING_VIOLATION on Windows maps to AccessDenied per the std
        // NtCreateFile error tables (Threaded.zig). Retry; on other OSes,
        // AccessDenied means the user can't write here — fail fast.
        const is_retryable = builtin.os.tag == .windows and err == error.AccessDenied;
        if (!is_retryable or attempt + 1 >= WIN_SAVE_RETRY_ATTEMPTS) {
            // Map remaining failures to the public error. We don't
            // differentiate disk-full vs permission denied here — both
            // mean "this run's pipeline cache won't persist" which the
            // caller handles uniformly.
            return error.SaveContention;
        }
        Sleep(@intCast(WIN_SAVE_RETRY_BACKOFF_MS));
    }
}

// ── Tests ────────────────────────────────────────────────────────
// These exercise the pure-string helpers only. The Vulkan-touching
// load/save paths land an integration test at M8b once the descriptor
// factory wires a real VkDevice into the test harness.

test "cachePath formats uuid + tier" {
    const allocator = std.testing.allocator;
    const uuid: [16]u8 = .{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10 };
    const path = try cachePath(allocator, "C:\\cache", uuid, "tier1");
    defer allocator.free(path);
    // path.join uses the platform separator; we only assert the trailing
    // filename to keep the test cross-platform.
    try std.testing.expect(std.mem.endsWith(u8, path, "0123456789abcdeffedcba9876543210.tier1.bin"));
}

test "cachePath accepts tier1_nv tag" {
    const allocator = std.testing.allocator;
    const uuid: [16]u8 = @splat(0xaa);
    const path = try cachePath(allocator, "/tmp/c", uuid, "tier1_nv");
    defer allocator.free(path);
    try std.testing.expect(std.mem.endsWith(u8, path, ".tier1_nv.bin"));
}
