//! 1:1 port of src/encode/cuda_ffi.zig (RENAMED per Section B).
//!
//! Encode-side driver FFI shim - shared by every srcVK/encode/*.zig
//! module. Holds the Win32 LoadLibrary handles, the VkResult /
//! VkDeviceBuffer aliases, the function-pointer typedefs (`Fn*`), and
//! the `pub var` slots the loader (module_loader.zig) populates at
//! init time. Every other encode sub-module reads the function
//! pointers from here so we have exactly one definition site for each
//! driver entrypoint.
//!
//! The encode side does NOT own VkDevice creation: module_loader.init
//! calls into the decode driver first so that exactly one
//! vkCreateInstance + vkCreateDevice runs per process (a second device
//! would clobber the decode side's allocations). Encode then loads its
//! own vulkan-1.dll handle + function pointers because every encode
//! sub-module imports vulkan_ffi.zig for its function-pointer slots -
//! this is a per-side namespace duplication, not a capability gap; the
//! decode side already resolves the same Vulkan entries into its own
//! slots. There are no vkCreateInstance_fn or vkCreateDevice_fn slots
//! here on purpose - encode never resolves them.
//!
//! The encode side's slot bank stays shaped EXACTLY like CUDA's
//! cuda_ffi.zig surface so call sites read identically across backends.
//! Slots are funneled through the same `procs` indirection used by
//! decode; the module-level `vk<Op>_fn` slots are the typed handles the
//! procs.* shim binds to.

const std = @import("std");

pub const win32 = struct {
    pub extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.c) ?*anyopaque;
    pub extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.c) ?*anyopaque;
};

/// CUDA reference: src/encode/cuda_ffi.zig:27. Driver result code.
/// Mirrors VkResult (c_int) - 0 is success, negatives are errors.
pub const VkResult = c_int;

/// CUDA reference: src/encode/cuda_ffi.zig:28. Logical device index.
pub const VkDevice = c_int;

/// CUDA reference: src/encode/cuda_ffi.zig:29 (CUdeviceptr → VkDeviceBuffer).
/// Opaque device-buffer handle the encode codec passes around in lieu
/// of CUdeviceptr. Same registry-key shape as decode/vulkan_api.zig
/// (must be type-compatible so encode + decode can share allocations
/// via VMA without re-wrapping).
pub const VkDeviceBuffer = u64;

/// CUDA reference: src/encode/cuda_ffi.zig:30 (CUDA_SUCCESS → VK_SUCCESS_RC).
/// Numeric 0 sentinel preserved.
pub const VK_SUCCESS_RC: VkResult = 0;

/// CUDA reference: src/encode/cuda_ffi.zig:35 (CUmodule). Opaque pipeline-
/// layout handle. CUDA had CUmodule (a loaded PTX module); the VK port
/// collapses the loaded-module concept to the per-pipeline VkPipelineLayout
/// since SPV blobs are one-pipeline-each (no aggregated PTX). Modelled as
/// usize so module_loader can use the typed name instead of bare usize
/// slots.
pub const VkPipelineLayout = usize;

/// CUDA reference: src/encode/cuda_ffi.zig:36 (CUfunction). Opaque
/// kernel/pipeline handle. CUDA had CUfunction (a __global__ entry
/// pulled from a CUmodule); the VK port maps it to a VkPipeline handle.
pub const VkPipeline = usize;

/// vulkan-1.dll handle - populated by module_loader.init(). Underscore-
/// prefixed because it's an implementation detail; only this file and
/// the encode module_loader touch it.
pub var _lib: ?*anyopaque = null;

// ── Driver API function signatures ──────────────────────────────
// CUDA reference: src/encode/cuda_ffi.zig:43-52. One typedef per
// driver entry point the encode side resolves. Names mirror the CUDA
// `Fn<Op>` pattern verbatim so call sites stay shape-compatible.

pub const FnModuleLoadData = *const fn (*VkPipelineLayout, [*]const u8) callconv(.c) VkResult;
pub const FnModuleGetFunction = *const fn (*VkPipeline, VkPipelineLayout, [*:0]const u8) callconv(.c) VkResult;
pub const FnMemAlloc = *const fn (*VkDeviceBuffer, usize) callconv(.c) VkResult;
pub const FnMemFree = *const fn (VkDeviceBuffer) callconv(.c) VkResult;
pub const FnMemcpyHtoD = *const fn (VkDeviceBuffer, *const anyopaque, usize) callconv(.c) VkResult;
pub const FnMemcpyDtoH = *const fn (*anyopaque, VkDeviceBuffer, usize) callconv(.c) VkResult;
pub const FnMemcpyDtoDAsync = *const fn (VkDeviceBuffer, VkDeviceBuffer, usize, usize) callconv(.c) VkResult;
pub const FnLaunchKernel = *const fn (
    VkPipeline,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    usize,
    [*]?*anyopaque,
    [*]?*anyopaque,
) callconv(.c) VkResult;
pub const FnCtxSync = *const fn () callconv(.c) VkResult;
pub const FnMemsetD8 = *const fn (VkDeviceBuffer, u8, usize) callconv(.c) VkResult;

// ── Function-pointer slots ──────────────────────────────────────
// CUDA reference: src/encode/cuda_ffi.zig:54-63. One slot per
// `Fn<Op>` above. module_loader.zig fills each slot at init time;
// every encode sub-module reads them on every launch. Named with the
// `vk` prefix instead of `cu` per Section B - the surface shape (one
// optional function-pointer slot per entry) is preserved verbatim.

pub var vkModuleLoadData_fn: ?FnModuleLoadData = null;
pub var vkModuleGetFunction_fn: ?FnModuleGetFunction = null;
pub var vkMemAlloc_fn: ?FnMemAlloc = null;
pub var vkMemFree_fn: ?FnMemFree = null;
pub var vkMemcpyHtoD_fn: ?FnMemcpyHtoD = null;
pub var vkMemcpyDtoH_fn: ?FnMemcpyDtoH = null;
pub var vkMemcpyDtoDAsync_fn: ?FnMemcpyDtoDAsync = null;
pub var vkLaunchKernel_fn: ?FnLaunchKernel = null;
pub var vkCtxSynchronize_fn: ?FnCtxSync = null;
pub var vkMemsetD8_fn: ?FnMemsetD8 = null;

/// CUDA reference: src/encode/cuda_ffi.zig:69-73. Resolve a single
/// exported function from the already-loaded vulkan-1.dll handle.
/// Returns null if either the handle or the symbol is missing - the
/// caller decides whether that's fatal (vkModuleLoadData etc.) or
/// merely disables an optional kernel path (huffman, assemble).
pub fn getProc(comptime T: type, name: [*:0]const u8) ?T {
    const h = _lib orelse return null;
    const raw = win32.GetProcAddress(h, name) orelse return null;
    return @ptrCast(raw);
}

// ────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "encode FFI slots default to null" {
    // Module_loader has not run; every encode sub-module that touches
    // these slots before init must see null so the caller can surface
    // BackendNotAvailable. Mirrors CUDA's "cu*_fn is null until init"
    // contract.
    try testing.expect(vkModuleLoadData_fn == null);
    try testing.expect(vkLaunchKernel_fn == null);
    try testing.expect(vkMemAlloc_fn == null);
}

test "getProc returns null when the library handle is unset" {
    // _lib starts at null before module_loader.init runs; getProc must
    // bail without touching GetProcAddress so a too-early call site
    // doesn't crash.
    const orig = _lib;
    _lib = null;
    defer _lib = orig;
    const p: ?FnMemAlloc = getProc(FnMemAlloc, "vkMemAlloc");
    try testing.expect(p == null);
}
