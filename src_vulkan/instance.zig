//! VkInstance lifecycle: create with Vulkan 1.3 → 1.2 fallback, optional
//! KHRONOS validation layer, and resolves the instance-level dispatch
//! slots into vk_api.zig once the handle is alive.
//!
//! M1 keeps the surface deliberately tiny: no extensions, no debug-
//! messenger, no surface (we are a compute-only port). Later milestones
//! will add VK_EXT_debug_utils plumbing alongside the validation toggle.

const std = @import("std");

const vk = @import("vk_api.zig");

const VALIDATION_LAYER_NAME: [*:0]const u8 = "VK_LAYER_KHRONOS_validation";

pub const InstanceError = error{
    LoaderNotReady,
    CreateInstanceFailed,
    InstanceProcMissing,
};

/// Build a VkInstance trying apiVersion 1.3 first; if the driver rejects
/// it with VK_ERROR_INCOMPATIBLE_DRIVER, retry at 1.2 (the Tier-2 floor
/// per the milestone plan). Validation layer is enabled when `want_validation`
/// is true AND the env var SLZ_VK_VALIDATION=1 — gating on both means the
/// caller can hard-disable validation even in debug builds, and CI can opt-
/// in without code changes.
pub fn createInstance(want_validation: bool) InstanceError!vk.VkInstance {
    if (vk.init_state != .ready) return error.LoaderNotReady;

    const validation_env = std.c.getenv("SLZ_VK_VALIDATION");
    const env_enabled = blk: {
        if (validation_env) |p| {
            const s = std.mem.span(p);
            break :blk std.mem.eql(u8, s, "1");
        }
        break :blk false;
    };
    const enable_validation = want_validation and env_enabled;

    var layer_names_storage: [1][*:0]const u8 = .{VALIDATION_LAYER_NAME};
    const layer_count: u32 = if (enable_validation) 1 else 0;
    const layer_ptr: ?[*]const [*:0]const u8 = if (enable_validation)
        @ptrCast(&layer_names_storage)
    else
        null;

    const create_fn = vk.vkCreateInstance_fn orelse return error.LoaderNotReady;

    // VK_EXT_debug_utils — pure-instrumentation extension that exposes the
    // vkCmd{Begin,End}DebugUtilsLabelEXT entry points dispatch.zig wraps
    // around recorded compute dispatches / buffer copies. Nsight Systems'
    // Vulkan trace and RenderDoc both consume these labels to attribute
    // per-kernel GPU intervals in their UI (without labels, Nsight reports
    // anonymous "vkCmdDispatch" entries that aren't useful for the per-
    // kernel breakdown the host-side profiling workflow needs).
    //
    // We attempt the create-instance call with the extension requested
    // first; if the loader/driver rejects with VK_ERROR_EXTENSION_NOT_
    // PRESENT (clean box without the Vulkan SDK installed) we retry
    // without it. The dispatch.zig label call sites null-check the
    // function-pointer slots before invoking, so missing extension is a
    // silent no-op on the production path.
    var ext_names_storage: [1][*:0]const u8 = .{vk.VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

    // Try 1.3 first, fall back to 1.2 on VK_ERROR_INCOMPATIBLE_DRIVER.
    // For each API version we first try WITH the debug_utils extension;
    // if that fails with EXTENSION_NOT_PRESENT we retry without it. Any
    // other failure exits the retry loop (e.g. host OOM, layer missing).
    const try_versions = [_]u32{ vk.VK_API_VERSION_1_3, vk.VK_API_VERSION_1_2 };
    var inst: vk.VkInstance = null;

    for (try_versions) |api_ver| {
        const app_info: vk.VkApplicationInfo = .{
            .pApplicationName = "streamlz_vk",
            .applicationVersion = vk.VK_MAKE_API_VERSION(0, 0, 1, 0),
            .pEngineName = "streamlz",
            .engineVersion = vk.VK_MAKE_API_VERSION(0, 0, 1, 0),
            .apiVersion = api_ver,
        };
        // Attempt #1 — request VK_EXT_debug_utils.
        const create_info_with_ext: vk.VkInstanceCreateInfo = .{
            .pApplicationInfo = &app_info,
            .enabledLayerCount = layer_count,
            .ppEnabledLayerNames = layer_ptr,
            .enabledExtensionCount = 1,
            .ppEnabledExtensionNames = @ptrCast(&ext_names_storage),
        };
        var r = create_fn(&create_info_with_ext, null, &inst);
        if (r == vk.VK_ERROR_EXTENSION_NOT_PRESENT) {
            // Retry without the extension — instrumentation is best-effort.
            const create_info_no_ext: vk.VkInstanceCreateInfo = .{
                .pApplicationInfo = &app_info,
                .enabledLayerCount = layer_count,
                .ppEnabledLayerNames = layer_ptr,
                .enabledExtensionCount = 0,
                .ppEnabledExtensionNames = null,
            };
            r = create_fn(&create_info_no_ext, null, &inst);
        }
        if (r == vk.VK_SUCCESS) {
            try resolveInstanceLevel(inst);
            return inst;
        }
        // Anything other than "wrong api version" → no point retrying with
        // a lower version (e.g. host OOM, layer missing).
        if (r != vk.VK_ERROR_INCOMPATIBLE_DRIVER) break;
    }
    return error.CreateInstanceFailed;
}

pub fn destroyInstance(inst: vk.VkInstance) void {
    if (inst == null) return;
    const f = vk.vkDestroyInstance_fn orelse return;
    f(inst, null);
}

/// Populate the instance-level function-pointer slots in vk_api.zig.
/// Called once vkCreateInstance returns VK_SUCCESS.
fn resolveInstanceLevel(inst: vk.VkInstance) InstanceError!void {
    vk.vkDestroyInstance_fn = vk.getInstanceProc(vk.FnDestroyInstance, inst, "vkDestroyInstance");
    vk.vkEnumeratePhysicalDevices_fn = vk.getInstanceProc(vk.FnEnumeratePhysicalDevices, inst, "vkEnumeratePhysicalDevices");
    vk.vkGetPhysicalDeviceProperties_fn = vk.getInstanceProc(vk.FnGetPhysicalDeviceProperties, inst, "vkGetPhysicalDeviceProperties");
    vk.vkGetPhysicalDeviceQueueFamilyProperties_fn = vk.getInstanceProc(vk.FnGetPhysicalDeviceQueueFamilyProperties, inst, "vkGetPhysicalDeviceQueueFamilyProperties");
    vk.vkCreateDevice_fn = vk.getInstanceProc(vk.FnCreateDevice, inst, "vkCreateDevice");
    vk.vkGetDeviceProcAddr_fn = vk.getInstanceProc(vk.FnGetDeviceProcAddr, inst, "vkGetDeviceProcAddr");
    // M2: core-1.1 Properties2/Features2 + device-ext enumeration. Probe
    // path null-checks these; we don't fail instance bring-up on absence
    // because Tier-2 classification only needs the M1 props for the
    // vendor_id read, but we expect them present when apiVersion >= 1.2.
    vk.vkGetPhysicalDeviceProperties2_fn = vk.getInstanceProc(vk.FnGetPhysicalDeviceProperties2, inst, "vkGetPhysicalDeviceProperties2");
    vk.vkGetPhysicalDeviceFeatures2_fn = vk.getInstanceProc(vk.FnGetPhysicalDeviceFeatures2, inst, "vkGetPhysicalDeviceFeatures2");
    vk.vkEnumerateDeviceExtensionProperties_fn = vk.getInstanceProc(vk.FnEnumerateDeviceExtensionProperties, inst, "vkEnumerateDeviceExtensionProperties");
    // vkGetDeviceQueue / vkDestroyDevice resolved here too — they're
    // technically device-level but vkGetInstanceProcAddr returns valid
    // dispatch-table thunks for them. device.zig refines vkGetDeviceQueue
    // via vkGetDeviceProcAddr once a VkDevice exists (one less hop).
    vk.vkDestroyDevice_fn = vk.getInstanceProc(vk.FnDestroyDevice, inst, "vkDestroyDevice");
    vk.vkGetDeviceQueue_fn = vk.getInstanceProc(vk.FnGetDeviceQueue, inst, "vkGetDeviceQueue");

    if (vk.vkDestroyInstance_fn == null or
        vk.vkEnumeratePhysicalDevices_fn == null or
        vk.vkGetPhysicalDeviceProperties_fn == null or
        vk.vkGetPhysicalDeviceQueueFamilyProperties_fn == null or
        vk.vkCreateDevice_fn == null or
        vk.vkGetDeviceProcAddr_fn == null or
        vk.vkDestroyDevice_fn == null or
        vk.vkGetDeviceQueue_fn == null)
    {
        return error.InstanceProcMissing;
    }
}
