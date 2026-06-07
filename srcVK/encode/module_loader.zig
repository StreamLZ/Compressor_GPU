//! 1:1 port of src/encode/module_loader.zig.
//!
//! Loads the encode-side SPV blobs (LZ encode, Huffman tables / encode,
//! and the assemble kernels) and resolves them into pipeline handles.
//! Calls the decode-side init first (single VkDevice — handle / instance
//! shared across encode + decode). LZ kernel resolution must succeed
//! for L1 encode; Huff + assemble are optional (Huff stays optional, the
//! L1 frame-assembly trio is required at L1).
//!
//! CUDA reference: src/encode/module_loader.zig (entire file).
//!
//! VK adaptation: the encode side imports decode/driver.zig and chains
//! into its init() so exactly one VkInstance + VkDevice + VMA allocator
//! is created per process. After the decode bring-up succeeds, encode
//! resolves the dynamic-linkage Vulkan entries into its parallel
//! vulkan_ffi.zig slots (the encode-side namespace), then builds a real
//! VkPipelineLayout + VkPipeline per encode kernel. Each pipeline is
//! registered with decode/module_loader.registerExternalPipeline so the
//! shared procs.launch_kernel can dispatch it through the same code
//! path the decode kernels use.

const std = @import("std");
const vulkan_ffi = @import("vulkan_ffi.zig");
const vulkan_api = @import("../decode/vulkan_api.zig");
const decode_driver = @import("../decode/driver.zig");
const decode_module_loader = @import("../decode/module_loader.zig");
const vma = @import("../vma.zig");
const spv_blobs = @import("spv_blobs");

const VkResult = vulkan_ffi.VkResult;
const VkPipelineLayout = vulkan_ffi.VkPipelineLayout;
const VkPipeline = vulkan_ffi.VkPipeline;
const VkDeviceBuffer = vulkan_ffi.VkDeviceBuffer;
const VK_SUCCESS_RC = vulkan_ffi.VK_SUCCESS_RC;

// CUDA reference: src/encode/module_loader.zig:39-48. Slot names verbatim
// per audit Section C.6.1 row 276. Handles store the real VkPipeline cast
// to usize through the VkPipelineLayout / VkPipeline aliases (both
// ultimately usize per vulkan_ffi.zig). procs.launch_kernel resolves the
// VkPipelineLayout used at bind time from the shared metadata registry
// decode/module_loader staples on.
pub var module: VkPipelineLayout = 0;
pub var kernel_fn: VkPipeline = 0;
pub var huff_module: VkPipelineLayout = 0;
pub var huff_tables_kernel_fn: VkPipeline = 0;
pub var huff_encode_kernel_fn: VkPipeline = 0;
pub var assemble_module: VkPipelineLayout = 0;
pub var assemble_measure_fn: VkPipeline = 0;
pub var assemble_write_fn: VkPipeline = 0;
pub var frame_assemble_fn: VkPipeline = 0;
pub var initialized: bool = false;

// Local Vulkan API minimum used to call the pipeline-building entries on
// the already-brought-up VkDevice. The decode-side module_loader
// resolves the full instance/device entry-point set; encode only needs
// the descriptor-set-layout + pipeline-layout + compute-pipeline +
// shader-module create entry points through the encode-side FFI
// namespace.
const VkDevice = ?*opaque {};
const VkShaderModule = u64;
const VkDescriptorSetLayout = u64;
const VkPipelineCache = u64;
const PFN_vkVoidFunction = ?*const fn () callconv(.c) void;

const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: c_int = 16;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: c_int = 30;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: c_int = 32;
const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: c_int = 29;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: c_int = 18;
const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: c_int = 7;
const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x00000020;

// VK adaptation: subgroup size pinned to 32 via REQUIRE_FULL_SUBGROUPS_BIT
// + VkPipelineShaderStageRequiredSubgroupSizeCreateInfo. Kernels in
// gpu_warp.glsl hardcode WARP_SIZE=32 (the warpShuffle/Ballot/Any
// cooperative ops are correct only at exactly 32 lanes). Without this
// pin, Intel iGPU (supports [8,32]) would silently miscompile while
// NVIDIA (only supports [32,32]) works by accident. Encode-side mirror
// of the decode-side pin in decode/module_loader.zig::buildKernel.
// Per vulkan_core.h (1.4.341):
//   VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT = 0x00000002
//   VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO = 1000225001
// Prior values (0x8 / 1000225002) were both wrong: 0x8 is a non-existent
// VkPipelineShaderStageCreateFlagBits value (silently ignored by the driver),
// and 1000225002 is VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES
// (driver therefore interpreted our pNext as the features struct, ignoring
// the requiredSubgroupSize field). Net effect: on Intel iGPU the encode
// pipeline ran with the driver's default subgroupSize=16 (two 16-wide
// subgroups per 32-thread workgroup) and corrupted output. NVIDIA hardware
// only supports subgroupSize=32 so the broken pin was inert there.
// Validation layer surfaced both errors as
// VUID-VkPipelineShaderStageCreateInfo-pNext-pNext (wrong sType) +
// VUID-VkPipelineShaderStageCreateInfo-flags-parameter (flag 0x8 invalid).
const VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT: u32 = 0x00000002;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO: c_int = 1000225001;

const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
    pNext: ?*anyopaque = null,
    requiredSubgroupSize: u32,
};

const VkShaderModuleCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    codeSize: usize,
    pCode: [*]const u32,
};

const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32,
    descriptorType: c_int,
    descriptorCount: u32,
    stageFlags: u32,
    pImmutableSamplers: ?*const anyopaque = null,
};

const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    bindingCount: u32,
    pBindings: ?[*]const VkDescriptorSetLayoutBinding,
};

const VkPushConstantRange = extern struct {
    stageFlags: u32,
    offset: u32,
    size: u32,
};

const VkPipelineLayoutCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    setLayoutCount: u32,
    pSetLayouts: ?[*]const VkDescriptorSetLayout,
    pushConstantRangeCount: u32,
    pPushConstantRanges: ?[*]const VkPushConstantRange,
};

const VkSpecializationInfo = extern struct {
    mapEntryCount: u32 = 0,
    pMapEntries: ?*const anyopaque = null,
    dataSize: usize = 0,
    pData: ?*const anyopaque = null,
};

const VkPipelineShaderStageCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: u32,
    module: VkShaderModule,
    pName: [*:0]const u8,
    pSpecializationInfo: ?*const VkSpecializationInfo = null,
};

const VkComputePipelineCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: VkPipelineShaderStageCreateInfo,
    layout: u64, // VkPipelineLayout in this loader's local namespace.
    basePipelineHandle: u64 = 0,
    basePipelineIndex: i32 = -1,
};

const FnCreateShaderModule = *const fn (VkDevice, *const VkShaderModuleCreateInfo, ?*const anyopaque, *VkShaderModule) callconv(.c) VkResult;
const FnGetDeviceProcAddr = *const fn (VkDevice, [*:0]const u8) callconv(.c) PFN_vkVoidFunction;
const FnGetInstanceProcAddr = *const fn (?*anyopaque, [*:0]const u8) callconv(.c) PFN_vkVoidFunction;
const FnCreateDescriptorSetLayout = *const fn (VkDevice, *const VkDescriptorSetLayoutCreateInfo, ?*const anyopaque, *VkDescriptorSetLayout) callconv(.c) VkResult;
const FnCreatePipelineLayout = *const fn (VkDevice, *const VkPipelineLayoutCreateInfo, ?*const anyopaque, *u64) callconv(.c) VkResult;
const FnCreateComputePipelines = *const fn (VkDevice, VkPipelineCache, u32, [*]const VkComputePipelineCreateInfo, ?*const anyopaque, [*]u64) callconv(.c) VkResult;

var vkCreateShaderModule_fn: ?FnCreateShaderModule = null;
var vkCreateDescriptorSetLayout_fn: ?FnCreateDescriptorSetLayout = null;
var vkCreatePipelineLayout_fn: ?FnCreatePipelineLayout = null;
var vkCreateComputePipelines_fn: ?FnCreateComputePipelines = null;

// Per-encode-kernel binding metadata. Mirrors the .comp layout
// declarations under srcVK/encode/. Bindings are populated from each
// kernel's .comp source file; an entry with n_bindings = 0 and
// push_constant_size = 0 reflects the current .comp source declaring
// no bindings.
const EncodeKernelDecl = struct {
    name: []const u8,
    n_bindings: u32,
    push_constant_size: u32,
};

const ENCODE_KERNELS = struct {
    // Bindings + push-constant sizes mirror the .comp layout declarations
    // under srcVK/encode/. Each entry is CUDA arg-list verbatim — N SSBO
    // args become bindings 0..N-1, the trailing scalar args ride in a
    // single push-constant block.
    //
    // lz_encode: SSBO[0]=input, [1]=output, [2]=descs, [3]=global_hash,
    //   [4]=comp_sizes; push={total_chunks,hash_bits,use_chain,l4_features} = 16 B.
    pub const lz: EncodeKernelDecl = .{ .name = "lz_encode", .n_bindings = 5, .push_constant_size = 16 };
    // huff_build_tables: SSBO[0]=input, [1]=descs, [2]=code_lengths_out,
    //   [3]=codes_out; push={tables_stride, n_blocks} = 8 B.
    pub const huff_tables: EncodeKernelDecl = .{ .name = "huff_build_tables", .n_bindings = 4, .push_constant_size = 8 };
    // huff_encode_4stream: SSBO[0]=input, [1]=descs_in, [2]=code_lengths,
    //   [3]=codes, [4]=scratch, [5]=output, [6]=out_sizes;
    //   push={scratch_per_stream, tables_stride, n_blocks} = 12 B.
    pub const huff_encode: EncodeKernelDecl = .{ .name = "huff_encode_4stream", .n_bindings = 7, .push_constant_size = 12 };
    // assemble_measure: SSBO[0..3]={d_raw,d_huff_lit,d_huff_tok,d_huff_off16},
    //   [4]=descs, [5]=enc_sizes, [6]=scratch_u8 (a tiny placeholder needed
    //   by the assembleSubChunk macro's static l-value check — see the
    //   comment in assemble_measure_kernel.comp); push={n_subchunks} = 4 B.
    pub const assemble_measure: EncodeKernelDecl = .{ .name = "assemble_measure", .n_bindings = 7, .push_constant_size = 4 };
    // assemble_write: SSBO[0..3] same as measure, [4]=descs, [5]=d_frame;
    //   push={n_subchunks} = 4 B.
    pub const assemble_write: EncodeKernelDecl = .{ .name = "assemble_write", .n_bindings = 6, .push_constant_size = 4 };
    // frame_assemble: SSBO[0]=d_input, [1]=d_asm_out, [2]=d_asm_offsets,
    //   [3]=d_asm_chunk_sizes, [4]=d_chunk_dst, [5]=d_prefix_bytes,
    //   [6]=d_output; push={prefix_size, hdr0, hdr1, n_chunks, eff_chunk_size,
    //   src_len, sc_tail_off, end_mark_off} = 32 B.
    pub const frame_assemble: EncodeKernelDecl = .{ .name = "frame_assemble", .n_bindings = 7, .push_constant_size = 32 };
};

/// CUDA reference: src/encode/module_loader.zig:50-98. One-shot loader.
/// Brings up the encode-side pipelines on top of the decode-side init.
pub fn init() bool {
    // Fast path: post-init, every dispatch enters here. Read the flag
    // first so the steady-state hot path never touches the SRWLOCK.
    if (initialized) return kernel_fn != 0;
    // VK adaptation: serialize the one-shot encode init across the
    // ptest_vk 16-worker test runner. Two threads both seeing
    // `initialized == false` would each run LoadLibraryA + buildPipeline
    // against the same module slots; the second runner clobbers the
    // first's VkPipeline handles. Use a dedicated encode-init lock —
    // distinct from the decode g_init_lock so the recursive
    // encode → decode init chain is non-recursive on each lock.
    decode_module_loader.lockEncodeInitMutex();
    defer decode_module_loader.unlockEncodeInitMutex();
    if (initialized) return kernel_fn != 0;
    // VK adaptation: ONLY flip `initialized` to true after the full
    // init body succeeds. Pre-iter5 the flag was set at entry; a
    // mid-init failure (decode_driver.init false / vulkan-1.dll load
    // failure / pipeline-build failure) latched `initialized=true`
    // with `kernel_fn=0`, locking every future encode dispatch into
    // the false branch of `return kernel_fn != 0`.
    var ok = false;
    defer if (ok) {
        initialized = true;
    };

    // Reuse the VkDevice + VMA allocator the decode driver brings up.
    // CUDA reference: src/encode/module_loader.zig:56-57.
    if (!decode_driver.init()) return false;

    // Load vulkan-1.dll into the encode-side namespace. The decode side
    // owns its own LoadLibraryA handle; the encode side duplicates the
    // namespace so encode sub-modules read function pointers off
    // vulkan_ffi.zig instead of vulkan_api.zig.
    // CUDA reference: src/encode/module_loader.zig:59-60.
    vulkan_ffi._lib = vulkan_ffi.win32.LoadLibraryA("vulkan-1.dll");
    if (vulkan_ffi._lib == null) return false;

    // Resolve the encode-side FFI slot bank. Each slot in vulkan_ffi.zig
    // mirrors the encode-side cu*_fn slots in src/encode/cuda_ffi.zig.
    // VK adaptation: every slot points at the same module-private VMA-
    // backed implementation the decode loader staged into
    // vulkan_api.procs; the encode-side wrappers re-export those calls
    // through the encode-shaped FFI signatures so encode sub-modules
    // never touch vulkan_api.zig directly.
    vulkan_ffi.vkModuleLoadData_fn = encodeModuleLoadData;
    vulkan_ffi.vkModuleGetFunction_fn = encodeModuleGetFunction;
    vulkan_ffi.vkMemAlloc_fn = encodeMemAlloc;
    vulkan_ffi.vkMemFree_fn = encodeMemFree;
    vulkan_ffi.vkMemcpyHtoD_fn = encodeMemcpyHtoD;
    vulkan_ffi.vkMemcpyDtoH_fn = encodeMemcpyDtoH;
    vulkan_ffi.vkMemcpyDtoDAsync_fn = encodeMemcpyDtoDAsync;
    vulkan_ffi.vkLaunchKernel_fn = encodeLaunchKernel;
    vulkan_ffi.vkCtxSynchronize_fn = encodeCtxSync;
    vulkan_ffi.vkMemsetD8_fn = encodeMemsetD8;

    // Resolve the four Vulkan entry points the encode loader needs to
    // build its pipelines. They live off the same VkInstance/VkDevice
    // the decode side already brought up.
    const gipa = vulkan_ffi.getProc(FnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse return false;
    const inst_handle: ?*anyopaque = @ptrFromInt(vulkan_api.instance);
    const gdpa_raw = gipa(inst_handle, "vkGetDeviceProcAddr") orelse return false;
    const gdpa: FnGetDeviceProcAddr = @ptrCast(gdpa_raw);
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    vkCreateShaderModule_fn = @ptrCast(gdpa(dev, "vkCreateShaderModule") orelse return false);
    vkCreateDescriptorSetLayout_fn = @ptrCast(gdpa(dev, "vkCreateDescriptorSetLayout") orelse return false);
    vkCreatePipelineLayout_fn = @ptrCast(gdpa(dev, "vkCreatePipelineLayout") orelse return false);
    vkCreateComputePipelines_fn = @ptrCast(gdpa(dev, "vkCreateComputePipelines") orelse return false);

    // LZ encode kernel. Required for L1 encode per audit Section C.6.1.
    // CUDA reference: src/encode/module_loader.zig:73-77.
    if (!buildPipeline(spv_blobs.lz_encode, ENCODE_KERNELS.lz, &module, &kernel_fn)) return false;

    // Huffman encode kernels. Optional — chunk_type=4 only. CUDA loader
    // returns true even if these are missing (the encoder falls back to
    // the CPU Huffman path). Mirrored here.
    // CUDA reference: src/encode/module_loader.zig:81-86.
    if (buildPipeline(spv_blobs.huff_build_tables, ENCODE_KERNELS.huff_tables, &huff_module, &huff_tables_kernel_fn)) {
        var tmp_layout: VkPipelineLayout = 0;
        var tmp_fn: VkPipeline = 0;
        if (buildPipeline(spv_blobs.huff_encode_4stream, ENCODE_KERNELS.huff_encode, &tmp_layout, &tmp_fn)) {
            huff_encode_kernel_fn = tmp_fn;
        }
    }

    // Frame-assembly kernels. Required at L1 per audit Section C.6.1
    // (the L1 frame-assemble trio is on the L1 hot path).
    // CUDA reference: src/encode/module_loader.zig:90-95.
    if (buildPipeline(spv_blobs.assemble_measure, ENCODE_KERNELS.assemble_measure, &assemble_module, &assemble_measure_fn)) {
        var aw_layout: VkPipelineLayout = 0;
        if (buildPipeline(spv_blobs.assemble_write, ENCODE_KERNELS.assemble_write, &aw_layout, &assemble_write_fn)) {}
        var fa_layout: VkPipelineLayout = 0;
        if (buildPipeline(spv_blobs.frame_assemble, ENCODE_KERNELS.frame_assemble, &fa_layout, &frame_assemble_fn)) {}
    }

    // Tell the deferred `initialized = true` setter at the top that we
    // got here cleanly (kernel_fn was populated by the lz buildPipeline
    // call above; subsequent dispatches return kernel_fn != 0).
    ok = true;
    return true;
}

/// CUDA reference: src/encode/module_loader.zig:100-102. True iff init()
/// completed successfully.
pub fn isAvailable() bool {
    return init();
}

/// VK adaptation (H2): mirrors decode/module_loader.zig::ensurePipelineStream.
/// Lazily allocates the EncodeContext's library-owned VkStream via the
/// shared `procs.stream_create` slot the decode loader staged. The stream
/// inherits the split-submit + dedicated transfer-queue plumbing from
/// decode iters 11-13 — every subsequent encode H2D / launch / D2H on
/// it routes through the same transfer cmdbuf + compute cmdbuf pair that
/// decode's pipeline_stream uses, batching the encode pipeline's ~20
/// sync points into a single end-of-frame procs.stream_sync.
pub fn ensurePipelineStream(enc_ctx: *@import("encode_context.zig").EncodeContext) bool {
    if (enc_ctx.pipeline_stream_created) return true;
    const create = vulkan_api.procs.stream_create orelse return false;
    var stream: vulkan_api.VkStream = 0;
    if (create(&stream, 0) != vulkan_api.VK_SUCCESS_RC) return false;
    enc_ctx.pipeline_stream = stream;
    enc_ctx.pipeline_stream_created = true;
    return true;
}

// ── Internal helpers ──────────────────────────────────────────────────

/// Load a SPV blob into a VkShaderModule.
///
/// VK adaptation (valfix C — VUID-VkShaderModuleCreateInfo-pCode-08737):
/// `glslc -O` reliably emits `OpConstant %uchar N` instructions when it
/// constant-folds expressions that mask a uint down to a single byte
/// (e.g. `uint8_t(magic & 0xFFu)` collapsing to `uint8_t(0x31)` =
/// `OpConstant %uchar 49`). The compiler emits the
/// `StorageBuffer8BitAccess` capability (because the shader's storage
/// buffer reads/writes uint8_t) but FAILS to emit the `Int8` capability
/// that 8-bit scalar constants require per SPIR-V spec — surfacing as
/// VUID-VkShaderModuleCreateInfo-pCode-08737 ("Cannot form constants of
/// 8- or 16-bit types"). spirv-opt does not rewrite this; -O0 avoids
/// it but kills perf.
///
/// Patch in driver code: if the blob declares any 8-bit storage
/// capability but is missing Int8, splice in `OpCapability Int8`
/// right after the last OpCapability instruction. This is the SAME
/// fix the validation-error suggestion implies and is byte-equal to
/// what glslc would emit if its optimizer behaved correctly. Cost is
/// 1 SPIR-V word (8 bytes) per shader, applied at load time only.
fn loadShaderModule(spv: []const u8) VkShaderModule {
    if (spv.len == 0 or spv.len % 4 != 0) return 0;
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);

    var patched_buf: ?[]u32 = null;
    defer if (patched_buf) |p| std.heap.page_allocator.free(p);
    const patched = spvPatchAddInt8If8BitStorage(spv) catch null;
    const code_ptr: [*]const u32 = blk: {
        if (patched) |p| {
            patched_buf = p;
            break :blk p.ptr;
        }
        break :blk @ptrCast(@alignCast(spv.ptr));
    };
    const code_size: usize = if (patched) |p| p.len * 4 else spv.len;
    const ci = VkShaderModuleCreateInfo{
        .codeSize = code_size,
        .pCode = code_ptr,
    };
    var sm: VkShaderModule = 0;
    if (vkCreateShaderModule_fn.?(dev, &ci, null, &sm) != VK_SUCCESS_RC) return 0;
    return sm;
}

// SPIR-V capability constants (Capability enum, vulkan spec).
const SPV_CAP_INT16: u32 = 22;
const SPV_CAP_INT8: u32 = 39;
const SPV_CAP_STORAGE_BUFFER_8BIT_ACCESS: u32 = 4448;
const SPV_CAP_UNIFORM_AND_STORAGE_BUFFER_8BIT_ACCESS: u32 = 4449;
const SPV_CAP_STORAGE_PUSH_CONSTANT_8: u32 = 4450;
const SPV_OP_CAPABILITY: u32 = 17;
const SPV_MAGIC: u32 = 0x07230203;

/// Returns a patched copy of `spv` with `OpCapability Int8` inserted
/// if the blob declares any 8-bit storage capability but lacks Int8.
/// Returns null if no patch is needed (caller passes original blob to
/// vkCreateShaderModule untouched).
fn spvPatchAddInt8If8BitStorage(spv: []const u8) !?[]u32 {
    if (spv.len < 20) return null; // header is 5 words
    const words_in: []const u32 = @as([*]const u32, @ptrCast(@alignCast(spv.ptr)))[0 .. spv.len / 4];
    if (words_in[0] != SPV_MAGIC) return null;

    var has_8bit_storage = false;
    var has_int8 = false;
    var last_cap_idx: usize = 5; // after header
    var i: usize = 5;
    while (i < words_in.len) {
        const opcode: u32 = words_in[i] & 0xFFFF;
        const length: u32 = words_in[i] >> 16;
        if (length == 0) return null; // malformed
        if (opcode == SPV_OP_CAPABILITY and length == 2) {
            const cap = words_in[i + 1];
            if (cap == SPV_CAP_INT8) has_int8 = true;
            if (cap == SPV_CAP_STORAGE_BUFFER_8BIT_ACCESS or
                cap == SPV_CAP_UNIFORM_AND_STORAGE_BUFFER_8BIT_ACCESS or
                cap == SPV_CAP_STORAGE_PUSH_CONSTANT_8) has_8bit_storage = true;
            last_cap_idx = i + length;
        } else if (opcode == SPV_OP_CAPABILITY) {
            last_cap_idx = i + length;
        } else if (opcode != 5 and opcode != 14 and opcode != 11) {
            // OpExtension = 10, OpExtInstImport = 11, OpMemoryModel = 14,
            // OpSource = 3, OpName = 5 — all may follow capability block.
            // The capability block ends before the first non-capability/
            // non-extension instruction; once we pass OpMemoryModel we've
            // definitively left it. Simpler: stop scanning once we hit
            // OpMemoryModel (opcode 14) — capabilities must come before.
            if (opcode == 14) break;
        }
        i += length;
    }

    if (!has_8bit_storage or has_int8) return null;

    // Patch: insert `OpCapability Int8` (2 words: (2<<16)|17, 39) at
    // last_cap_idx (right after the last OpCapability instruction).
    const out = try std.heap.page_allocator.alloc(u32, words_in.len + 2);
    @memcpy(out[0..last_cap_idx], words_in[0..last_cap_idx]);
    out[last_cap_idx] = (@as(u32, 2) << 16) | SPV_OP_CAPABILITY;
    out[last_cap_idx + 1] = SPV_CAP_INT8;
    @memcpy(out[last_cap_idx + 2 ..], words_in[last_cap_idx..]);
    return out;
}

fn buildDescriptorSetLayout(n_bindings: u32) VkDescriptorSetLayout {
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    if (n_bindings == 0) {
        const ci = VkDescriptorSetLayoutCreateInfo{
            .bindingCount = 0,
            .pBindings = null,
        };
        var layout: VkDescriptorSetLayout = 0;
        if (vkCreateDescriptorSetLayout_fn.?(dev, &ci, null, &layout) != VK_SUCCESS_RC) return 0;
        return layout;
    }
    const gpa = std.heap.page_allocator;
    const bindings = gpa.alloc(VkDescriptorSetLayoutBinding, n_bindings) catch return 0;
    defer gpa.free(bindings);
    for (bindings, 0..) |*b, i| {
        b.* = .{
            .binding = @intCast(i),
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    const ci = VkDescriptorSetLayoutCreateInfo{
        .bindingCount = n_bindings,
        .pBindings = bindings.ptr,
    };
    var layout: VkDescriptorSetLayout = 0;
    if (vkCreateDescriptorSetLayout_fn.?(dev, &ci, null, &layout) != VK_SUCCESS_RC) return 0;
    return layout;
}

fn buildPipelineLayout(layout: VkDescriptorSetLayout, push_size: u32) u64 {
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    var pc_range: VkPushConstantRange = .{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_size,
    };
    const layouts_arr = [_]VkDescriptorSetLayout{layout};
    const has_layout: u32 = if (layout != 0) 1 else 0;
    const ci = VkPipelineLayoutCreateInfo{
        .setLayoutCount = has_layout,
        .pSetLayouts = if (has_layout > 0) layouts_arr[0..1].ptr else null,
        .pushConstantRangeCount = if (push_size > 0) 1 else 0,
        .pPushConstantRanges = if (push_size > 0) @ptrCast(&pc_range) else null,
    };
    var pl: u64 = 0;
    if (vkCreatePipelineLayout_fn.?(dev, &ci, null, &pl) != VK_SUCCESS_RC) return 0;
    return pl;
}

fn buildPipeline(
    spv: []const u8,
    decl: EncodeKernelDecl,
    out_layout: *VkPipelineLayout,
    out_fn: *VkPipeline,
) bool {
    const sm = loadShaderModule(spv);
    if (sm == 0) return false;

    const dset_layout = buildDescriptorSetLayout(decl.n_bindings);
    if (decl.n_bindings > 0 and dset_layout == 0) return false;

    const pl = buildPipelineLayout(dset_layout, decl.push_constant_size);
    if (pl == 0) return false;

    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    // VK adaptation: subgroup size pinned to 32 via REQUIRE_FULL_SUBGROUPS_BIT
    // + VkPipelineShaderStageRequiredSubgroupSizeCreateInfo. Kernels in
    // gpu_warp.glsl hardcode WARP_SIZE=32 (the warpShuffle/Ballot/Any
    // cooperative ops are correct only at exactly 32 lanes). Without this
    // pin, Intel iGPU (supports [8,32]) would silently miscompile while
    // NVIDIA (only supports [32,32]) works by accident.
    const required_size_info = VkPipelineShaderStageRequiredSubgroupSizeCreateInfo{
        .requiredSubgroupSize = 32,
    };
    const stage = VkPipelineShaderStageCreateInfo{
        .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
        .pNext = @ptrCast(&required_size_info),
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = sm,
        .pName = "main",
    };
    const ci = VkComputePipelineCreateInfo{
        .stage = stage,
        .layout = pl,
    };
    var pipeline: u64 = 0;
    if (vkCreateComputePipelines_fn.?(dev, 0, 1, @ptrCast(&ci), null, @ptrCast(&pipeline)) != VK_SUCCESS_RC) return false;

    if (!decode_module_loader.registerExternalPipeline(pipeline, pl, dset_layout, decl.n_bindings, decl.push_constant_size)) {
        return false;
    }

    out_layout.* = @intCast(pl);
    out_fn.* = @intCast(pipeline);
    return true;
}

// ── Encode-side FFI shim implementations ──────────────────────────────
// Each one bridges the encode-shaped vulkan_ffi signatures to the
// decode-side procs table (a single shared implementation). The signatures
// mirror the CUDA cu<Op> entry points byte for byte; the bodies route
// every operation back to vulkan_api.procs.*.

fn encodeModuleLoadData(out_mod: *VkPipelineLayout, spv: [*]const u8) callconv(.c) VkResult {
    // VK adaptation: cuModuleLoadData takes a NUL-terminated PTX text
    // string and no length; SPIR-V is a binary stream so we recover the
    // length by walking the instruction stream (each instruction's high
    // 16 bits hold its word count). The encode codec only ever passes
    // well-formed SPV produced by the project's own glslc compile, so
    // the walk terminates at the OpFunctionEnd that closes the trailing
    // function. We then hand the recovered byte length to
    // vkCreateShaderModule and return the resulting VkShaderModule
    // through the VkPipelineLayout-shaped out slot (both u64 aliases).
    out_mod.* = 0;
    if (vkCreateShaderModule_fn == null) return -1;

    const words: [*]align(1) const u32 = @ptrCast(spv);
    if (words[0] != 0x07230203) return -1;

    // SPIR-V header is 5 words: magic, version, generator, bound, schema.
    // Then instructions back-to-back: word 0 high 16 bits = wordCount.
    const MAX_WORDS: usize = 1 << 22; // 16 MiB ceiling on a single module.
    var i: usize = 5;
    while (i < MAX_WORDS) {
        const wc: usize = @intCast(words[i] >> 16);
        if (wc == 0) return -1; // malformed instruction
        const op: u32 = words[i] & 0xFFFF;
        i += wc;
        // OpFunctionEnd (56) with wordCount=1 closes a function body. The
        // SPIR-V module ends after the final OpFunctionEnd of the entry
        // point's function — there are no global declarations after it
        // in glslc-produced compute modules. Use it as the natural stop.
        if (op == 56) break;
    }
    if (i >= MAX_WORDS) return -1;

    const ci = VkShaderModuleCreateInfo{
        .codeSize = i * @sizeOf(u32),
        .pCode = @ptrCast(@alignCast(spv)),
    };
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    var sm: VkShaderModule = 0;
    const rc = vkCreateShaderModule_fn.?(dev, &ci, null, &sm);
    if (rc != VK_SUCCESS_RC) return rc;
    out_mod.* = @intCast(sm);
    return VK_SUCCESS_RC;
}

fn encodeModuleGetFunction(out_fn: *VkPipeline, mod: VkPipelineLayout, name: [*:0]const u8) callconv(.c) VkResult {
    // The encode-side public *_fn slots are already populated at
    // init() time with the VkPipeline handles built off the prebaked
    // SPV blobs. The CUDA cuModuleGetFunction shape (mod + name → fn)
    // implies a runtime lookup against a module that holds named entry
    // points; the VK side bakes one entry point per pipeline so the
    // mapping is direct. Look up the entry-point name against the
    // known encode kernels and return the matching slot.
    _ = mod; // matched by name, not handle
    const name_slice = std.mem.span(name);
    if (std.mem.eql(u8, name_slice, "slzLzEncodeKernel")) {
        out_fn.* = kernel_fn;
        return if (kernel_fn != 0) VK_SUCCESS_RC else -1;
    }
    if (std.mem.eql(u8, name_slice, "slzHuffBuildTablesKernel")) {
        out_fn.* = huff_tables_kernel_fn;
        return if (huff_tables_kernel_fn != 0) VK_SUCCESS_RC else -1;
    }
    if (std.mem.eql(u8, name_slice, "slzHuffEncode4StreamKernel")) {
        out_fn.* = huff_encode_kernel_fn;
        return if (huff_encode_kernel_fn != 0) VK_SUCCESS_RC else -1;
    }
    if (std.mem.eql(u8, name_slice, "slzAssembleMeasureKernel")) {
        out_fn.* = assemble_measure_fn;
        return if (assemble_measure_fn != 0) VK_SUCCESS_RC else -1;
    }
    if (std.mem.eql(u8, name_slice, "slzAssembleWriteKernel")) {
        out_fn.* = assemble_write_fn;
        return if (assemble_write_fn != 0) VK_SUCCESS_RC else -1;
    }
    if (std.mem.eql(u8, name_slice, "slzFrameAssembleKernel")) {
        out_fn.* = frame_assemble_fn;
        return if (frame_assemble_fn != 0) VK_SUCCESS_RC else -1;
    }
    return -1;
}

fn encodeMemAlloc(out: *VkDeviceBuffer, size: usize) callconv(.c) VkResult {
    const proc = vulkan_api.procs.malloc_device orelse return -1;
    return proc(out, size);
}

fn encodeMemFree(handle: VkDeviceBuffer) callconv(.c) VkResult {
    const proc = vulkan_api.procs.free_device orelse return -1;
    return proc(handle);
}

fn encodeMemcpyHtoD(dst: VkDeviceBuffer, src: *const anyopaque, size: usize) callconv(.c) VkResult {
    const proc = vulkan_api.procs.h2d orelse return -1;
    return proc(dst, src, size);
}

fn encodeMemcpyDtoH(dst: *anyopaque, src: VkDeviceBuffer, size: usize) callconv(.c) VkResult {
    const proc = vulkan_api.procs.d2h orelse return -1;
    return proc(dst, src, size);
}

fn encodeMemcpyDtoDAsync(dst: VkDeviceBuffer, src: VkDeviceBuffer, size: usize, stream: usize) callconv(.c) VkResult {
    const proc = vulkan_api.procs.d2d orelse return -1;
    return proc(dst, src, size, stream);
}

fn encodeLaunchKernel(
    pipeline: VkPipeline,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_bytes: c_uint,
    stream: usize,
    params: [*]?*anyopaque,
    extra: [*]?*anyopaque,
) callconv(.c) VkResult {
    const proc = vulkan_api.procs.launch_kernel orelse return -1;
    return proc(pipeline, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes, stream, params, extra);
}

fn encodeCtxSync() callconv(.c) VkResult {
    const proc = vulkan_api.procs.ctx_sync orelse return -1;
    return proc();
}

fn encodeMemsetD8(dst: VkDeviceBuffer, value: u8, size: usize) callconv(.c) VkResult {
    const proc = vulkan_api.procs.memset_d8 orelse return -1;
    return proc(dst, value, size);
}
