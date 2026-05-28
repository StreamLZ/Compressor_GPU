//! Persistent per-kernel parameter storage for the back-half CUDA Graph
//! capture (Phase 4). Every kernel that is going to live inside the captured
//! graph needs the storage backing its `kernelParams` array to outlive the
//! capture - `cuLaunchKernel` records the *addresses* of each `?*anyopaque`
//! slot, not the values, so stack-local `var p_x: u64 = ...` patterns
//! invalidate the moment the launching function returns and replays segfault.
//!
//! Each `*Params` struct here bundles the value fields one kernel needs plus
//! a `params: [N]?*anyopaque` array. `bind()` is called once after the struct
//! reaches its final memory location (on `DecodeContext`) and points each
//! `params[i]` slot at the matching field inside `self`. After that, each
//! per-decode launch just writes new values into the fields - the addresses
//! the graph captured stay valid forever.
//!
//! See `src/gpu/CudaGraph.md` for the design rationale; in particular, the
//! "trap I fell into" section explains the segfault this storage avoids.
//!
//! Step 1 of the Phase 4 roadmap: the structs exist and are wired onto
//! `DecodeContext`, but no launch site uses them yet. Step 2 migrates the
//! kernel launches to read/write these fields; Step 3 wraps the region in
//! `cuStreamBeginCapture` / `cuStreamEndCapture`.

const cuda = @import("cuda_api.zig");

const CUdeviceptr = cuda.CUdeviceptr;

/// `slzMergeHuffDescsKernel` - 12 parameters. Inputs: four per-stream
/// HuffDecChunkDesc arrays + their counts, the tok/off16 region offsets,
/// and the merged-output buffer + its written-count slot.
pub const MergeHuffParams = struct {
    p_lit:         CUdeviceptr = 0,
    p_tok:         CUdeviceptr = 0,
    p_hi:          CUdeviceptr = 0,
    p_lo:          CUdeviceptr = 0,
    p_n_lit:       CUdeviceptr = 0,
    p_n_tok:       CUdeviceptr = 0,
    p_n_hi:        CUdeviceptr = 0,
    p_n_lo:        CUdeviceptr = 0,
    p_tok_region:  u32 = 0,
    p_off16_region:u32 = 0,
    p_merged:      CUdeviceptr = 0,
    p_n_merged:    CUdeviceptr = 0,

    params: [12]?*anyopaque = .{null} ** 12,
    extra:  [1]?*anyopaque = .{null},

    pub fn bind(self: *MergeHuffParams) void {
        self.params = .{
            @ptrCast(&self.p_lit),         @ptrCast(&self.p_tok),
            @ptrCast(&self.p_hi),          @ptrCast(&self.p_lo),
            @ptrCast(&self.p_n_lit),       @ptrCast(&self.p_n_tok),
            @ptrCast(&self.p_n_hi),        @ptrCast(&self.p_n_lo),
            @ptrCast(&self.p_tok_region),  @ptrCast(&self.p_off16_region),
            @ptrCast(&self.p_merged),      @ptrCast(&self.p_n_merged),
        };
    }
};

/// `slzGatherRawOff16Kernel` - 5 parameters.
pub const GatherOff16Params = struct {
    p_comp:     CUdeviceptr = 0,
    p_comp_len: u32 = 0,
    p_scratch:  CUdeviceptr = 0,
    p_descs:    CUdeviceptr = 0,
    p_count:    CUdeviceptr = 0,

    params: [5]?*anyopaque = .{null} ** 5,
    extra:  [1]?*anyopaque = .{null},

    pub fn bind(self: *GatherOff16Params) void {
        self.params = .{
            @ptrCast(&self.p_comp), @ptrCast(&self.p_comp_len), @ptrCast(&self.p_scratch),
            @ptrCast(&self.p_descs), @ptrCast(&self.p_count),
        };
    }
};

/// `slzHuffBuildLutKernel` - 4 parameters.
pub const HuffBuildParams = struct {
    p_comp:  CUdeviceptr = 0,
    p_descs: CUdeviceptr = 0,
    p_lut:   CUdeviceptr = 0,
    p_n:     CUdeviceptr = 0,

    params: [4]?*anyopaque = .{null} ** 4,
    extra:  [1]?*anyopaque = .{null},

    pub fn bind(self: *HuffBuildParams) void {
        self.params = .{
            @ptrCast(&self.p_comp), @ptrCast(&self.p_descs),
            @ptrCast(&self.p_lut),  @ptrCast(&self.p_n),
        };
    }
};

/// `slzHuffDecode4StreamKernel` - 5 parameters.
pub const HuffDecodeParams = struct {
    p_comp:  CUdeviceptr = 0,
    p_descs: CUdeviceptr = 0,
    p_lut:   CUdeviceptr = 0,
    p_out:   CUdeviceptr = 0,
    p_n:     CUdeviceptr = 0,

    params: [5]?*anyopaque = .{null} ** 5,
    extra:  [1]?*anyopaque = .{null},

    pub fn bind(self: *HuffDecodeParams) void {
        self.params = .{
            @ptrCast(&self.p_comp), @ptrCast(&self.p_descs),
            @ptrCast(&self.p_lut),  @ptrCast(&self.p_out),
            @ptrCast(&self.p_n),
        };
    }
};

/// `slzLzDecodeRawKernel` - 6 parameters (L1/L2 fast path, no entropy).
pub const LzRawParams = struct {
    p_comp:      CUdeviceptr = 0,
    p_descs_dev: CUdeviceptr = 0,
    p_dst:       CUdeviceptr = 0,
    p_cpg:       u32 = 0,
    p_total:     CUdeviceptr = 0,
    p_sc_cap:    u32 = 0,

    params: [6]?*anyopaque = .{null} ** 6,
    extra:  [1]?*anyopaque = .{null},

    pub fn bind(self: *LzRawParams) void {
        self.params = .{
            @ptrCast(&self.p_comp),  @ptrCast(&self.p_descs_dev),
            @ptrCast(&self.p_dst),   @ptrCast(&self.p_cpg),
            @ptrCast(&self.p_total), @ptrCast(&self.p_sc_cap),
        };
    }
};

/// `slzLzDecodeKernel` - 9 parameters (general path, reads entropy_scratch).
pub const LzGeneralParams = struct {
    p_comp:               CUdeviceptr = 0,
    p_descs_dev:          CUdeviceptr = 0,
    p_dst:                CUdeviceptr = 0,
    p_cpg:                u32 = 0,
    p_total:              CUdeviceptr = 0,
    p_sc_cap:             u32 = 0,
    p_entropy_scratch:    CUdeviceptr = 0,
    p_entropy_slot_stride:u64 = 0,
    p_first_sub_idx:      CUdeviceptr = 0,

    params: [9]?*anyopaque = .{null} ** 9,
    extra:  [1]?*anyopaque = .{null},

    pub fn bind(self: *LzGeneralParams) void {
        self.params = .{
            @ptrCast(&self.p_comp),               @ptrCast(&self.p_descs_dev),
            @ptrCast(&self.p_dst),                @ptrCast(&self.p_cpg),
            @ptrCast(&self.p_total),              @ptrCast(&self.p_sc_cap),
            @ptrCast(&self.p_entropy_scratch),    @ptrCast(&self.p_entropy_slot_stride),
            @ptrCast(&self.p_first_sub_idx),
        };
    }
};

/// `finalizeOutput`'s D2D async memcpy (when caller passed a device target).
/// Captured as a CU_GRAPH_MEMCPY_NODE; updates use
/// `cuGraphExecMemcpyNodeSetParams` (different shape than kernel-node
/// updates). For Step 1 we just hold the fields; Step 3 builds the
/// CUDA_MEMCPY3D descriptor at capture time and Step 4 updates it per call.
pub const OutputCopyParams = struct {
    p_dst:  CUdeviceptr = 0,
    p_src:  CUdeviceptr = 0,
    p_size: usize = 0,
};

/// Aggregates one param struct per captured kernel + a `bound` flag so we
/// can lazily call each `bind()` once when the owning `DecodeContext`
/// reaches its final memory location. Hung off `DecodeContext` (see
/// `decode_context.zig`).
pub const BackHalfGraphParams = struct {
    merge:      MergeHuffParams = .{},
    gather:     GatherOff16Params = .{},
    huff_build: HuffBuildParams = .{},
    huff_dec:   HuffDecodeParams = .{},
    lz_raw:     LzRawParams = .{},
    lz_gen:     LzGeneralParams = .{},
    out_copy:   OutputCopyParams = .{},
    bound:      bool = false,

    /// Wire up every nested params struct's `params[i]` array to point at
    /// its own value fields. Must be called AFTER `self` is in its final
    /// memory home (i.e. inside the DecodeContext on the heap / facade
    /// singleton) and BEFORE any captured launch reads `self.merge.params`,
    /// etc. Idempotent.
    pub fn bindAll(self: *BackHalfGraphParams) void {
        if (self.bound) return;
        self.merge.bind();
        self.gather.bind();
        self.huff_build.bind();
        self.huff_dec.bind();
        self.lz_raw.bind();
        self.lz_gen.bind();
        self.bound = true;
    }
};
