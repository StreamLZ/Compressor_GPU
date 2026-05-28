# CUDA Graph capture for the decode back-half — Phase 4 design doc

Status: **NOT YET IMPLEMENTED**. Scaffolding committed; the actual capture
logic is the work that remains. Live across multiple sessions — read this
end-to-end before resuming. Updated 2026-05-28.

## What we're trying to do

Replace the per-decompress sequence of ~12 individual `cuLaunchKernel` /
`cuMemcpyDtoDAsync` calls in `fullGpuLaunchImpl`'s back-half with a single
`cuGraphLaunch` of a pre-captured + pre-instantiated CUDA Graph. Target:
the measured 1.26–1.75 ms `stream-idle gap` reported by
`tools/slz_gpu_d2d_bench` between the cudaEvent-bracketed total ("gpu
kernel best") and the kernel-active sum.

Realistic ceiling — see "expected wins" near the end. Don't expect the
entire gap to disappear; some of it is event-record overhead and host
code that lives outside the capturable region.

## Where things stand right now

Three commits in place. Read them in order before touching anything.

| Commit  | What                                                                |
|---------|---------------------------------------------------------------------|
| e241b5a | CUDA Driver Graph API bindings in `cuda_api.zig` + `module_loader.zig` |
| b520b38 | `graph_exec` / `graph_captured` fields on `DecodeContext`; placeholder comment at the would-be capture site in `decode_dispatch.zig`; cleanup in `deinit` |

These are *correct and useful*. Don't revert them. They're the
foundation the real implementation builds on.

No behavior changes have shipped yet — the back-half still runs as
individual `cuLaunchKernel` calls.

## The trap I fell into (don't repeat it)

First attempt was the obvious "wrap the back-half in `cuStreamBeginCapture`
/ `cuStreamEndCapture` / `cuGraphInstantiate` / `cuGraphLaunch`, cache
the graph_exec, replay on next call." Compiled, ran, segfaulted on the
first replay.

**Root cause:** `cuLaunchKernel` during stream capture stores the
**addresses** of the `kernelParams` array entries into the graph node,
not the values they point at. Our current kernel-launch sites all look
like:

```zig
fn runLzPipeline(...) ... {
    ...
    {
        var p_comp: u64 = self.d_comp_persist;        // <- stack local
        var p_dst:  u64 = req.d_output_target.?;       // <- stack local
        var p_chunks: u64 = self.d_descs_persist;      // <- stack local
        ...
        var params = [_]?*anyopaque{
            @ptrCast(&p_comp), @ptrCast(&p_dst), @ptrCast(&p_chunks), ...
        };                                              // <- stack-local array of pointers to stack locals
        var extra = [_]?*anyopaque{null};
        cuLaunchKernel(..., &params, &extra);
    }
}
```

During capture, the graph stores `&p_comp`, `&p_dst`, `&p_chunks`, etc.
When `runLzPipeline` returns, those stack frames are popped. The graph's
captured pointers now reference invalidated memory. Subsequent
`cuGraphLaunch` reads garbage → segfault.

**This affects every kernel launch site in the back-half:**
- `runHuffPredecode` (decode_dispatch.zig:453) — two kernels (build_lut, decode)
- `runLzPipeline` (decode_dispatch.zig:530) — one kernel per pipeline group
- `mergeHuffDescs` (decode_dispatch.zig:92) — merge_huff kernel
- `gatherRawOff16` (decode_dispatch.zig:182) — gather_off16 kernel
- `finalizeOutput` (decode_dispatch.zig:652) — cuMemcpyDtoDAsync (also stack-local args)

Every one of these has stack-local `var p_*: u64 = ...; var params = [...]&p_*...;` patterns.

## The fix: persistent param storage

For every kernel that gets captured, its kernel-param array AND every
value the array points at must live in memory that **outlives the
capture**. The natural home is `DecodeContext` — it owns everything
else cross-call.

### Param struct layout

For each captured kernel, define an `extern struct` of the param fields,
plus the `params: [N]?*anyopaque` array of pointers into those fields.
Example for the LZ kernel:

```zig
// In decode_context.zig or a new gpu/decode/graph_params.zig
pub const LzKernelParams = extern struct {
    p_comp:        u64 = 0,
    p_chunks:      u64 = 0,
    p_dst:         u64 = 0,
    p_cpg:         u32 = 0,
    p_total:       u64 = 0,  // device pointer to *d_total
    p_sc_cap:      u32 = 0,
    p_entropy:     u64 = 0,
    p_stride:      u64 = 0,
    p_first_sub:   u64 = 0,

    // params pointer array — pointer into the fields above. Populated
    // once at first-capture, doesn't change across calls.
    params: [9]?*anyopaque = .{null} ** 9,
    extra:  [1]?*anyopaque = .{null},

    pub fn bind(self: *LzKernelParams) void {
        self.params = .{
            @ptrCast(&self.p_comp),    @ptrCast(&self.p_chunks),
            @ptrCast(&self.p_dst),     @ptrCast(&self.p_cpg),
            @ptrCast(&self.p_total),   @ptrCast(&self.p_sc_cap),
            @ptrCast(&self.p_entropy), @ptrCast(&self.p_stride),
            @ptrCast(&self.p_first_sub),
        };
    }
};
```

One of these per captured kernel:
- `MergeHuffParams`, `HuffBuildParams`, `HuffDecodeParams`, `GatherOff16Params`,
  `LzDecodeParams` (raw + general — separate `kernel_fn` vs `kernel_raw_fn`)
- For `finalizeOutput`'s `cuMemcpyDtoDAsync` (the output copy at the end),
  see "memcpy nodes" below — that one needs a different shape.

Recommend: collect these into a single `BackHalfGraphParams` struct that
owns one of each, hung off `DecodeContext`.

```zig
pub const BackHalfGraphParams = struct {
    merge:     MergeHuffParams = .{},
    huff_build:HuffBuildParams = .{},
    huff_dec:  HuffDecodeParams = .{},
    gather:    GatherOff16Params = .{},
    lz:        LzDecodeParams = .{},
    out_copy:  OutputCopyParams = .{},
    bound:     bool = false,  // .bind() called yet?
};
```

Add `graph_params: BackHalfGraphParams = .{}` to `DecodeContext`.

### Memcpy nodes

`finalizeOutput`'s `cuMemcpyDtoDAsync` is also captured. The captured
node is a CU_GRAPH_MEMCPY_NODE. Updates use
`cuGraphExecMemcpyNodeSetParams` (not `cuGraphExecKernelNodeSetParams`).
You'll need to add a binding for it — it's not in `cuda_api.zig` yet.

The memcpy node stores src/dst pointers and a byte count. Updating it
to point at a new `d_output_target` per call requires building a
`CUDA_MEMCPY3D` (or `CUDA_MEMCPY2D` for 2D) descriptor and passing it
to `cuGraphExecMemcpyNodeSetParams`. The 3D variant is the safe one for
device→device linear copies. See CUDA Driver API docs.

## The actual implementation, step by step

### Step 1 — Add params structs and persistent storage (~2 hours, low risk)

1. Create `src/gpu/decode/graph_params.zig` with one `extern struct` per
   captured kernel + a `BackHalfGraphParams` aggregator.
2. Each has a `bind()` method that wires up the `params: [N]?*anyopaque`
   array to point at the struct's own fields. Called once after the
   struct is in its final memory location.
3. Add `graph_params: BackHalfGraphParams` to `DecodeContext`.
4. **Verify**: build cleanly. Run `bench_all.bat` + `bench_d2d.bat` — no
   behavior change yet, just new unused fields.
5. **Commit.**

### Step 2 — Migrate every captured-region kernel launch to use the persistent params (~3 hours, medium risk)

Touch points (every call site in the back-half):
- `runHuffPredecode` — two `cuLaunchKernel` calls. Replace stack-local `var p_*` + `params` array with reads/writes against `self.graph_params.huff_build` and `self.graph_params.huff_dec`.
- `runLzPipeline` — one `cuLaunchKernel`. Move to `self.graph_params.lz`.
- `mergeHuffDescs` — one `cuLaunchKernel`. Move to `self.graph_params.merge`.
- `gatherRawOff16` — one `cuLaunchKernel`. Move to `self.graph_params.gather`.
- `finalizeOutput` — one `cuMemcpyDtoDAsync`. Move to `self.graph_params.out_copy` — see "memcpy nodes" above.

For each: set the persistent fields to the current call's values, then
launch with `&self.graph_params.{name}.params` and `&self.graph_params.{name}.extra`.

Critically: do NOT yet wrap in `cuStreamBeginCapture`. This step just
proves the persistent-param machinery is correct under direct launches.
Output must still be byte-exact.

**Verify**: `tools/bench_all.bat` SHA-OK on all 10 cells.
`tools/bench_d2d.bat` SHA-OK on all 10 cells. Decode times should be
within noise of current baselines.

**Commit.**

### Step 3 — Wrap the back-half in cuStreamBeginCapture/EndCapture (~3 hours, high risk)

In `fullGpuLaunchImpl` at the placeholder comment (decode_dispatch.zig
around line 897, just after the front-half `stream_sync`):

1. Resolve all the graph-API function pointers; check they're non-null.
   If `cuStreamBeginCapture_fn` is null, fall through to the existing
   direct-launch path.
2. Decide if this call should use a graph at all:
   - `SLZ_GPU_GRAPHS=0` env var → disable for A/B comparison.
   - `split_timer` set → disable (the inner stream-syncs break capture).
   - Otherwise → enable.
3. **Cache hit check**: compare current shape vs `self.graph_shape_key`
   (see Step 4 for the key). If match AND `self.graph_exec != 0`:
   - Update kernel-node params for ONLY the fields that change per call
     (see Step 4).
   - `cuGraphLaunch(self.graph_exec, heavy_stream)`.
   - Skip the rest of the back-half code.
4. **Cache miss** (first call, or shape changed):
   - Destroy any existing `self.graph_exec` and `self.graph_captured`.
   - `cuStreamBeginCapture(heavy_stream, CU_STREAM_CAPTURE_MODE_GLOBAL)`.
   - Run the existing back-half code (`runHuffPredecode`, `runLzPipeline`, `finalizeOutput`). Because params are now persistent, the captured node parameters will be valid at replay time.
   - `cuStreamEndCapture(heavy_stream, &graph)`.
   - `cuGraphInstantiate(&graph_exec, graph, &error_node, &log, log.len)`.
   - Walk graph nodes via `cuGraphGetNodes` and stash the node handles for each kernel/memcpy of interest into `DecodeContext` (e.g. `self.graph_nodes.lz`, `self.graph_nodes.huff_build`, etc.). These are needed for per-call param updates in Step 4.
   - Store new `graph` and `graph_exec` in `self.graph_captured` / `self.graph_exec`.
   - Store the shape key.
   - `cuGraphLaunch(self.graph_exec, heavy_stream)`.

Note: the work executes during `cuGraphLaunch`, NOT during the
`runHuff/runLz/finalize` calls when in capture mode. The CPU side runs
those functions to BUILD the graph; nothing actually computes on the
GPU until the launch.

**Verify**: `tools/bench_d2d.bat` SHA-OK on all 10 cells. If decode
times are now BETTER than the current baseline, the cache is working.
If they're WORSE or equal, the per-call instantiate cost is dominating —
check that the shape cache is actually hitting.

**Commit.**

### Step 4 — Shape key + per-call param updates (~2 hours, medium risk)

Define what shapes are equivalent (can share a graph):

```zig
pub const GraphShapeKey = struct {
    n_chunks: u32,
    n_huff: u32,
    have_huff: bool,
    sub_chunk_cap: u32,
    chunks_per_group: u32,
    total_subchunks: u32,
    // Hash these into a single u64 for fast comparison.
};
```

Shapes match when all fields are equal. Pointers (`d_frame`,
`d_output_target`, `d_descs_persist`, etc.) and counts that change per
call but DON'T affect grid shape (the kernel grid sizes are derived
from `n_chunks` / `n_huff`, which ARE in the key) get updated via
`cuGraphExecKernelNodeSetParams` between same-shape calls.

For each captured kernel, on a same-shape replay, you write the new
values into the persistent params struct AND call
`cuGraphExecKernelNodeSetParams(self.graph_exec, self.graph_nodes.lz, &kernel_node_params)`
where `kernel_node_params` is a freshly-built `CudaKernelNodeParams`
referencing the (updated) persistent fields.

For the memcpy node (finalize output), use `cuGraphExecMemcpyNodeSetParams`
with a fresh `CUDA_MEMCPY3D` struct.

**Verify**: same as Step 3 plus run a stress test that decodes alternating
frames of different sizes (forcing shape-key mismatches and re-captures)
and confirm both paths are byte-exact.

**Commit.**

### Step 5 — Measure and tune (~1 hour)

Run `tools/bench_d2d.bat` with and without `SLZ_GPU_GRAPHS=1`. The
`gap` column in the summary should shrink. Realistic targets:

- Today's gap: 1.26–1.75 ms across cells.
- After Phase 4: probably 0.7–1.0 ms across cells (gap shrinks ~30–50%).
- Wall-clock decode: should drop ~300–500 µs (~5–7% of D2D total).

If the gap is unchanged or worse, the most likely culprits:
- Cache thrashing: the shape key is too restrictive, every call
  invalidates. Loosen if safe.
- Param-update cost > launch cost: `cuGraphExecKernelNodeSetParams`
  per kernel might be expensive enough to wash out the savings. If so,
  consider only updating the params that actually changed.
- Some operation inside the back-half is non-capturable and CUDA falls
  back to per-launch silently. Add error checking around the
  `cuStreamEndCapture` return value.

**Commit final numbers + update `src/gpu/README.md` perf table.**

## Files you will be touching

- `src/gpu/decode/decode_context.zig` — add `graph_params`, `graph_nodes`, `graph_shape_key` fields. Already has `graph_exec` and `graph_captured`.
- `src/gpu/decode/decode_dispatch.zig` — the placeholder block around line 897. Plus migration of kernel launch sites at lines 453 (`runHuffPredecode`), 530 (`runLzPipeline`), 92 (`mergeHuffDescs`), 182 (`gatherRawOff16`), 652 (`finalizeOutput`).
- `src/gpu/decode/cuda_api.zig` — add `cuGraphExecMemcpyNodeSetParams` binding (the memcpy-node update function). Already has the rest.
- `src/gpu/decode/module_loader.zig` — wire up the new binding.
- New file: `src/gpu/decode/graph_params.zig` — the persistent param structs (one per captured kernel + the aggregator).

## Verification commands

After each step:

```
zig build -Doptimize=ReleaseFast -Dgpu=true
zig build gpulib -Doptimize=ReleaseFast
tools/build_d2d_bench.bat
tools/bench_all.bat       # CLI path — should be unaffected (uses CPU scan)
tools/bench_d2d.bat        # the path Phase 4 actually changes
```

`bench_d2d` checks SHA-OK on enwik8 L1-L5 + silesia 128MB L1-L5. Any
FAIL = revert and figure out why before continuing.

## Decisions to make as you go

1. **Capture region boundaries**: should `mergeHuffDescs` and
   `gatherRawOff16` be inside the captured region, or do they belong
   with the front-half (which runs BEFORE the `stream_sync(work_stream)`
   at line 888)? They currently run between the front-half compact and
   the back-half Huff predecode. Either choice works; just be
   consistent.

2. **Default-on vs opt-in**: should `SLZ_GPU_GRAPHS` be required at
   runtime, or should graphs be on by default with a "kill switch"
   env var? I lean toward default-on once the implementation is solid,
   but ship it opt-in for the first commit so it can be A/B tested.

3. **Profiling compatibility**: `beginKernelTiming` records cuEvent
   pairs that get captured into the graph. Per docs this is fine but
   profile-mode output may differ subtly (events are recorded
   against graph node positions, not original launches). Sanity-check
   the per-kernel breakdown in `tools/slz_gpu_d2d_bench` still
   sums to roughly `kernel active best`.

4. **NUM_PIPELINE_STREAMS > 1**: today this constant is 1, so the
   pipeline only uses one stream. If a future revision bumps it,
   cross-stream synchronization inside a captured region needs
   `cuEventRecord` + `cuStreamWaitEvent` nodes in the graph. The
   current design doesn't handle this; document as out-of-scope or
   add explicit guard.

5. **Falling back when shape changes too often**: if a workload
   uses lots of different shapes, the constant re-capture +
   re-instantiate will be slower than direct launches. Consider a
   counter — if N consecutive cache-misses, fall back to direct
   launches for the rest of the call.

## Expected wins (be honest with yourself)

- Per-call save (cache hit): ~300–500 µs from replacing 12 ×
  `cuLaunchKernel` with 1 × `cuGraphLaunch` + ~12 ×
  `cuGraphExec{Kernel,Memcpy}NodeSetParams`.
- First call on a new shape: ~500 µs OVERHEAD from `cuGraphInstantiate`.
  This amortizes across same-shape replays.
- Wall-clock D2D: 5–10% improvement on workloads that decompress many
  same-shape frames (game asset streaming, LLM batch decode). 0% or
  worse on one-shot decompresses.

Not a transformational win. Don't oversell it. Worth doing for the
target workloads, not worth doing if your only use case is one-shot.

## Open question: is Phase 4 worth doing AT ALL?

Honest gut check: 5–10% wall-clock on the D2D path is modest. If the
caller can already overlap their decompress with their other GPU work
on the same stream (which they can today, after the v3 API ships and
Phase 2 had front-half kernels using `work_stream`), the marginal
benefit is small.

The bigger user-visible wins are probably:
- Phase 3-final (eliminate walkMetaToHost CPU block — ~0.66 ms = ~10%
  wall-clock).
- A `slzDecompressBatchedAsync` that takes N frames at once and submits
  them all to the stream in one call (matches the nvCOMP batched
  pattern; would actually amortize the front-half walk overhead too).

Re-evaluate before committing days to Phase 4. The numbers may say
that Phase 3-final + batched API is a bigger win than graphs alone.
