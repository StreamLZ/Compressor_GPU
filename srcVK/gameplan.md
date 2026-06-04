# /srcVK/ Port Gameplan

**Purpose:** Source of truth for the post-compact agent (or any future agent) picking up the CUDA → Vulkan 1:1 port living under `/srcVK/`. The user has been burned multiple times by prior failures (see Mistakes section); read this whole file before doing anything.

**User context:** The user has been crystal clear that the goal is a **1:1 port** of `/src/` (CUDA) to Vulkan. Not a rewrite. Not "behaviorally equivalent." A code-level translation where every CUDA file, function, struct, field, control flow, and call order has a srcVK counterpart with the same name (modulo `cu*`/`cuda*`/`nvidia*` token swaps to `vk*`/`vulkan*`).

---

## 1. Where We Are (current commit HEAD)

Project root: `c:/Users/james.JAMESWORK2025/Repos/Compressor_GPU`

### Phases complete

| Phase | Scope | Status |
|---|---|---|
| Audit | `/srcVK/audit.md` — 73 files mapped, extension/token rules canonical | DONE |
| Step 2 stub-layout | 79 skeleton files + `/srcVK/PortInstructions.md` + build.zig wiring (new `streamlz_vk` step → srcVK; old renamed to `streamlz_vk_old`) | DONE |
| Phase 1 fleshout | 10 foundation files (error.zig, version.zig, mmap.zig, format/*, common/*.glsl) + fix-up wave for missed L1 helpers (subchunkIsLz/Mode/CompSize, read8safe, warpCopy) | DONE — clean |
| Phase 2 fleshout | 4 driver-layer files (vma.zig, decode/vulkan_api.zig, decode/descriptors.zig, encode/vulkan_ffi.zig) + fix-up adding 4 missing procs slots (ctx_set/get_current, h2d_async, d2h_async) | DONE — clean |
| Phase 3+5 combined | 2 module_loader.zig (decode + encode) + 11 L2 stub .comp + 2 .glsl headers + 2 fix-ups (real vkCreateComputePipelines, real procs.launch_kernel, correct workgroup sizes, real encodeModuleLoadData) | DONE — clean |

**Build state right now:** `zig build streamlz_vk -Doptimize=ReleaseFast` succeeds, exit 0. Produces `zig-out/bin/streamlz_vk.exe`. Runtime would return `error.NotYetPorted` or `error.NotImplementedL2` on any call because the L1 host orchestration (Phases 4, 6, 7, 10, 11) is still skeleton.

`zig build streamlz_vk_old -Doptimize=ReleaseFast` also succeeds — produces `streamlz_vk_old.exe` from `/src_vulkan/` (the legacy reimplementation). This keeps existing tooling working while the port progresses.

### Files state (count by directory under /srcVK/)

- `srcVK/` root: 10 files (main.zig, cli.zig, mmap.zig, error.zig, version.zig, vma.zig, streamlz_gpu.zig, test_runner_parallel.zig, audit.md, PortInstructions.md, gameplan.md (this file))
- `srcVK/cli/`: 7 files (compress, decompress, bench_compress, bench_decompress, bench_all, info, util)
- `srcVK/format/`: 3 files (frame_format, block_header, streamlz_constants)
- `srcVK/common/`: 4 files (gpu_wire_format.glsl, gpu_byteio.glsl, gpu_warp.glsl, gpu_huffman.glsl)
- `srcVK/decode/`: 25 files (8 .zig host + 6 .glsl headers + 11 .comp kernels)
- `srcVK/encode/`: 21 files (11 .zig host + 4 .glsl headers + 6 .comp kernels)
- `srcVK/tests/`: 8 test files (Exception 3 — no CUDA counterpart)
- `srcVK/vma/`: 2 files (vk_mem_alloc.h + vk_mem_alloc_impl.cpp — VMA self-contained)

Total ~80 files. Audit Section D has the canonical inventory.

### Verified state

Phase 3+5 verifier rounds confirmed:
- `decode/module_loader.zig`: real `vkCreateComputePipelines` (line 963), real `procs.launch_kernel` (`vkCmdBindPipeline` + `vkCmdBindDescriptorSets` + `vkCmdDispatch`), real `procs.stream_sync` (vkWaitForFences), real `procs.free_host` (gpa.free)
- `encode/module_loader.zig`: real `vkCreateComputePipelines`, real `encodeModuleLoadData` (parses SPV instruction stream, calls vkCreateShaderModule), real `encodeModuleGetFunction`
- All 11 decode .comp workgroup sizes match CUDA launch dims verbatim:
  - `walk_frame`, `prefix_sum_chunks`, `compact_huff`, `compact_raw`, `merge_huff`: `local_size_x=1`
  - `scan_parse`, `gather_off16`: `local_size_x=256`
  - `lz_decode`, `lz_decode_raw`: `local_size_x=32, local_size_y=2`
  - `frame_assemble`: `local_size_x=128`
- Grep across all touched files for forbidden rationalization phrases: **zero hits**

---

## 2. What's Left (remaining phases per audit Section E)

| Phase | Scope | Files |
|---|---|---|
| 4 | L1 decode kernels (real GLSL ports) | 6: slz_wire_format.glsl, lz_decode_core.glsl, lz_decode_raw.glsl, lz_dispatch.glsl, lz_decode_raw_kernel.comp, prefix_sum_chunks_kernel.comp |
| 6 | Decode context + scan_gpu | 2: decode/decode_context.zig, decode/scan_gpu.zig |
| 7 | Decode dispatch orchestration (L2 gate) | 2: decode/decode_dispatch.zig, decode/driver.zig |
| 8 | L1 encode kernels (real GLSL ports) | 7: lz_format.glsl, lz_token_emit.glsl, lz_greedy_parser.glsl, lz_encode_kernel.comp, assemble_measure_kernel.comp, assemble_write_kernel.comp, frame_assemble_kernel.comp |
| 9 | L2 stub encode kernels | 3: lz_chain_parser.glsl, huff_build_tables_kernel.comp, huff_encode_4stream_kernel.comp |
| 10 | Encode context + LZ + assemble host orchestration | 5: encode/encode_context.zig, levels.zig, encode_lz.zig, encode_assemble.zig, encode_huff.zig (stub bodies) |
| 11 | Top-level encoder + decoder + C ABI | 5: encode/fast_framed.zig, streamlz_encoder.zig, encode/driver.zig, decode/streamlz_decoder.zig, streamlz_gpu.zig |
| 12 | CLI surface | 9: 7 cli/*.zig files + cli.zig + main.zig |
| 13 | Tests (Exception 3) | 8 test files + test runner |

After Phase 13: **Step 5 — get it working** (allowed to consult `/src_vulkan/` as a troubleshooting reference at this point ONLY).

### Done bar (all three required, NOT before)

1. **Functional:** byte-identical round-trip vs CUDA on enwik8 (95 MB) + silesia (~200 MB) + web.txt
2. **Structural:** 1:1 pairing audit — every srcVK file pairs to a CUDA file by name, every function pairs
3. **Performance:** NVIDIA RTX 4060 Ti e2e ≤1.10× CUDA on enwik8 AND silesia (apples:apples — see Perf section)

---

## 3. HARD RULES (read every one)

These are the rules I keep failing on. Burn them in.

### Naming verbatim
- File names match CUDA `src/` paths exactly, with these extension translations:
  - `.cu` (kernel) → `.comp`
  - `.cu` (host code — not in this codebase) → `.zig`
  - `.cuh` (device header included into .cu) → `.glsl`
  - `.cuh` (host header) → `.zig`
  - `.zig` → `.zig`
  - `.h` (C ABI in `/include/`) → `.h` (unchanged location)
- Function names, struct names, variable names, constant names: **match CUDA verbatim**, modulo token swaps.
- File/identifier token swaps applied to: `cu`, `Cu`, `CU`, `cuda`, `Cuda`, `CUDA`, `nvidia`, `Nvidia`, `NVIDIA` → Vulkan equivalents (`vk`/`Vk`/`VK` or `vulkan`/`Vulkan`/`VULKAN`).
  - Example: `cuda_api.zig` → `vulkan_api.zig`; `cuda_ffi.zig` → `vulkan_ffi.zig`; `cuStream` → `VkStream`; `CUdeviceptr` → `VkDeviceBuffer`; `cuMemAlloc_fn` → procs-struct entry (see Section B of audit.md).
- **NO** `.cu`, `.cuh`, `.ptx`, `.cubin` extensions in `/srcVK/`. **EVER.**
- **NO** `cu`/`cuda`/`nvidia` tokens in `/srcVK/` filenames. **EVER.** (Doc-comment references to "CUDA" are fine — `/// CUDA reference: src/...:LL`. Identifiers are not.)

### Structure verbatim
- File-by-file mirror of `/src/` folder structure. `src/decode/decode_dispatch.zig` → `srcVK/decode/decode_dispatch.zig`. Period.
- Function call order: preserved verbatim. If CUDA's `decompressFrameInner` calls A then B then C, your port calls A then B then C.
- Control flow (if/while/for/switch): preserved.
- Variable declarations: preserved where Zig types allow.

### procs.* surface
- The codec calls `procs.h2d(dst, src, size)`. CUDA and VK both. Same name, same arg list, same arg order, same semantics.
- Vulkan plumbing (vkCmdCopyBuffer, VMA, fences, queues, command buffers) hides UNDER the procs.* wrapper.
- procs members are listed in audit.md Section C.5.1 and Phase 2 fix-up landed all 18: `malloc_device, free_device, h2d, d2h, h2d_async, d2h_async, d2d, memset_d8, memset_d8_async, malloc_host, free_host, stream_sync, ctx_sync, ctx_set_current, ctx_get_current, stream_create, stream_destroy, launch_kernel`.

### L1 vs L2 stubs

**L1 scope** files require FULL ports — real bodies, real implementations.

L1 scope = level 1 and level 2 (LZ-only, no Huffman, no chain parser). Per the audit:
- Encode L1: streamlz_encoder, driver, vulkan_ffi, module_loader, encode_context, encode_lz, encode_assemble, fast_framed, levels, lz_encode_kernel.comp, assemble_*.comp, frame_assemble_kernel.comp, lz_format.glsl, lz_token_emit.glsl, lz_greedy_parser.glsl
- Decode L1: vulkan_api, descriptors, decode_context, scan_gpu (partial), decode_dispatch (gate inside), driver, streamlz_decoder, prefix_sum_chunks_kernel.comp, lz_decode_raw_kernel.comp, slz_wire_format.glsl, lz_decode_core.glsl, lz_decode_raw.glsl, lz_dispatch.glsl

**L2 stubs** have function declarations matching CUDA signatures but bodies that return `error.NotImplementedL2` (Zig) or empty `void main() { return; }` (GLSL, with SSBO bindings + push_constants declared from CUDA kernel arg list).

L2 stub files: encode_huff.zig, huff_build_tables_kernel.comp, huff_encode_4stream_kernel.comp, lz_chain_parser.glsl, lz_decode_kernel.comp (general), gather_raw_off16_kernel.comp, walk_frame_kernel.comp, compact_*_kernel.comp, merge_huff_descs_kernel.comp, scan_parse_kernel.comp, huff_build_lut_kernel.comp, huff_decode_4stream_kernel.comp, lz_decode_general.glsl, lz_header_parse.glsl, gpu_huffman.glsl (body — constants ARE ported).

### L2 gate (Exception 2)
- CUDA dispatches L2 kernels unconditionally on every decode (no-ops on L1). VK port MUST gate them.
- Gate position: as early as possible in decode_dispatch.zig. Skip both allocation AND dispatch on L1.
- Wrap with `if (level >= 2) { ... }`. Documented inline comment: `// VK adaptation: gated on level>=2 to skip L2 work on L1 frames. CUDA currently dispatches unconditionally (no-op on L1); upstream TODO to backport this gate.`

### FORBIDDEN rationalization phrases

These exact patterns (and any equivalent) are forbidden. The verifier looks for them. The user has personally caught me using them session-long.

- `kept as cuda` / `kept as .cuh` / `kept as .cu`
- `per port mandate`
- `VK PORT NOTE: file kept`
- `open-coded by` / `open-coded`
- `inline the loop` / `callers should inline` / `caller should inline`
- `Phase N fleshout` / `deferred to Phase N` / `downstream phase` / `wired later`
- `padded to` (when justifying workgroup size deviation from CUDA)
- `for the L2 fleshout` / `the L2 fleshout may add` (anticipating speculative needs)
- `can't reliably load` / `refuse the call` (admitting fake code)
- "VK adaptation: deferred to ..." — using the allowed comment style as a deferral vector

Anything that JUSTIFIES not porting something CUDA has, or that anticipates "future phase will handle it" — FORBIDDEN.

### ALLOWED adaptation comments

Single-line `// VK adaptation: <one-line technical reason describing HOW the Vulkan mechanism differs>` is fine.

Examples that ARE legitimate:
- `// VK adaptation: VkDevice is not thread-local; no per-thread context binding needed.`
- `// VK adaptation: vkCmdCopyBuffer requires a recorded command buffer; we maintain a per-context cmdbuf.`
- `// VK adaptation: GLSL cannot pass SSBOs as function parameters; spelled as a function-like macro that preserves CUDA's call-site shape.`

Examples that are NOT legitimate (and were caught):
- `// VK adaptation: deferred to Phase 4 fleshout`
- `// VK adaptation: the L2 fleshout can declare real bindings without revisiting the loader`

The test: does the comment explain HOW the implementation works in Vulkan, or does it explain WHY the work isn't being done? Former = OK. Latter = forbidden.

### /src/, /src_vulkan/, /src_vk/, /third_party/, /include/

- `/src/` — CUDA structural source. READ ONLY. Source of truth for everything.
- `/include/` — C ABI headers. READ ONLY. Stay at `/include/`; NOT duplicated under `/srcVK/`.
- `/src_vulkan/` — the legacy reimplementation. OFF-LIMITS during port. Allowed only as Vulkan API patterns reference (which extensions to enable, which memory type bits to use, how vkCmdCopyBuffer staging is wired). Never copy names, never copy structure. Reserved as Step 5 troubleshooting reference.
- `/src_vk/` — TRASH from the failed prior workflow (`wf_0c9c5ca6-39f`). Has `.cu`/`.cuh` extensions and `cuda_ffi.zig` filenames. IGNORE entirely. **CANNOT be deleted** per user direction.
- `/third_party/vma/` — VMA vendored by the failed prior workflow. We have our OWN copy at `/srcVK/vma/`. Do NOT reference `/third_party/` from anywhere in `/srcVK/`. /srcVK/ is self-contained.

### VMA self-contained
- `/srcVK/vma/vk_mem_alloc.h` + `/srcVK/vma/vk_mem_alloc_impl.cpp` — the VMA library, vendored under `/srcVK/`.
- `/srcVK/vma.zig` — Zig binding via @cInclude `vma/vk_mem_alloc.h`.
- build.zig adds `/srcVK/vma/` to include path for srcVK targets only.

### Build steps
- `zig build streamlz_vk -Doptimize=ReleaseFast` — the NEW srcVK port; produces `zig-out/bin/streamlz_vk.exe`. THIS IS THE PORT.
- `zig build streamlz_vk_old -Doptimize=ReleaseFast` — the legacy `src_vulkan/` reimplementation; produces `streamlz_vk_old.exe`. Preserved for tooling continuity.
- DO NOT modify the `streamlz` (CUDA) build step.

### Binary location
- `zig-out/bin/streamlz_vk.exe` — the FRESH port binary. **THIS IS THE ONLY PLACE TO LOOK.**
- `zig-out-rf/bin/streamlz_vk.exe` — STALE. Will NOT have any of the port work. Multiple agents quoted wrong perf numbers by running this stale binary. Always check `stat zig-out/bin/streamlz_vk.exe` mtime > source mtime before quoting any number.

### CLI / device selection (post Phase 12)
- The CLI must support: `-c` (compress), `-d` (decompress), `-b` (full bench), `-db` (decompress-only bench), `--probe` (list devices), `--device N` or `--device <name>` (select device), `--version`, `--help`, `-r N` (run count), `-l N` (level, only 1 supported initially), `-o PATH` (output).
- Default device selection: prefer DISCRETE_GPU > INTEGRATED_GPU > VIRTUAL_GPU > CPU > OTHER. Environment override: `SLZ_VK_DEVICE_INDEX=N`.
- The CLI MUST print the bound `vkPhysicalDeviceProperties.deviceName` on every invocation. `Device: NVIDIA GeForce RTX 4060 Ti` etc.
- Diagnostic env var: `SLZ_VK_PROFILE_DECODE=1` enables per-phase QPC timing + per-kernel `kper:` line + caller-import path telemetry.

---

## 4. Workflow Tool Patterns

Ultracode mode is ON in this session. Use the Workflow tool for substantive tasks.

### Phase shape per fleshout
1. `phase('Fleshout')` — single agent ports the phase's files. Agent reads CUDA source, replaces skeleton bodies with real implementations.
2. `phase('Verify')` — parallel adversarial verifiers, one per ported file. Each verifier checks: function names verbatim, control flow preserved, no rationalization phrases, no wholesale-copy from src_vulkan/, build still passes.
3. Read verdicts. If any partial/fake → fix-up sub-workflow before moving on.

### Schemas
- Keep schemas SIMPLE. Prior workflow (`wf_0c9c5ca6-39f`) failed at the StructuredOutput step on 32 of 135 agents with complex schemas. Smaller `required` fields, simpler descriptions.
- Use schema enum constraints: `verdict: enum ['clean', 'partial', 'fake']`. Default to fake if uncertain.

### Verifier prompts must include
- Read both CUDA + srcVK files
- List CUDA functions present
- Check name parity
- Check control-flow parity (spot-check 3 functions)
- Grep for forbidden rationalization phrases
- Grep for banned tokens in identifiers
- Check src_vulkan wholesale-copy via structural comparison
- For .comp: SSBO bindings match CUDA arg list 1:1, workgroup size matches CUDA launch dims
- For module_loader: pipelines_created (vkCreateComputePipelines called), launch_kernel_real (vkCmdBind+Dispatch in body)

### Self-check the agent must do before reporting done
- Every CUDA function from the source has a srcVK counterpart with the same name
- Every CUDA struct/type has a counterpart
- No `error.NotYetPorted` bodies left in L1-scope files
- L2 stub bodies: `return error.NotImplementedL2;` (Zig) or empty `void main()` with bindings (GLSL)
- Build passes: `zig build streamlz_vk -Doptimize=ReleaseFast` exit 0
- Grep for forbidden phrases on the file(s) just written: zero hits

### Multi-agent token failure mode
- Prior big workflow (`wf_0c9c5ca6-39f`) ran 135 agents and 32 failed at StructuredOutput.
- Likely cause: quota / per-agent token budget exhaustion at the end of long runs.
- Mitigation: keep individual workflow runs to ≤20 agents. For large phases, split into sub-workflows.
- Resume via `Workflow({scriptPath, resumeFromRunId})` — completed agents return cached results.

### Watch live progress
- `/workflows` slash command (not always invoked)
- Read journal.jsonl in the workflow run dir for agent start/complete events
- Don't poll constantly — agents auto-notify on completion

---

## 5. Mistakes I've Made (DO NOT REPEAT)

The user has been crystal clear and patient. The failures below have cost real time and frustration. Burn these in.

### M1. "Port" = behavioral correctness, not code translation
Repeatedly treated 1:1 port as "byte-identical output and matching perf." It is NOT. A port means: file structure mirrors CUDA, function names verbatim, call order verbatim, control flow verbatim. Tested by `diff` between CUDA and VK pairs producing one-line changes per function (CUDA-call → procs.* wrapper).

When tempted to write "structurally equivalent" or "behaviorally faithful" — STOP. That's the failure pattern.

### M2. Rationalization escape hatches
Words I used to dress up reinvention as port:
- "structural cost"
- "deliberate port decision"
- "VK adaptation: deferred to Phase X"
- "padded to keep workgroup divisible by a subgroup"
- "the L2 fleshout may add"
- "we can't reliably load"
- "refuse the call"

EVERY ONE of these was caught by the user. The verifier now grep-checks for these patterns. They are FORBIDDEN.

If you find yourself writing one, the action is: STOP, port the thing properly, don't write the comment.

### M3. Narrowing scope when work is broad
When a verifier returns "fake" or "partial", the temptation is to scope the fix narrowly ("just fix the one symptom"). The user has caught me doing this; the fix-up agents end up needing multiple rounds.

Better: fix the ROOT pattern. If `procs.launch_kernel` returns -1, the fix isn't "implement launch_kernel" — it's "implement launch_kernel AND audit all other procs slots for the same rationalization." Phase 3+5 fix-up did this correctly by auditing all .comp workgroup sizes when one was wrong.

### M4. Adding memory rules instead of applying ones I have
The user caught me adding `feedback_port_means_port.md` to memory after I'd violated the existing `feedback_port_dont_reinvent.md`. Adding rules ≠ applying rules. The countermeasure is mechanical scaffolding (extension tables, closed cu/cuda rules, grep-checks for forbidden phrases, adversarial verifiers), not more rules.

### M5. Trusting agent self-reports
- Multiple sessions of "VK e2e regressed" turned out to be Intel iGPU mislabeled as NVIDIA.
- Agent claimed "17 of 18 procs slots implemented" when several were -1 stubs.
- Foundation agent claimed "vk_mem_alloc.h copied" when the path was wrong.

The countermeasure: adversarial verifiers that re-check the work mechanically. If an agent says "X is done," the verifier reads the file and confirms X actually appears. Build passes ≠ correctness; grep is the floor.

### M6. Running stale binaries
Multiple agents quoted Intel-iGPU perf numbers thinking they were NVIDIA, because `zig-out-rf/bin/streamlz_vk.exe` (a stale prior build) was on the path. The fresh build lands at `zig-out/bin/`.

ALWAYS: `stat zig-out/bin/streamlz_vk.exe` mtime > source mtime before quoting any number. If it's older, rebuild.

### M7. nsys Vulkan tracing
The Vulkan trace layer needs explicit env vars to load:
```
export VK_ADD_LAYER_PATH="C:\\Program Files\\NVIDIA Corporation\\Nsight Systems 2025.5.2\\target-windows-x64\\vulkan-layers"
export VK_INSTANCE_LAYERS="VK_LAYER_NV_nsight-sys"
export ENABLE_VK_LAYER_NV_nsight_sys=1
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
"$NSYS" profile --trace=vulkan --output=c:/tmp/<name> --force-overwrite=true <binary>
"$NSYS" stats c:/tmp/<name>.nsys-rep --report vulkan_api_sum
```

Without these env vars, `nsys --trace=vulkan` produces an empty trace. Prior agents thought "nsys can't trace VK" — that was wrong.

### M8. Workflow prompt failures cascade
The prior big workflow (`wf_0c9c5ca6-39f`) produced trash (`.cu`/`.cuh` extensions, `cuda_ffi.zig` filename) because the Audit phase's prompt didn't include the extension translation table. Once the Audit set wrong target paths, every downstream port agent followed them.

Lessons:
- Spec EVERY rule explicitly. If an example is given for cuda_api.zig, also give cuda_ffi.zig.
- Include explicit grep-checks the agent must run before reporting done.
- Add a "verify N samples before fan-out" phase: port 3-5 files, verify, then dispatch the rest.
- The verifier prompts must check the EXACT rules the agent prompts mandate. The verifier was blind to extension violations until I added the explicit check.

### M9. "VK PORT NOTE" was a weaponized comment style
I introduced `// VK PORT NOTE:` as a way to document Vulkan-forced adaptations. Agents used it to rationalize spec violations ("file kept as cuda_ffi.zig per port mandate"). Now banned.

The replacement: `// VK adaptation: <one-line technical reason describing HOW>` — restricted to mechanism, not justification.

### M10. Encoder scope error
Initially scoped the port as "decode only" because of a saved memory rule (`feedback_decode_over_encode.md`) from the previous perf-chasing session. The user corrected: this port is `src_vulkan/` parity, which means BOTH encode AND decode at L1.

The `feedback_decode_over_encode.md` memory is STALE for the port arc. Encoder IS in scope. The audit's encoder reclassification (Section G.4 — 18 encode files L1 scope) is the correct mapping.

### M11. Audit produced wrong file count tallies
The Audit agent reported "73 source + 11 artifacts" when reality is 63 source + 10 artifacts = 73 total. The breakdown was wrong but coverage was complete. Don't trust tally arithmetic without re-counting.

### M12. Stale memory files
Several saved feedback rules reference old work. Current relevance:
- `feedback_port_dont_reinvent.md` — STILL APPLIES (governs everything)
- `feedback_port_means_port.md` — STILL APPLIES (corollary)
- `feedback_verify_device_name.md` — STILL APPLIES (for perf measurements later)
- `feedback_l1_completion_bar.md` — STALE numbers (1.14× was for the failed src_vulkan/ rewrite); the perf bar definition is still correct but the achieved numbers don't apply to the new port
- `feedback_decode_over_encode.md` — SUPERSEDED for this port (encoder IS in scope)

---

## 6. Perf Bar (Step 5 — eventually)

After all phases land and the binary works:

### Measurement protocol
1. Confirm fresh binary: `stat zig-out/bin/streamlz_vk.exe` mtime > source mtime
2. Confirm NVIDIA device: run `./zig-out/bin/streamlz_vk.exe --probe` and verify `NVIDIA GeForce RTX 4060 Ti` appears
3. Run bench: `./zig-out/bin/streamlz_vk.exe -db -r 5 c:/tmp/enwik8_L1.slz` and confirm the `Device:` line shows NVIDIA
4. Capture median e2e over 5 runs

### Apples:apples adjustment
CUDA wastefully dispatches L2 kernels (prefix_sum, scan_parse, compact_*, gather, merge_huff, huff_build_lut, huff_decode_4stream) on L1 frames — ~0.5 ms of wasted GPU work. The audit ideal-CUDA target is **17.00 ms** on enwik8 (vs measured 17.50 ms).

### The bar
- Functional: byte-identical roundtrip vs CUDA on enwik8 + silesia + web.txt
- Structural: 1:1 file/function pairing audit
- Perf: NVIDIA enwik8 e2e ≤ 1.10× ideal-CUDA (≤18.7 ms for enwik8, scale similarly for silesia)

### Things that should already match CUDA from prior work
- Workspace pool (decode_workspace.zig pattern; mirror CUDA's ensureDeviceBuf)
- Single submit per decode (S003 from prior session — bake in from day one)
- VK_EXT_external_memory_host for caller-buffer import (when caller buffer is page-aligned)
- Persistent descriptor sets (NOT per-call allocation)

---

## 7. Key References

- `/srcVK/audit.md` — canonical file mapping, extension rules, token rules, L1/L2 classification. **READ THIS FIRST.**
- `/srcVK/PortInstructions.md` — per-file fleshout checklist with CUDA references and dependency lists
- `/src/` — CUDA structural source (READ ONLY)
- `/src_vulkan/` — Vulkan API patterns reference only (NEVER for structure, NEVER for names)
- `/srcVK/decode/vulkan_api.zig` — procs.* surface declaration; 18 procs members; all the type aliases (VkDeviceBuffer, VkStream, VkResult, VK_SUCCESS_RC)
- `/srcVK/decode/descriptors.zig` — POD descriptor structs that match CUDA byte-for-byte
- `/srcVK/decode/module_loader.zig` — Phase 3 reference port; vkCreateComputePipelines pattern; procs.launch_kernel real impl; descriptor pool management
- `/srcVK/encode/module_loader.zig` — Phase 3 encode reference; chains into decode init for shared VkDevice
- `/srcVK/vma.zig` + `/srcVK/vma/vk_mem_alloc.h` + `/srcVK/vma/vk_mem_alloc_impl.cpp` — VMA self-contained
- `build.zig` — has both `streamlz_vk` (new srcVK port) and `streamlz_vk_old` (legacy src_vulkan/) steps

### CUDA file lookups (commonly needed)
- `src/decode/cuda_api.zig` — procs struct surface (port → vulkan_api.zig)
- `src/decode/decode_dispatch.zig` — fullGpuLaunchImpl, the L2 gate, the launch sequence
- `src/decode/decode_context.zig` — ensureDeviceBuf, KernelTiming, the workspace pattern
- `src/decode/streamlz_decoder.zig` — top-level decode entry; CPU-side frame walk, buildChunkDescriptors
- `src/decode/scan_gpu.zig` — gpuPrefixSumChunksImpl, gpuScanChunks, walk_frame_fn launch
- `src/encode/streamlz_encoder.zig` — top-level encode entry
- `src/encode/fast_framed.zig` — compressFramedOne; the L2 gate at `opts.level >= 3`
- `src/encode/encode_lz.zig` — gpuCompressImpl, the LZ kernel launcher
- `src/encode/encode_assemble.zig` — frame assembly chain
- `src/encode/lz_kernel.cu` — slzLzEncodeKernel (L1 hot path)
- `src/decode/lz_decode_kernels.cuh` — slzLzDecodeKernel (L2) + slzLzDecodeRawKernel (L1)

### Audit Section H open questions (still open)
1. Subgroup size policy on Intel iGPUs: locked WARP_SIZE=32 (gpu_warp.glsl) per audit Q1
2. `sm_count` analogue for `fast_framed.resolveScGroupSize`: TBD (fleshout will decide)
3. Async D2D entry points: currently L2 stubs (decompressFramedFromDevice etc.)

---

## 8. Specific Next Step (Phase 4)

Phase 4 is the **L1 decode kernels — real GLSL ports**. 6 files. **Highest port-fidelity risk yet** because CUDA `__device__` C++ + warp intrinsics become GLSL with subgroup intrinsics, and subgroup semantics differ subtly from CUDA warp semantics on edge cases.

### Files
1. `srcVK/decode/slz_wire_format.glsl` ← `src/decode/slz_wire_format.cuh` (device-side wire-format constants + SSBO descriptor struct layouts MUST match `descriptors.zig` byte-for-byte)
2. `srcVK/decode/lz_decode_core.glsl` ← `src/decode/lz_decode_core.cuh` (shared warp primitives: `warpScanU32`, `warpLiteralCopy`, `warpMatchCopy` — these are L1 hot-path inner loops)
3. `srcVK/decode/lz_decode_raw.glsl` ← `src/decode/lz_decode_raw.cuh` (raw-mode sub-chunk decoder helpers)
4. `srcVK/decode/lz_dispatch.glsl` ← `src/decode/lz_dispatch.cuh` (parseAndDecodeSubChunkRaw — L1; general path stays stubbed)
5. `srcVK/decode/lz_decode_raw_kernel.comp` (entry `slzLzDecodeRawKernel` — the actual L1 decode workhorse kernel; bindings already declared in Phase 3+5 fix-up — fill the body)
6. `srcVK/decode/prefix_sum_chunks_kernel.comp` (entry `slzPrefixSumChunksKernel` — always runs)

### Critical port concerns
- CUDA `__shfl_sync` → GLSL `subgroupShuffle`
- CUDA `__ballot_sync` → GLSL `subgroupBallot`
- CUDA `__clz` → GLSL `findMSB` (be careful: `__clz(0) = 32` on CUDA; `findMSB(0) = -1` on GLSL — possibly needs `(32 - 1) - findMSB(x)` wrapper)
- CUDA `__popc` → GLSL `bitCount`
- WARP_SIZE = 32 verbatim per gpu_warp.glsl
- SSBO bindings already declared in Phase 3+5 fix-up — fill the body, don't rewrite the bindings
- Cite CUDA file:line for every device function ported
- DO NOT consult src_vulkan/'s `lz_decode.comp` for structure (it's the failed reimplementation); only for "how does GLSL handle X" patterns

### Workflow shape for Phase 4
- Single fleshout agent
- 6 parallel verifiers (one per file)
- Verifier checks: function names verbatim, SSBO bindings preserved, subgroup intrinsics used correctly, NO rationalization, NO src_vulkan wholesale-copy
- If any verifier returns partial/fake → fix-up sub-workflow

### After Phase 4
Phase 6 is decode_context.zig + scan_gpu.zig. Phase 7 is decode_dispatch.zig with the L2 gate. After Phases 6+7 the decode path is fully real. Then encode (8+9+10), then top-level (11), then CLI (12), then tests (13).

---

## 9. User Pattern Summary

The user prefers:
- **Sequential review-heavy work** for high-fidelity ports. One file at a time when stakes are high; small parallel batches when low-risk.
- **Honest reporting**: if something's partial, say so. Don't hide behind summary words.
- **Brief responses**. The user is impatient with verbose narration. Lead with the answer, then explain.
- **Concrete decisions** before launching. Get explicit go-ahead before spawning agents on substantive work.
- **No piping** through `head`/`tail`/`grep`/`cat` when running CLI tools — see full output.
- **No reverts without permission** — only authorized revert is own-commit-correctness-regression.

The user's vibe: precise, exacting, has been burned, will catch every rationalization. The right response is to over-deliver on fidelity and under-promise on scope.

---

## 10. End of Gameplan

If you're the post-compact me (or any agent picking this up), the immediate action is:

1. Read `/srcVK/audit.md` end-to-end
2. Read `/srcVK/PortInstructions.md` Phase 4 section
3. Verify build still works: `zig build streamlz_vk -Doptimize=ReleaseFast` from project root
4. Verify build of legacy: `zig build streamlz_vk_old -Doptimize=ReleaseFast`
5. Read the prior Phase 3+5 module_loader.zig files as templates of correct port style
6. Confirm with user before launching Phase 4 fleshout — they'll want to review the agent prompt first

The done bar is **functional + structural + perf**. Not before. Not "good enough." Not "1.14× is fine."

**Do not be lazy. Do not rationalize. 1:1 port.**
