# srcVK Port Adaptations Catalog

Every place where VK behavior intentionally diverges from CUDA. Each entry
declares its class and whether **runtime code-path verification** has been
done — not just per-line static review.

## Why this file exists

Phase 2A-decoder iter 4 shipped with the adversarial reviewer correctly
flagging a dispatch-fork divergence (entry A-001 below) as a "legitimate
VK adaptation" forced by GLSL's lack of pointer polymorphism. The
classification was right (GLSL really can't do CUDA's polymorphic
pointers) but the implications were under-explored: **VK and CUDA
literally execute different code paths for L3 SC1 with entropy lit+cmd**.
That divergence hid the byte-65544 silent-corruption bug for 5+ hours
of debugging because 8 rounds of byte-for-byte static diffs were
checking like-line-against-like-line, never asking "does VK reach the
same line CUDA reaches for this input?"

This catalog forces every adaptation to declare both static AND runtime
verification status. An unverified adaptation is a known risk surface.

## Entry classes

- **language-forced**: GLSL/SPIR-V genuinely can't express the CUDA
  construct. No alternative. Risk: incorrect emulation logic.
- **runtime-forced**: Vulkan execution model requires the change. Risk:
  wrong access flags / scope / timing.
- **structural-limit**: Vulkan device limit forces a behavioral cap
  (4 GiB SSBO range, alignment, etc.). Risk: silent truncation /
  workload-dependent behavior.
- **design-choice**: VK could have mirrored CUDA but chose a different
  shape. Risk: highest — both static AND runtime verification needed
  because there was no force-function compelling the divergence.

## Entry template

```markdown
### A-NNN: <short name>
- **File:line**:
- **CUDA reference**:
- **Class**:
- **Why**:
- **Risk if wrong**:
- **Static verification**: <how the per-line port was reviewed>
- **Runtime verification**: <how VK actual execution path was confirmed
  vs CUDA — or NOT DONE if pending>
- **Discovered**: <commit / iter / review>
- **Status**: ACTIVE | RESOLVED (commit X) | DELETED
```

---

## Entries

### A-001: Dispatch fork on `lit_in_scratch || cmd_in_scratch`
- **File:line**: `srcVK/decode/lz_dispatch.glsl:239-322` (resolution)
- **CUDA reference**: `src/decode/lz_dispatch.cuh:190-214` — CUDA
  dispatches to `decodeSubChunkRawMode<true>` unconditionally when
  `mode==1 && off32-empty`, even with entropy lit/cmd (pointer
  polymorphism lets `lit_ptr` point into entropy_scratch
  transparently)
- **Class**: design-choice (language-forced underlying constraint;
  resolved via macro-textual specialization)
- **Why**: GLSL has no pointer polymorphism. Pre-fix: VK forked to
  general mode for entropy-lit/cmd. Post-fix: 8-arm dispatch
  enumeration over `(off16_split, lit_in_scratch, cmd_in_scratch)`
  with the right SSBO names baked per arm. Mirrors CUDA's
  polymorphism through GLSL macro-textual expansion.
- **Risk if wrong (pre-fix)**: VK executed a code path CUDA never
  tests → bug surfaced only in VK at byte 65544
- **Static verification**: Per-line review of general mode body
  matched CUDA byte-for-byte (8 rounds of agent diffs found nothing
  because they were checking the wrong file — CUDA never enters
  general for the failing case)
- **Runtime verification**: ✅ Phase 2A-decoder iter 4f at `4270ea4`
  — ptest_vk 98/0/10 → 108/0/0; L3+L4 VK→VK roundtrip byte-identical
  to source on web + enwik8; verified via direct CLI test on 128 KiB
  web.txt head (CUDA→VK decode SHA matches)
- **Discovered**: Iter 4 adversarial review (workflow `wygjxjrvo`)
  flagged + recommended verification; verdict mis-classified as
  `LEGITIMATE_VK_ADAPTATION` / `ship_as_is`; verification dropped.
  Bug surfaced via 10 corpus L3/L4 ptest cases at `first_diff=65544`
- **Status**: ✅ **RESOLVED at `4270ea4`**. The iter 4f fix was
  authored mid-investigation but masked for 5+ hours by stale SPV
  binary (see A-012). Once forced clean rebuild via `rm
  zig-out/srcvk_shaders/lz_decode_kernel.spv`, the fix proved correct
  on first run.

### A-002: `__match_any_sync` → 32-iter shuffle loop
- **File:line**: `srcVK/decode/huff_build_lut_kernel.comp:133-140`
  (`matchAny32` helper)
- **CUDA reference**: `src/decode/huffman_kernel.cu:259` (single SASS
  instruction `__match_any_sync(0xFFFFFFFFu, val)`)
- **Class**: language-forced
- **Why**: GL_NV_shader_subgroup_partitioned is NV-only; emulation
  via 32-iter `subgroupShuffle` loop is the only cross-vendor option
- **Risk if wrong**: incorrect leader-detect in LUT Pass 5 → wrong
  per-symbol code assignment → garbage decode
- **Static verification**: Reviewed in iter 2 adversarial review
  (`wf` for `6a1da40`); 32 shuffles per warp per LUT build, minor cost
- **Runtime verification**: Yes — covered by 5 huff_decode_conformance
  tests in `srcVK/tests/huff_decode_conformance.zig`
- **Discovered**: Phase 2A-decoder iter 2 (`6a1da40`)
- **Status**: RESOLVED — verified by conformance test

### A-003: `__byte_perm(word, 0, 0x0123)` → `bswapU32` shift+mask
- **File:line**: `srcVK/decode/huff_decode_4stream_kernel.comp:133-138`
- **CUDA reference**: `src/decode/huffman_kernel.cu:431, 482, 528, 581`
- **Class**: language-forced
- **Why**: GLSL has no `__byte_perm` equivalent
- **Risk if wrong**: bit-stream byte order wrong → garbage Huffman
  decode
- **Static verification**: Walked through w=0xAABBCCDD case in iter 3
  adversarial review (`w5bqf1x6a`); produces 0xDDCCBBAA matching CUDA
- **Runtime verification**: Yes — covered by 5 huff_decode_conformance
  tests
- **Discovered**: Phase 2A-decoder iter 3 (`d5860d1`)
- **Status**: RESOLVED — verified by conformance test

### A-004: u64 `bit_buf` → `uvec2` pair (`hi`, `lo`)
- **File:line**: `srcVK/decode/huff_decode_4stream_kernel.comp:104-112`
- **CUDA reference**: `src/decode/huffman_kernel.cu:386-610`
  (`uint64_t bit_buf`)
- **Class**: language-forced (shaderInt64 not enabled)
- **Why**: 64-bit integer ops not enabled in this codec; logical u64
  bit-buffer split to high/low u32 pair with shift-overflow guards
- **Risk if wrong**: Critical — `bitbufLutIndex` reads `b.hi >> 22`
  expecting top 10 bits of logical u64. If refill puts bits in wrong
  half, decoder reads garbage
- **Static verification**: Initial iter 3 commit had hi/lo SWAPPED in
  `bitbufRefill`. Caught by iter 3 adversarial review (`w5bqf1x6a`)
  CRITICAL_PORT_VIOLATION
- **Runtime verification**: Yes — explicitly tested by re-introducing
  the bug, confirming 5 conformance tests fail, then restoring
- **Discovered**: iter 3 review caught; fixed at `ac6696f`
- **Status**: RESOLVED + conformance-test-pinned

### A-005: u64 `entropy_slot_stride` → `uvec2.x` only
- **File:line**: `srcVK/decode/lz_decode_kernel.comp:107`
- **CUDA reference**: `src/decode/lz_decode_kernels.cuh:42`
  (`uint64_t entropy_slot_stride`)
- **Class**: language-forced + structural-limit
- **Why**: shaderInt64 not enabled; for supported workloads
  `total_subchunks * 131072 < 4 GiB` so `.x` (low 32 bits) suffices
- **Risk if wrong**: Silent truncation on workloads where
  `entropy_slot_stride.y != 0` (i.e., total_subchunks > 32768 at
  131072 B slots = >4 GiB scratch). Host code at `decode_dispatch.zig:766`
  packs both halves but kernel ignores `.y` with no assert
- **Static verification**: Iter 4c review (`wc30y5xdn` finding F-005)
  flagged as MEDIUM — recommended host- or kernel-side assert that
  `.y == 0u`
- **Runtime verification**: Implicitly verified for current workloads
  (web/enwik8 fit in u32). NOT verified for very-large inputs
- **Discovered**: Iter 4 (`ac013b5`)
- **Status**: ACTIVE — accept truncation for current workloads;
  add assert when targeting >4 GiB scratch

### A-006: `compute_to_compute_barrier` between Huff decode + LZ decode
- **File:line**: `srcVK/decode/decode_dispatch.zig::runBackHalf` +
  `srcVK/decode/module_loader.zig::procComputeToComputeBarrier`
- **CUDA reference**: implicit per-stream `cuLaunchKernel` ordering
  (no explicit barrier needed)
- **Class**: runtime-forced
- **Why**: Vulkan `vkCmdDispatch` has no implicit RAW ordering
  between consecutive dispatches; explicit `vkCmdPipelineBarrier`
  with VK_ACCESS_SHADER_WRITE_BIT → VK_ACCESS_SHADER_READ_BIT
  required
- **Risk if wrong**: LZ decoder reads stale entropy_scratch data
- **Static verification**: Access flags reviewed in iter 4 commit msg
- **Runtime verification**: 5 huff_decode_conformance tests pass +
  L1+L2 roundtrip preserved (probe #1 confirmed entropy_scratch is
  correctly populated when LZ decoder reads it)
- **Discovered**: Iter 4 (`ac013b5`)
- **Status**: RESOLVED

### A-007: Per-binding offset ABI
- **File:line**: `srcVK/decode/module_loader.zig::procLaunchKernel`
  + `srcVK/decode/vulkan_api.zig::procs.launch_kernel`
- **CUDA reference**: CUDA `cuLaunchKernel` takes raw `void**` params;
  pointer arithmetic on device pointers is transparent
- **Class**: runtime-forced
- **Why**: Vulkan `VkDescriptorBufferInfo` requires explicit `offset`
  field for sub-range binding; raw arithmetic on `VkBuffer` handles
  is invalid (handles are 1-based indices into a registry, not
  pointers)
- **Risk if wrong**: KernelLaunchFailed (-1 from lookupAlloc) — loud
- **Static verification**: 14 call sites audited backward-compat;
  per-binding `null` default preserves legacy behavior
- **Runtime verification**: ptest_vk 88/0/0 preserved after change;
  L3 dispatches reach kernel without `KernelLaunchFailed`
- **Discovered**: Iter 4c (`11eb101`)
- **Status**: RESOLVED

### A-008: `hash_bits` clamp 19→18 at L3 for >128 MiB single-frame
- **File:line**: `srcVK/encode/encode_lz.zig` (24 LOC added in
  `30f36d3`)
- **CUDA reference**: `src/encode/encode_lz.zig:43` — CUDA allocates
  full `num_chunks × (1<<hash_bits) × 4` bytes with no cap
- **Class**: structural-limit
- **Why**: Vulkan `maxStorageBufferRange = 4 GiB - 1` on every
  desktop GPU; at L3 + sufficient input size, hash table exceeds cap
- **Risk if wrong**: silent silesia L3 0.02% size delta; would be
  worse on inputs >200 MiB
- **Static verification**: Documented in Phase 2A.5 commit `30f36d3`
- **Runtime verification**: Empirically verified silesia L3 ratio
  drops from 1.176× to 1.0002× CUDA
- **Discovered**: Phase 2A.5 (`30f36d3`)
- **Status**: ACTIVE — documented accepted residual; future BDA work
  would close it

### A-009: 256-byte hardcoded `SSBO_ALIGN`
- **File:line**: `srcVK/decode/scan_gpu.zig:215, 333` (constant)
- **CUDA reference**: N/A (CUDA has no SSBO alignment concept)
- **Class**: design-choice
- **Why**: VK requires `minStorageBufferOffsetAlignment` (16-256 B
  varies by GPU); 256 chosen as a safe upper bound for all
  supported devices
- **Risk if wrong**: 240-byte waste per slot on NVIDIA (align=16);
  unlikely-to-break but bloats `d_compact_counts` from 24 B to 1.5 KB
- **Static verification**: Iter 4c review (`wc30y5xdn` finding F-003)
  flagged as HIGH but not must-fix
- **Runtime verification**: Yes — works on both NVIDIA (16-align) and
  Intel iGPU (64-align)
- **Discovered**: Iter 4c (`11eb101`)
- **Status**: ACTIVE — accept bloat; future cleanup could derive
  from `g_min_storage_buffer_offset_alignment`

### A-010: `INITIAL_LITERAL_COPY_BYTES` host post-pass restore
- **File:line**: `srcVK/decode/streamlz_decoder.zig:226-246`
- **CUDA reference**: `src/decode/streamlz_decoder.zig:287-308` —
  CUDA has the same host post-pass
- **Class**: design-choice (mirrors CUDA exactly)
- **Why**: First 8 bytes of each chunk are written by host from a
  prefix table, not by GPU decoder
- **Risk if wrong**: Misleading symptom — bytes 0..7 of failing
  sub-chunks always look "correct" because host post-pass writes
  them, masking GPU decoder failures
- **Static verification**: Byte-identical to CUDA
- **Runtime verification**: ✅ L1/L2 work
- **Discovered**: pre-existing in srcVK
- **Status**: RESOLVED — direct port of CUDA logic

---

### A-012: Build-graph misses `.glsl` includes as deps of `.comp`
- **File:line**: `build.zig::addSrcVkShaderSteps` ~1031-1071
- **CUDA reference**: `build.zig` has a PTX-freshness gate per
  `tools/build_gpu.bat:11-13` comment — fails the build if any
  `.cu`/`.cuh` is newer than its `.ptx`. The Vk side has no
  equivalent gate.
- **Class**: design-choice (build infrastructure, not runtime code)
- **Why**: `addFileArg()` declares only the `.comp` file as glslc's
  input. The `#include`d `.glsl` files are read at glslc-time but
  not registered as Zig dep-graph inputs. So editing a `.glsl`
  header doesn't invalidate the cached `.spv`.
- **Risk if wrong**: catastrophic in debugging — every "verify the
  fix" test runs stale bytecode. Cost A-001 debugging ~5 hours
  during iter 4f because nobody noticed the SPV wasn't rebuilding.
- **Static verification**: N/A — this is a build-system gap
- **Runtime verification**: empirically confirmed during iter 4f
  recovery (`rm zig-out/srcvk_shaders/lz_decode_kernel.spv` +
  rebuild → fix verified working immediately)
- **Discovered**: iter 4f recovery (workflow `a0d2d6c5`)
- **Status**: ✅ **RESOLVED 2026-06-08.** `build.zig::addSrcVkShaderSteps`
  now passes `-MD -MF <depfile>` to glslc and registers the depfile via
  `addDepFileOutputArg`, so Zig parses the Make-style dependency list
  glslc emits and adds every `#include`d `.glsl` to the cache hash.
  Verified empirically: editing `srcVK/encode/lz_format.glsl` triggers
  `lz_encode_kernel.spv` rebuild on plain `zig build streamlz_vk`;
  `scan_parse_kernel.spv` (which doesn't include lz_format.glsl) does
  NOT rebuild, confirming the depfile is selective per-kernel.
  `tools/build_vk.bat` is kept as a force-clean utility for other
  cache-busting scenarios but is no longer required after `.glsl` edits.

### A-013: Chain parser `uint16_t* next_hash` → u16-packed u32 SSBO
- **File:line**: `srcVK/encode/lz_chain_parser.glsl:60-78`
  (`chainNextHashRead` / `chainNextHashWrite` helpers), used by
  `findMatchChain` + `insertChainRange`; backed by
  `srcVK/encode/lz_encode_kernel.comp:158` (`table_stride = hash_size
  + hash_size + (NEXT_HASH_SIZE / 2u)` u32 words) and host alloc
  `srcVK/encode/encode_lz.zig:87-91`
- **CUDA reference**: `src/encode/lz_chain_parser.cuh:87`
  (`uint16_t* next_hash`), `NEXT_HASH_SIZE = 65536` u16 entries
- **Class**: language-forced
- **Why**: SPIR-V std430 has no native 16-bit-element SSBO and the
  codec deliberately avoids requiring `shaderInt16` / 16-bit storage
  extensions. Two u16 entries are packed into each u32 word inside
  the same `global_hash` SSBO. The chain parser runs serially on
  lane 0, so the read-modify-write pack does not need atomics.
- **Risk if wrong**: wrong chain link → wrong match positions →
  garbage compressed output (would surface as cross-backend SHA
  mismatch + L5 roundtrip failure)
- **Static verification**: helper macros reviewed for shift/mask
  correctness; index `idx` is always `pos & NEXT_HASH_INDEX_MASK`
  (= `pos & 0xFFFF`) so `idx >> 1 < 32768` = exactly the allocated
  word count. Read formula `(word >> (16*(idx&1))) & 0xFFFF`
  identical to CUDA's u16 lookup; write formula preserves the
  other half via `~(0xFFFF << shift)` mask.
- **Runtime verification**: ✅ VK L5 encode SHA byte-identical to
  CUDA L5 on web.txt (1.46 MB), enwik8 (95 MB), silesia (203 MB);
  VK→VK roundtrip MATCH on all three; ptest_vk 99 passed / 9
  skipped / 0 failed on both NVIDIA RTX 4060 Ti + Intel iGPU
- **Discovered**: Phase 2B chain parser port (this commit)
- **Status**: RESOLVED — verified by 3-corpus cross-backend SHA gate

### A-015: CLI decompress writes via heap buffer instead of mmap'd output (Intel iGPU file-handle survival)
- **File:line**: `srcVK/cli/decompress.zig:69-118` (heap-allocate +
  decode-into-heap + write-to-disk path)
- **CUDA reference**: `src/cli/decompress.zig:11-end` — CUDA CLI
  mmap's the output file, decodes straight into the mmap, then
  `setLength` truncates the pre-sized file down to the actual written
  bytes. CUDA has no notion of host-memory import (cuMemAlloc /
  cuMemcpyDtoH operate on a separate GPU heap), so the mmap pages
  are never referenced by any GPU-side kernel object — `setLength`
  always succeeds.
- **Class**: structural-limit (Intel/WDDM-specific kernel-mode lock
  on `VK_EXT_external_memory_host`-imported pages)
- **Why**: The VK decoder's iter-7 fast path imports the user dst
  pointer as a VkDeviceMemory via `VK_EXT_external_memory_host` so
  the GPU writes the decoded bytes straight into the caller's pages
  (zero-copy D2H). On NVIDIA discrete `vkFreeMemory` synchronously
  drops the kernel-mode `MmProbeAndLockPages` reference and the
  subsequent `SetEndOfFile` on the mmap'd file succeeds. On Intel
  iGPU the WDDM 2.x driver defers the actual `MmUnlockPages` to a
  kernel-mode worker that does not run until process exit; the file's
  clusters stay "in use" and `SetEndOfFile` returns
  ERROR_ACCESS_DENIED. Verified empirically (2026-06-08): inserting
  `vkDeviceWaitIdle` before AND after `vkFreeMemory`, evicting all
  LRU cache entries, closing the original file handle and reopening
  a fresh one with read_write access, and combinations thereof all
  still fail with `AccessDenied` on Intel — the lock is on the FILE
  object, not on any user-mode handle. The encoder CLI doesn't hit
  this because it imports `allocHost`-anonymous heap buffers (not
  file-backed mmap pages) for its D2H gather, then host-memcpys the
  result into the mmap; the file is never the import target.
- **Risk if wrong**: regression to AccessDenied on Intel iGPU at the
  end of every `-d` run, with the output file 64 bytes too large
  (the `decoder.safe_space` pad) and exit code 1
- **Static verification**: traced all `prepareImportHostBuffer` /
  `tryImportHostBuffer` callers; confirmed decode path is the only
  one that imports a CLI-owned `dst` pointer (encode imports its
  internal `gpu_out_buf` / `d2h_final_buf` only)
- **Runtime verification**: ✅ Intel iGPU: L1 web (4.5 MB) + L1 enwik8
  (100 MB) + L5 web + L5 silesia (~212 MB) all decode with exit 0,
  size exactly `content_size` (no 64-B overshoot), SHA256 matches
  source. NVIDIA RTX 4060 Ti: L5 web decode unaffected (exit 0,
  correct size + SHA). `ptest_vk` 99 passed / 9 skipped / 0 failed
  on both backends.
- **Discovered**: 2026-06-08 root-cause investigation of the
  pre-existing `streamlz_vk -d` Intel iGPU AccessDenied bug
- **Status**: RESOLVED — heap buffer + writeAll path in place. The
  one-time `~content_size` memcpy on CLI exit is small (<50 ms for
  100 MB) and does not affect `-db` benchmark or library callers
  (decoder API is unchanged; only the CLI driver moved off the
  mmap'd-output shape). Could be revisited if WDDM ever ships a
  synchronous unlock for imported pages, but the current shape
  matches what the encoder CLI has always done.

### A-017: Fused 4× compact_huff_descs dispatches (`grid_x=4` + `gl_WorkGroupID.x`)
- **File:line**: `srcVK/decode/compact_huff_descs_kernel.comp` (10-binding
  rewrite + per-stream switch on `gl_WorkGroupID.x`); host call site at
  `srcVK/decode/scan_gpu.zig:335-385` (1 dispatch with grid_x=4 replaces
  the 4-iteration loop). KERNEL_DECLS at
  `srcVK/decode/module_loader.zig:1865` (`n_bindings=10`).
- **CUDA reference**: `src/decode/scan_gpu.zig:214-239` — CUDA dispatches
  `slzCompactHuffDescsKernel` 4 times (one per Huffman stream type: lit,
  tok, off16hi, off16lo) via `cuLaunchKernel`. Each call ~73 µs on RTX
  4060 Ti → ~293 µs total.
- **Class**: design-choice (port-violating in *dispatch shape*; identical
  in algorithm — each stream's compact still runs single-thread)
- **Why**: VK-on-WDDM per-dispatch overhead floor is ~75 µs higher than
  CUDA per dispatch. With 4 back-to-back dispatches the gap compounds:
  VK was at 148 µs × 4 = 593 µs vs CUDA's 293 µs (2.0×). Fusing collapses
  the 4 dispatches to one (grid_x=4); each workgroup is still
  `local_size_x=1` (compact is inherently serial per stream — the loop
  body has a sequential output cursor) but the 3 redundant
  pipeline-state-load / descriptor-fetch / workgroup-launch overheads
  evaporate. 10 bindings = 4× staged source views (one per stream at
  distinct binding-offsets in `d_scan_staged`, since 256-byte SSBO_ALIGN
  padding makes the per-stream stride non-integer in descriptor units),
  `d_total_subs`, 4× per-stream dst (`d_compact_lit/tok/hi/lo`),
  `d_compact_counts` at offset 0 (kernel writes 4 slots via
  `d_n_out[stream_type * COUNTS_STRIDE_U32]`).
- **Risk if wrong**: incorrect per-stream compaction → wrong descriptor
  arrays for downstream merge/build_lut/huff_decode → garbage decode.
  Covered by every L3/L4/L5 ptest case + cross-backend roundtrip.
- **Static verification**: per-stream switch (`if stream_type == 0u
  d_staged_lit[i] else ...`) is uniform across the workgroup since
  `local_size_x=1` — no warp divergence. Output count writes to the same
  slot the prior unfused code wrote (`stream_type * COUNTS_STRIDE_U32`
  matches the prior `n_off = COUNTS_STRIDE * stream_type` binding offset
  expressed in u32 index space).
- **Runtime verification**: ✅ **2026-06-08.** ptest_vk 99 passed / 9
  skipped / 0 failed on both NVIDIA RTX 4060 Ti + Intel iGPU. Perf:
  `compact_huff_descs_fn` 593 µs → 245 µs (**2.4× faster on this kernel**,
  -348 µs/decode). L5 enwik8 e2e 17.07 → 16.62 ms (1.054× → **1.027× CUDA**).
  L5 silesia e2e 33.12 → 32.68 ms (1.044× → **1.030× CUDA**).
- **Discovered**: 2026-06-08 secondary-perf investigation after A-016
  closed the bulk of the gap. Web search of NVIDIA Vulkan best-practices
  + WebGPU dispatch-overhead research confirmed multi-dispatch fusion
  as the canonical mitigation for the WDDM dispatch-floor pattern.
- **Status**: RESOLVED.

### A-016: u32-aliased SSBO view for the Phase 2 hot-loop store (Huffman decode)
- **File:line**: `srcVK/decode/huff_decode_4stream_kernel.comp:79-93`
  (extra binding 5 `OutputBufU32`); usage at lines 405 (Phase 2a) and
  464 (Phase 2b). Host wiring at `srcVK/decode/decode_dispatch.zig:669-686`
  and `srcVK/tests/huff_decode_conformance.zig:299-318`. KERNEL_DECLS
  entry at `srcVK/decode/module_loader.zig:1865`
  (`huff_decode_fn n_bindings=6`).
- **CUDA reference**: `src/decode/huffman_kernel.cu:506` (Phase 2a),
  `:552` (Phase 2b) — `*reinterpret_cast<uint32_t*>(out + written) = acc;`
  A single 32-bit aligned store of the bit-buffer accumulator.
- **Class**: language-forced (with significant perf consequences)
- **Why**: GLSL has no reinterpret cast across a byte-typed SSBO. The
  naive port emits four sequential `uint8_t` stores per inner-k iteration
  (the Phase 2 unroll runs the body 2× per outer iter, so 8 byte stores
  per word decoded). Empirical bisection on 2026-06-08 showed those 8
  byte stores cost **1.77 ms of the 2.46 ms huff_decode_fn kernel time**
  on enwik8 L5 — the SPIR-V→NVIDIA driver does not coalesce sequential
  uint8_t stores into a single u32 transaction even when they are
  4-aligned and consecutive. Fix: bind the SAME `VkBuffer` at two
  descriptor slots — binding 3 as `uint8_t output_buf[]` for the
  byte-granular accesses (Phase 1 preamble, Phase 2 drain, Phase 3
  finisher, lz_decode kernel) and binding 5 as `uint output_buf_u32[]`
  for the Phase 2 hot loop's 4-aligned store. Phase 1 guarantees
  `(out_base_abs + written) & 3 == 0` before Phase 2a starts, so the
  u32-aligned write is safe by construction.
- **Risk if wrong**: garbage decoded output (since the byte path was
  the correctness oracle); caught by ptest_vk's
  `huff_decode_conformance.zig` 5 isolated kernel tests.
- **Static verification**: byte layout: u32 store of `acc_lo` at
  offset `(out_base_abs + written)` writes bytes 0..3 of that word in
  LE order (low byte at lowest address) — identical to the four
  `output_buf[written + N] = uint8_t((acc_lo >> 8*N) & 0xFF)` writes
  it replaces. Alignment: Phase 1 byte-drain loop runs until
  `(out_base_abs + written) & 3 == 0`, after which Phase 2a/2b increment
  `written` by exactly 4 per store, preserving alignment.
- **Runtime verification**: ✅ **2026-06-08.** ptest_vk 99 passed / 9
  skipped / 0 failed on both NVIDIA RTX 4060 Ti + Intel iGPU. Empirical
  perf: VK huff_decode_fn 2466 µs → 772 µs (**3.19× faster**, now at
  parity with CUDA's 715 µs = 1.08×). L5 enwik8 e2e 18.60 → 17.07 ms
  (1.15× CUDA → **1.05× CUDA**, inside the 10% bar). L5 silesia e2e
  36.99 → 33.12 ms (1.17× CUDA → **1.04× CUDA**, inside the bar).
- **Discovered**: 2026-06-08 perf-gap investigation. Localized via
  bisection (early-`return` at end of each phase), then sub-bisected
  Phase 2a (skipping just the 4-byte store dropped huff_decode to 0.70 ms,
  confirming the store accounted for ~92% of Phase 2a).
- **Status**: RESOLVED.

### A-014: `ChainMatch.length` field shadowed by GLSL `.length()` method
- **File:line**: `srcVK/encode/lz_chain_parser.glsl:28-43`
  (`isLazyMatchBetter` signature)
- **CUDA reference**: `src/encode/lz_chain_parser.cuh:38-45`
  (`isLazyMatchBetter(ChainMatch cand, ChainMatch current, int32_t step)`)
- **Class**: language-forced
- **Why**: GLSL treats `.length` as the built-in array/string method
  even when applied to a struct field of that name — glslc errors
  with `'length' : does not operate on this type: in
  structure{...int length, int offset}` on `cand.length`. The
  `ChainMatch` struct is mirrored from CUDA verbatim in
  `lz_format.glsl:107-110` and is not changed here — instead the
  helper's signature takes four explicit `int` parameters
  (`cand_length, cand_offset, current_length, current_offset`)
  and call sites pass the two ints directly. Arithmetic is
  byte-identical to CUDA.
- **Risk if wrong**: spurious compile error blocking the kernel
  build (caught immediately by glslc, not a runtime risk)
- **Static verification**: function body diffed line-for-line
  against CUDA's expression
- **Runtime verification**: ✅ same as A-013 (3-corpus cross-backend
  SHA gate covers all chain-parser paths including lazy compare)
- **Discovered**: Phase 2B chain parser port (this commit)
- **Status**: RESOLVED — workaround in place; could close fully by
  renaming the ChainMatch struct field, but that touches the CUDA
  file too for 1:1 parity

### A-018: `_vk` async/poll = worker-thread + atomic-done flag (no VkFence)
- **File:line**: `srcVK/streamlz_gpu.zig` — `AsyncSlotVk`,
  `asyncCompressWorkerVk`, `asyncDecompressWorkerVk`, `pollSlotVk`
- **CUDA reference**: `src/streamlz_gpu.zig::slzCompressAsync` runs
  inline on the caller's thread + queues GPU work on the caller's
  `CUstream`; the caller's `cudaStreamSynchronize` is the only sync
- **Class**: design-choice
- **Why**: The Vulkan codec is many submits + many fences internally
  (per-chunk gather, transfer-queue split, single-submit, final D2H
  import). A single top-level VkFence cannot span all of them; the
  natural VK analog to CUDA's "fire-and-poll" would be a VkSemaphore
  with timeline values, but that requires plumbing the timeline
  values through every internal submit. The simpler shape that
  matches the C ABI contract: spawn a worker thread that runs the
  full sync codec, atomic-store `done = true` after it commits its
  result. The Poll export checks the atomic (blocking=0) or joins
  the thread (blocking=1). One std.Thread per async call (~ms
  spawn cost on Windows + Linux; dominated by the GPU work in
  every workload our async callers care about).
- **Risk if wrong**: subtle ordering bugs across the worker → poller
  result handoff. Mitigated by `release` store on `done` + `acquire`
  load in pollSlotVk so the rest of the slot (`result`, `written`)
  is visible the instant the caller observes `done == true`. Also
  mitigated by busy-flag guard in `slzCompressAsync_vk` /
  `slzDecompressAsync_vk` (returns UNSUPPORTED rather than
  stomping an in-flight slot).
- **Static verification**: Pattern mirrored verbatim from the prior
  `src_vulkan/streamlz_gpu_vk.zig::pollSlot` (same atomic ordering,
  same slot-reset semantics, same error mapping). The src_vulkan
  version shipped in the M4 wave + has been covered by
  `vk-abi-async-test` since.
- **Runtime verification**: ✅ 6 tests in
  `srcVK/tests/async_d2d_api.zig` exercise both poll modes
  (blocking + non-blocking) on both encode + decode async paths.
  Phase-4-fix iter (2026-06-08): added F-002 regression tests
  exercising back-to-back decompress on the same dst slice; these
  failed RED on the agent's first cut (F-001: missing
  `releaseImportsByHostRange` after decode through the C ABI) and
  pass GREEN after the fix landed in `decompressCore`. F-003 thread-
  join-before-slot-overwrite fix also landed in
  `slzCompressAsync_vk` / `slzDecompressAsync_vk`. All pass GREEN
  on NVIDIA RTX 4060 Ti + Intel iGPU at 122/9/0.
- **Discovered**: Phase 4 (this commit). The CUDA backend's inline-
  spawn pattern doesn't fit Vulkan's submit floor + multi-fence
  internals, but a worker-thread wrap of the same sync codec is a
  faithful behavioural mirror at the C ABI layer.
- **Status**: ACTIVE — could revisit with timeline-semaphore
  plumbing in Phase 5 if profiling shows the per-async thread
  spawn dominates real-world latency. Today the GPU work is ~ms
  and the thread spawn is ~10s of µs, so the spawn is in the noise.

### A-019: `slzMakeDeviceOnlyHandle_vk` hands back a synthetic sentinel pointer (Tier-1 D2D-test stub)
- **File:line**: `srcVK/streamlz_gpu.zig` —
  `slzMakeDeviceOnlyHandle_vk`, `g_device_only_next`,
  `isDeviceOnlySentinel`
- **CUDA reference**: NONE — the CUDA backend doesn't export
  `slzMakeDeviceOnlyHandle`. `src/streamlz_gpu.zig:44` defines an
  internal-only `device_only_host_stub_addr = 0x10` used by
  `slzCompressAsync` to mark the `src` slice as "device-resident,
  consult `d_src` for the real address". That sentinel is NOT
  exposed in the CUDA C ABI surface.
- **Class**: design-choice
- **Why**: The Vulkan header (`include/streamlz_gpu_vk.h:62`)
  reserves the symbol. The prior `src_vulkan/streamlz_gpu_vk.zig`
  stubbed it to `SLZ_ERROR_UNSUPPORTED`; this Phase-4 implementation
  hands back a monotonically-increasing non-dereferenceable pointer
  in the `[device_only_sentinel_base, +1 TiB)` range, then
  recognises addresses in that range inside `slzCompress_vk` /
  `slzDecompress_vk` and returns `SLZ_ERROR_UNSUPPORTED` (since the
  codec can't dereference a sentinel through the Tier-1 host
  bounce). This lets test code construct a non-null "device
  pointer" without owning real device memory, and asserts the
  expected D2D-not-yet-wired error path.
- **Risk if wrong**: A real device pointer accidentally falling
  into the sentinel range would be misclassified as
  "non-dereferenceable" and the call would fail with UNSUPPORTED.
  Mitigation (post-F-005, 2026-06-08): the sentinel range now starts
  at `1 << 62` (upper-half of the 64-bit address space, well outside
  any realistic `vkGetBufferDeviceAddress` return — the Vulkan spec
  places no upper bound on `VkDeviceAddress` so we cannot rely on
  driver convention). Window stays at `1 << 40` = 1 TiB.
- **Static verification**: Range check is a single mask compare.
  `1 << 62` base + 1 TiB window = `(1 << 62) + (1 << 40)`, well within
  the 64-bit address space and safely above all driver BDA returns.
  The bump stride is `@max(bytes, 4096)` page-aligned (F-006); the
  counter wraps after ~`(1 << 40) / 4096 = 256M` calls of arbitrary
  size, which we reject.
- **Runtime verification**: ✅ 2 tests in
  `srcVK/tests/async_d2d_api.zig`: "sentinels distinct" + "compress
  on sentinel returns UNSUPPORTED" — both pass GREEN on both
  backends.
- **Discovered**: Phase 4 (this commit)
- **Status**: ACTIVE — useful for D2D path testing today; Tier-2
  BDA wiring (Phase 5) will replace the sentinel with real device
  addresses queried via `vkGetBufferDeviceAddress`, at which point
  this synthetic range can retire.

### A-020: C ABI decompress releases `g_import_cache` entries on caller buffers
- **File:line**: `srcVK/streamlz_gpu.zig` — `decompressCore` (post-fix
  scope brackets the entire decode call), mirroring the
  `srcVK/cli/decompress.zig:115,120` release pattern around
  `decompressFramedThreaded`.
- **CUDA reference**: `src/streamlz_gpu.zig::decompressCore` — CUDA has
  no import cache; the decoder does plain `cudaMemcpyAsync` H2D/D2H
  against the caller's buffers and never holds a reference past the
  stream sync.
- **Class**: runtime-forced
- **Why**: The VK decoder's `VK_EXT_external_memory_host` fast path
  imports the caller's input AND output buffers into an LRU
  `g_import_cache` keyed by `(host_addr, size, usage_src)` (see
  `srcVK/decode/module_loader.zig:1244` and `releaseImportsByHostRange`
  at line 4280). Cache entries outlive any single decode so back-to-
  back decodes on the same host pages amortize the ~1 ms
  vkCreateBuffer + vkAllocateMemory + vkMapMemory import. The CLI
  already releases imports on the caller's `out_buf` (iter-8 subfix
  1, Intel iGPU file-truncate fix at `cli/decompress.zig:115,120`).
  Pre-fix, the C ABI surface (`slzDecompressHost`,
  `slzDecompressHost_vk`, `slzDecompressAsync_vk`) did NOT — and
  callers who freed-and-reallocated a same-sized `dst` between
  decodes (common allocator pattern) hit a stale (vk_buf, vk_mem)
  pair whose VkDeviceMemory imported the OLD physical pages. GPU
  writes went to the stale mapping; the caller's `dst` was never
  touched and the decoder reported `SLZ_SUCCESS`. Silent corruption.
- **Risk if wrong**: silent decompress corruption when the same dst
  address is reused with fresh physical pages — exactly the failure
  mode F-002 regression tests catch. Class is runtime-forced because
  the import cache is a Vulkan-execution-model artifact (CUDA's
  cuMemcpyAsync has no equivalent need for it).
- **Static verification**: Pattern mirrors the CLI's iter-8 subfix 1
  release calls byte-for-byte. The release is idempotent (linear
  walk over the LRU; misses are no-ops) so the belt-and-suspenders
  pre-decode release is free of correctness risk.
- **Runtime verification**: ✅ 4 F-002 regression tests in
  `srcVK/tests/async_d2d_api.zig` exercise back-to-back decompress
  on the same dst slice (sync + async, _vk + CUDA-shaped). All
  failed RED on the pre-fix code (`TestExpectedEqual`) and pass
  GREEN after the fix — confirmed on both NVIDIA RTX 4060 Ti and
  Intel iGPU at 122/9/0.
- **Discovered**: Phase 4 fix iteration (2026-06-08) — adversarial
  reviewer flagged F-001 after running the Phase 4 tests and noting
  the absent release pattern by comparison with the CLI.
- **Status**: ACTIVE — the import cache itself is a deliberate
  performance feature; the release-on-exit invariant is mandatory
  for any caller (CLI, C ABI, future bindings) that owns the host
  buffer lifecycle.

## Process notes

When the next port adaptation lands:

1. **Add an entry here BEFORE the commit lands** — the catalog should
   be a precondition of merging, not a post-hoc cleanup
2. **Both Static AND Runtime verification fields must be filled** —
   "NOT DONE" is a valid value, but it surfaces the risk
3. **"NOT DONE" entries should be PR review blockers** unless an
   explicit testing plan is attached
4. **Quarterly sweep**: review every ACTIVE entry — is the assumption
   that justified the adaptation still valid? Has the relevant
   workload changed?

## When to convert design-choice → structural-limit OR remove

Some design-choices were forced by *the absence of a feature we
hadn't enabled yet* (e.g., shaderInt64 disabled, BDA not opted into).
If enabling that feature would let us match CUDA without trickery,
the entry's resolution path is "enable feature + remove adaptation",
not "verify adaptation correctness." Track which feature would close
each design-choice in the **Status** field.
