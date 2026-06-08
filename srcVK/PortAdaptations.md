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
- **Status**: **ACTIVE**. **Workaround**: `tools/build_vk.bat`
  (commit `db4406c`) does clean-build with explicit cache wipe.
  **Proper fix** (would close this entry): add the included
  `.glsl` files as `addFileInput()` to the SPV build step, OR
  migrate to `glslc -MD` depfile output for automatic dependency
  tracking. ~30-60 min Zig build-system work, no functional
  impact on the binary.

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
