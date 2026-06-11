# Security Policy

## Safe for Untrusted Input

The decoder validates every frame header field, block header,
internal-block descriptor, chunk header, sub-chunk header and
entropy-table parameter before any pointer arithmetic touches the
output buffer. Malformed inputs surface as a typed error
(`error.BadFrame`, `error.Truncated`, `error.OutOfMemory`,
`error.UnknownDictionary`, `error.ChecksumMismatch`, etc.) - never a
crash, never a write past the caller's output buffer.

Host-side validation lives in `src/format/frame_format.zig` and
`src/format/block_header.zig`. Two parsing surfaces exist:

- The host walker parses frame structure on the CPU and hands the
  kernels explicit descriptors with checked bounds.
- `slzWalkFrameKernel` parses frame bytes ON DEVICE for the
  device-resident (D2D) decode path. A malformed frame there does
  not segfault like a CPU parser would; it must be (and is) bounded
  by the same descriptor checks, and this surface is covered by the
  differential fuzzer (see Testing).

## Integrity: corruption is detected by default

Since 2026-06-11 every frame the GPU encoder produces carries a
chunk-Merkle content checksum (wire flag bit 5): XXH32 over the
per-chunk hashes of the source, one 4-byte trailer. The decoder
recomputes the per-chunk hashes of its own output on the GPU and
compares the root; any corruption - in transit, at rest, or from a
decoder defect - returns `error.ChecksumMismatch` instead of
silently wrong bytes. Cost is ~zero (the hash kernels overlap the
output transfer). Pass `chunk_checksum = false` in the encoder
options to strip it.

The LZ4-style whole-file checksum (flag bit 1, `content_checksum`)
remains available as an opt-in second layer for callers that want a
single hash comparable across chunk-grid settings.

For maximum safety when processing untrusted data:

- Use the framed APIs (`compressFramed` / `decompressFramed`); never
  hand-construct frame bytes.
- Cap the decompressed size with `decoder.max_content_size` (default
  4 GB). A malicious header claiming a larger content size returns
  `error.ContentSizeTooLarge` before any allocation.
- Use the `slzDecompressHost` / `slzDecompressAsync` C ABI rather
  than raw kernel launches; the wrapper enforces the same checks
  before the kernel touches caller memory. Note: the Async (D2D)
  decode path does not surface checksum verdicts to the caller yet;
  device-resident callers should validate by other means until that
  error channel lands.

## Threat model: GPU specifics

The GPU codec adds an attack surface the CPU codec did not have. The
decoder's input is a wire-format frame, but the kernels operate on
descriptors that the walker computed. Consequences:

- A crafted frame that confuses the walker can produce malformed
  descriptors, but the descriptor structs have explicit bounds
  (`comp_size`, `decomp_size`, `dst_offset`, `src_offset` - all
  checked against the block payload and the output buffer before
  the kernel launches). The on-device kernels trust their
  descriptors; the host must not.
- A frame that drives a decode kernel into a pathological loop is a
  denial-of-service vector specific to GPUs: the Windows watchdog
  (TDR) resets the device after ~2 seconds. The fuzzer tracks
  "hang" as an explicit outcome class for exactly this reason.
- The descriptor structs are hand-mirrored between Zig and CUDA.
  ABI drift is caught at build time by `static_assert(sizeof(...))`
  guards in the `.cuh` files.

## Testing

The encode + decode roundtrip is verified across:

- `zig build test` - unit + integration tests including a
  multi-shape, multi-level GPU roundtrip in
  `src/encode/gpu_roundtrip_tests.zig`. GPU tests skip when CUDA
  is unavailable in the test process.
- `zig build fuzz` then `fuzz_frames.exe` - the frame-mutation
  differential fuzzer: mutates valid frames (bit flips, header and
  field corruption, truncation, extension) and decodes them on BOTH
  backends, classifying outcomes as both-reject, both-accept-match,
  accept-mismatch, split, or hang. The two backends act as each
  other's oracle.
- `tools/bench_all.bat` - encode + decode on enwik8 (100 MB) and
  silesia (213 MB) across L1-L5, SHA-256 comparison vs the input.
  This is the canonical "is everything still byte-exact" check.
- `tools/bench_d2d.bat` - same coverage via the device-resident
  C ABI entry points (`slzCompressAsync` / `slzDecompressAsync`),
  exercising the pure-D2D path that has different bookkeeping than
  the host-bounce path.

Both bench scripts hard-fail on the first SHA mismatch.

## Reporting a Vulnerability

Please report security issues privately via GitHub Security
Advisories rather than as a public issue.

https://github.com/StreamLZ/Compressor_GPU/security/advisories/new
