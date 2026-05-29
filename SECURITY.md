# Security Policy

## Safe for Untrusted Input

The decoder validates every frame header field, block header,
internal-block descriptor, chunk header, sub-chunk header and
entropy-table parameter before any pointer arithmetic touches the
output buffer. Malformed inputs surface as a typed error
(`error.BadFrame`, `error.Truncated`, `error.OutOfMemory`,
`error.UnknownDictionary`, etc.) — never a crash, never a write
past the caller's output buffer.

The host-side validation lives in `src/format/frame_format.zig` and
`src/format/block_header.zig`. The CUDA kernels operate exclusively
on validated descriptors that the host walker assembled; they have
no parsing logic of their own that could be exploited by a crafted
frame.

For maximum safety when processing untrusted data:

- Enable content checksums (`content_checksum = true` in the
  encoder options; the decoder verifies them when present).
- Use the framed APIs (`compressFramed` / `decompressFramed`); never
  hand-construct frame bytes.
- Cap the decompressed size with `decoder.max_content_size` (default
  4 GB). A malicious header claiming a larger content size returns
  `error.ContentSizeTooLarge` before any allocation.
- Use the `slzDecompressHost` / `slzDecompressAsync` C ABI rather
  than raw kernel launches; the wrapper enforces the same checks
  before the kernel touches caller memory.

## Threat model — GPU specifics

The GPU codec adds an attack surface the CPU codec did not have. The
decoder's input is a wire-format frame, but the kernels operate on
descriptors that the host walker computed. Two consequences:

- A crafted frame that confuses the *walker* can produce malformed
  descriptors, but the descriptor structs have explicit bounds
  (`comp_size`, `decomp_size`, `dst_offset`, `src_offset` — all
  checked against the block payload and the output buffer before
  the kernel launches). The on-device kernels trust their
  descriptors; the host must not.
- The descriptor structs are hand-mirrored between Zig and CUDA
  side. ABI drift is caught at build time by `static_assert(sizeof(...))`
  guards in the `.cuh` files.

## Testing

The encode + decode roundtrip is verified across:

- `zig build test` — unit + integration tests including a
  multi-shape, multi-level GPU roundtrip in
  `src/encode/gpu_roundtrip_tests.zig`. GPU tests skip when CUDA
  is unavailable in the test process.
- `tools/bench_all.bat` — encode + decode on enwik8 (100 MB) and
  silesia (213 MB) across L1-L5, SHA-256 comparison vs the input.
  This is the canonical "is everything still byte-exact" check.
- `tools/bench_d2d.bat` — same coverage via the device-resident
  C ABI entry points (`slzCompressAsync` / `slzDecompressAsync`),
  exercising the pure-D2D path that has different bookkeeping than
  the host-bounce path.

Both bench scripts hard-fail on the first SHA mismatch.

## Reporting a Vulnerability

Please report security issues privately via GitHub Security
Advisories rather than as a public issue.

https://github.com/StreamLZ/Compressor_Native/security/advisories/new
