//! Top-level StreamLZ framed compressor — public API surface.
//!
//! Terminology: "sc" / "SC" = "self-contained" throughout this module.
//!
//! Library callers reach `compressFramed` and `compressBound` from this
//! file; the actual frame construction lives in `fast_framed.zig`. Only
//! levels 1-5 are supported; anything outside that range returns
//! `error.BadLevel`.

const std = @import("std");
const frame = @import("../format/frame_format.zig");
const lz_constants = @import("../format/streamlz_constants.zig");

const fast_framed = @import("fast_framed.zig");
const gpu_encoder = @import("driver.zig");

/// Errors the encoder can surface. Members embed
/// `std.mem.Allocator.Error` (host alloc) so the encoder propagates
/// `error.OutOfMemory` from any internal allocation without re-wrapping;
/// `gpu_encoder.GpuError` is reachable via `error.DestinationTooSmall`
/// (the only way the encoder currently surfaces a CUDA failure - the
/// downstream pipeline maps every kernel-side error to "destination
/// too small" so the caller's retry shape stays simple).
pub const CompressError = error{
    /// `opts.level` was outside the supported range. Only levels 1..5
    /// are accepted; anything else returns this immediately.
    BadLevel,
    /// `opts.block_size` is not a power of two in `[64 KiB, 4 MiB]`.
    BadBlockSize,
    /// `opts.sc_group_size_override` is non-null and non-positive (the
    /// header writer rejects `<= 0`); also returned when the resolved
    /// `sc_group_size` would push `sub_chunk_count` past the kernel
    /// grid limit.
    BadScGroupSize,
    /// `dst` is smaller than `compressBound(src.len)` OR the GPU
    /// pipeline returned a kernel-side failure. Callers cannot
    /// distinguish the two cases at this surface; the typical
    /// remediation (grow the output buffer) helps the first case and
    /// is a no-op against the second.
    DestinationTooSmall,
    /// `opts.dictionary_id` does not resolve in the dictionary
    /// registry (`src/dict/dictionary.zig`). Only registry built-ins
    /// are accepted today; custom-dictionary registration lands with
    /// the C ABI surface.
    UnknownDictionary,
} || std.mem.Allocator.Error;

pub const Options = struct {
    /// Compression level (1..5). L1-L2 are LZ-only; L3-L5 add 32-stream
    /// Huffman. The default of `1` matches the Zig CLI's speed
    /// preference; the C ABI's `slzCompressOpts_t` defaults to `5` for
    /// compression-ratio parity with nvCOMP — the two surfaces serve
    /// different audiences.
    level: u8 = 1,

    /// Emit the 8-byte content-size field in the frame header. Always
    /// true for normal use; disable only for non-seekable streams.
    include_content_size: bool = true,

    /// Block size advertised in the frame header. Must be a power of 2
    /// in `[64 KB, 4 MB]`. The GPU encoder always emits a single block
    /// per frame; this only affects the advertised value.
    block_size: u32 = lz_constants.chunk_size,

    /// SC group size override in units of 256 KB chunks. `null` lets
    /// `fast_framed.compressFramedOne` pick adaptively (0.25 below the
    /// GPU saturation threshold, 0.5 at or above).
    sc_group_size_override: ?f32 = null,

    /// Emit an XXH32 content checksum after the end-mark. The decoder
    /// verifies it post-decode and returns error.ChecksumMismatch on
    /// any corruption (v4 #13, 2026-06-10).
    content_checksum: bool = false,
    /// v4 #19: chunk-Merkle checksum (flag bit 5) - XXH32 over the
    /// per-chunk XXH32s of the source, 4-byte trailer. Default ON:
    /// integrity costs ~1 ms/100 MB (parallel host hash) and lets the
    /// decoder KNOW when output is corrupt instead of returning wrong
    /// bytes silently. Pass false to strip it.
    chunk_checksum: bool = true,

    /// v4 #16: preset-dictionary ID (a `src/dict/dictionary.zig`
    /// built-in; flag bit 3 + 4 ID bytes in the frame header). The
    /// decoder must resolve the same ID or it rejects the frame with
    /// `error.UnknownDictionary`. Phase 1 records and enforces the ID
    /// on the wire; the match finder does not search the dictionary
    /// yet, so there is no ratio benefit until the kernel phases land.
    dictionary_id: ?u32 = null,

    /// v4 #20: emit the chunk-size table footer (flag bit 6) -
    /// 3 bytes per chunk after the end mark, before the trailers.
    /// Lets the device-resident decode path locate every chunk with
    /// one parallel read instead of the serial chunk-chain walk
    /// (~0.7 ms -> tens of us on a 100 MB frame). Decoders that
    /// predate the bit ignore the footer and decode normally.
    /// Opt-in (default off) until the VK encoder mirrors it - the
    /// cross-backend SHA gate requires both backends' default frames
    /// to stay byte-identical. Not emitted on uncompressed-body
    /// frames or when the effective chunk size is under 64 KB.
    chunk_size_table: bool = false,
};

/// Upper bound on the compressed-output size for an input of `src_len`
/// bytes. Generous; sized so the worst-case incompressible path can store
/// the source verbatim plus all frame / block / chunk / sub-chunk headers
/// and the SC tail prefix table.
pub fn compressBound(src_len: usize) usize {
    const chunk_count: usize = (src_len + lz_constants.chunk_size - 1) / lz_constants.chunk_size;
    const sub_chunks: usize = (src_len + lz_constants.sub_chunk_size - 1) / lz_constants.sub_chunk_size;
    // Per sub-chunk: 3-byte BE header + 8 initial raw bytes + 3-byte
    // literal-stream header + 256 bytes of slack for the token / off16 /
    // off32 / length headers that follow.
    const per_sub_chunk_overhead: usize = 3 + 8 + 3 + 256;
    const sc_prefix_upper_bound: usize = chunk_count * 8;
    // v4 #20 chunk-size table: 3 B per effective chunk. The table is
    // only emitted for eff_chunk >= 64 KB, so ceil(len/64K) bounds the
    // entry count at every legal sc setting.
    const chunk_table_upper_bound: usize = (src_len / 65536 + 2) * 3;
    return frame.max_header_size + 4 + 4 // (+4: v4 #19 merkle trailer)
        + chunk_count * (8 + 2 + 4)
        + sub_chunks * per_sub_chunk_overhead
        + src_len
        + 64
        + sc_prefix_upper_bound
        + chunk_table_upper_bound
        + 4; // XXH32 content checksum trailer
}

/// Compress host bytes `src` into host bytes `dst`, producing one
/// SLZ1 frame. Returns the number of bytes written to `dst`.
///
/// `dst.len` must be at least `compressBound(src.len)`. `allocator` is
/// used for small per-call host scratch (descriptor arrays, raw-LZ
/// staging); persistent device buffers live on `enc_ctx`. The caller
/// owns `enc_ctx`; the encoder grows its buffers across calls and the
/// caller is responsible for `enc_ctx.deinit(allocator)`.
///
/// Errors: every member of `CompressError`. See the per-variant
/// docs on the error set above for trigger conditions.
pub fn compressFramed(
    allocator: std.mem.Allocator,
    src: []const u8,
    dst: []u8,
    opts: Options,
    enc_ctx: *gpu_encoder.EncodeContext,
) CompressError!usize {
    return compressFramedWithIo(allocator, std.Io.failing, src, dst, opts, enc_ctx);
}

/// Same contract as `compressFramed`, plus a `std.Io` for telemetry
/// instrumentation. The `io` plumbing is consulted only when the
/// `SLZ_E2E_TIMER` / `SLZ_SPLIT_TIMER` env vars are set; passing
/// `std.Io.failing` (which `compressFramed` does) disables the timers
/// cheaply.
pub fn compressFramedWithIo(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    dst: []u8,
    opts: Options,
    enc_ctx: *gpu_encoder.EncodeContext,
) CompressError!usize {
    if (opts.level < 1 or opts.level > 5) return error.BadLevel;
    if (dst.len < compressBound(src.len)) return error.DestinationTooSmall;
    if (opts.dictionary_id) |did| {
        const dict = @import("../dict/dictionary.zig");
        if (dict.resolve(enc_ctx.registered_dicts.items, did) == null) return error.UnknownDictionary;
    }
    var frame_len = try fast_framed.compressFramedOne(allocator, io, src, dst, opts, enc_ctx);
    if (opts.content_checksum) {
        if (frame_len + 4 > dst.len) return error.DestinationTooSmall;
        const xxh = @import("../format/xxhash32.zig");
        std.mem.writeInt(u32, dst[frame_len..][0..4], xxh.xxhash32(src), .little);
        frame_len += 4;
    }
    // v4 #19: chunk-Merkle trailer (after the content trailer when both
    // are set - the decoder reads them in the same order).
    // v4 #19 v2: the device path (fast_framed) computes per-chunk
    // hashes on GPU and writes the root trailer INTO the device frame
    // before the final D2H - in that case the trailer is already in
    // `dst` and frame_len already includes it. This host block is the
    // FALLBACK (kernels unavailable), and it can only run when `src`
    // is dereferenceable: d_input_override != 0 means the caller's
    // data is device-resident and the host slice is a length-only
    // sentinel that must not be touched.
    if (opts.chunk_checksum and !enc_ctx.merkle_device_done and enc_ctx.d_input_override == 0) {
        if (frame_len + 4 > dst.len) return error.DestinationTooSmall;
        const xxh = @import("../format/xxhash32.zig");
        const eff = fast_framed.effChunkFor(src.len, opts.sc_group_size_override);
        const root = xxh.chunkMerkleRoot(allocator, src, eff) catch return error.OutOfMemory;
        std.mem.writeInt(u32, dst[frame_len..][0..4], root, .little);
        frame_len += 4;
    }
    return frame_len;
}
