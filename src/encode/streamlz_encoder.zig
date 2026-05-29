//! Top-level StreamLZ framed compressor — public API surface.
//!
//! Terminology: "sc" / "SC" = "self-contained" throughout this module.
//!
//! Library callers reach `compressFramed` and `compressBound` from this file;
//! the actual frame construction lives in `fast_framed.zig`. Levels 6-11
//! historically routed through a High codec — that path was removed during
//! the GPU-only strip, so any level outside 1-5 returns `error.BadLevel`.

const std = @import("std");
const frame = @import("../format/frame_format.zig");
const lz_constants = @import("../format/streamlz_constants.zig");
const fast_constants = @import("fast/fast_constants.zig");
const entropy_enc = @import("entropy/entropy_encoder.zig");
const EntropyOptions = entropy_enc.EntropyOptions;

const fast_framed = @import("fast_framed.zig");
const gpu_encoder = @import("../gpu/encode/driver.zig");

/// Default match-window distance written to the resolved params (1 GB).
/// The GPU LZ kernel does not consult this directly — it sizes its own
/// hash tables via `hashBitsForLevel` — but the value still flows through
/// `ParserConfig` for the CPU-side hasher used by the chain parser path.
pub const default_dictionary_size: u32 = @intCast(lz_constants.max_dictionary_size);

pub const CompressError = error{
    BadLevel,
    BadBlockSize,
    BadScGroupSize,
    DestinationTooSmall,
} || std.mem.Allocator.Error;

pub const Options = struct {
    // ── Primary options (most callers only set these) ────────────────────

    /// Compression level 1-5. Default: 1 (fastest).
    level: u8 = 1,
    /// Worker thread count. 0 = auto. The GPU encode path uses this only
    /// to size CPU-side fallback work; the kernel itself is single-stream.
    num_threads: u32 = 0,

    // ── Frame options ────────────────────────────────────────────────────

    /// Include content-size in frame header. Default: true (recommended).
    include_content_size: bool = true,
    /// Block size advertised in frame header.
    block_size: u32 = lz_constants.chunk_size,
    /// Force self-contained mode (each block independently decodable).
    self_contained: bool = false,
    /// Override SC group size (units of 256 KB chunks). 0.25 = 64 KB blocks,
    /// 1.0 = one chunk, 4.0 = four chunks. null = auto from level + GPU
    /// saturation heuristic. See `fast_framed.compressFramedOne`.
    sc_group_size_override: ?f32 = null,
    /// GPU encode mode. Currently the only supported path; kept as a flag
    /// for symmetry with the previous CPU/GPU dispatch and because the
    /// orchestrator gates its kernel launches on it.
    gpu_mode: bool = false,

    // ── Advanced (rarely changed) ───────────────────────────────────────

    /// Override hash-table bit count (0 = adaptive from input size).
    hash_bits: u32 = 0,
    /// Minimum match length override (0 = auto from level).
    min_match_length: u32 = 0,
    /// Two-phase mode (implies self_contained).
    two_phase: bool = false,
};

/// Resolved per-input parameters derived from `Options`.
pub const ResolvedParams = struct {
    engine_level: i32,
    use_entropy: bool,
    hash_bits: u6,
    /// Passed to `FastMatchHasher.init` as the hash `k` parameter — sets the
    /// Fibonacci hash multiplier. The text-detector bump that used to push
    /// this from 4 to 6 on text input was removed with the rest of the CPU
    /// codec; the GPU kernel runs its own hash tables sized via
    /// `hashBitsForLevel(level)` and does not consult this field.
    hasher_k: u32,
    /// Passed to the parser's `buildMinimumMatchLengthTable` as the acceptance
    /// threshold for match lengths.
    parser_min_match_length: u32,
    dict_size: u32,
};

pub fn resolveParams(src: []const u8, opts: Options) ResolvedParams {
    const mapped = fast_constants.mapLevel(opts.level);
    const eng = mapped.engine_level;
    const hasher_k: u32 = if (opts.min_match_length >= 4) opts.min_match_length else 4;
    const parser_min_ml: u32 = hasher_k;

    const bits = fast_constants.getHashBits(
        src.len,
        @max(eng, 2),
        opts.hash_bits,
        16,
        20,
        17,
        24,
    );

    return .{
        .engine_level = eng,
        .use_entropy = mapped.use_entropy_coding,
        .hash_bits = bits,
        .hasher_k = hasher_k,
        .parser_min_match_length = parser_min_ml,
        .dict_size = default_dictionary_size,
    };
}

/// Returns the entropy-option mask for a given user-level. Matches the
/// historical CPU-codec selection used downstream by `reencodeGpuWithEntropy`
/// when it picks between Huffman, RLE, and short-memset coders for each
/// stream. The mask drives the host-side wire-format assembly, not the GPU
/// kernels themselves.
pub fn entropyOptionsForLevel(user_level: u8) EntropyOptions {
    const mapped = fast_constants.mapLevel(user_level);
    if (!mapped.use_entropy_coding) {
        return .{ .supports_short_memset = true };
    }
    const eng = mapped.engine_level;
    if (eng >= 5) {
        return .{
            .allow_tans = true,
            .allow_rle_entropy = true,
            .allow_double_huffman = true,
            .allow_rle = true,
            .allow_multi_array = true,
            .supports_new_huffman = true,
            .supports_short_memset = true,
        };
    }
    var opts: EntropyOptions = .{
        .allow_rle_entropy = true,
        .allow_double_huffman = true,
        .allow_rle = true,
        .supports_new_huffman = true,
        .supports_short_memset = true,
    };
    // Engines 1, 2 (user L3, L4) drop RLE. Engine 4 (user L5) keeps it.
    if (eng != 3 and eng != 4) {
        opts.allow_rle = false;
        opts.allow_rle_entropy = false;
    }
    return opts;
}

/// Upper bound on the compressed-output size for an input of `src_len` bytes.
/// Generous; the worst-case incompressible path stores source verbatim plus
/// frame / block / chunk / sub-chunk headers and the SC prefix table.
pub fn compressBound(src_len: usize) usize {
    const chunk_count: usize = (src_len + lz_constants.chunk_size - 1) / lz_constants.chunk_size;
    const sub_chunks: usize = (src_len + fast_constants.sub_chunk_size - 1) / fast_constants.sub_chunk_size;
    // sub-chunk hdr + initial 8 raw bytes + per-sub-chunk literal hdr +
    // assembly slack (entropy encoder needs ~256 bytes past the literal stream
    // to write token / off16 / off32 headers without bounds-checking each
    // append).
    const per_sub_chunk_overhead: usize = 3 + 8 + 3 + 256;
    const sc_prefix_upper_bound: usize = chunk_count * 8;
    return frame.max_header_size + 4
        + chunk_count * (8 + 2 + 4)
        + sub_chunks * per_sub_chunk_overhead
        + src_len
        + 64
        + sc_prefix_upper_bound;
}

pub fn compressFramed(
    allocator: std.mem.Allocator,
    src: []const u8,
    dst: []u8,
    opts: Options,
    enc_ctx: *gpu_encoder.EncodeContext,
) CompressError!usize {
    return compressFramedWithIo(allocator, std.Io.failing, src, dst, opts, enc_ctx);
}

pub fn compressFramedWithIo(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    dst: []u8,
    opts: Options,
    enc_ctx: *gpu_encoder.EncodeContext,
) CompressError!usize {
    if (opts.level < 1 or opts.level > 5) return error.BadLevel;
    if (opts.hash_bits != 0 and (opts.hash_bits < 8 or opts.hash_bits > 24)) return error.BadLevel;
    if (dst.len < compressBound(src.len)) return error.DestinationTooSmall;
    return fast_framed.compressFramedOne(allocator, io, src, dst, opts, enc_ctx);
}
