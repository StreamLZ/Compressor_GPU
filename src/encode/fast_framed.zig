//! Fast codec frame builder — produces a single SLZ1-framed block
//! for Fast levels L1-L5.
//!
//! Terminology: "sc" / "SC" = "self-contained" throughout this module.
//!
//! Extracted from `streamlz_encoder.zig` to isolate the Fast-codec
//! frame construction from the top-level dispatch and the High-codec
//! path.  `compressFramedOne` is the sole public entry point.

const std = @import("std");
const frame = @import("../format/frame_format.zig");
const block_header = @import("../format/block_header.zig");
const lz_constants = @import("../format/streamlz_constants.zig");
const pdm = @import("../format/parallel_decode_metadata.zig");
const cleanness = @import("../decode/cross_chunk_analyzer.zig");
const fast_constants = @import("fast/fast_constants.zig");
const FastMatchHasher = @import("fast/fast_match_hasher.zig").FastMatchHasher;
const match_hasher = @import("match_hasher.zig");
const fast_enc = @import("fast/fast_lz_encoder.zig");
const entropy_enc = @import("entropy/entropy_encoder.zig");
const cost_coeffs = @import("cost_coefficients.zig");
const EntropyOptions = entropy_enc.EntropyOptions;

const MatchHasher2 = match_hasher.MatchHasher2;

const encoder = @import("streamlz_encoder.zig");
const Options = encoder.Options;
const CompressError = encoder.CompressError;
const resolveParams = encoder.resolveParams;
const entropyOptionsForLevel = encoder.entropyOptionsForLevel;
const compressBound = encoder.compressBound;

/// Re-encode a raw GPU sub-chunk payload with entropy (tANS).
/// Parses the raw format (lit/token/off16/off32/length streams),
/// applies encodeArrayU8 to token and off16 hi/lo, writes entropy output to dst.
/// Returns encoded size, or 0 on failure / no benefit.
fn reencodeGpuWithEntropy(
    allocator: std.mem.Allocator,
    raw: []const u8,
    dst: []u8,
    options: entropy_enc.EntropyOptions,
    speed_tradeoff: f32,
    initial_bytes: usize,
    src_size: usize,
) !usize {
    var rp: usize = 0;
    var wp: usize = 0;

    // Copy initial bytes (first 8 raw bytes for is_first chunks)
    if (initial_bytes > 0) {
        if (rp + initial_bytes > raw.len) return 0;
        if (wp + initial_bytes > dst.len) return 0;
        @memcpy(dst[wp..][0..initial_bytes], raw[rp..][0..initial_bytes]);
        rp += initial_bytes;
        wp += initial_bytes;
    }

    // Parse literal stream
    if (rp + 3 > raw.len) return 0;
    const lit_count: usize = (@as(usize, raw[rp]) << 16) | (@as(usize, raw[rp + 1]) << 8) | raw[rp + 2];
    rp += 3;
    if (rp + lit_count > raw.len) return 0;
    const literals = raw[rp..][0..lit_count];
    rp += lit_count;

    // Parse token stream
    if (rp + 3 > raw.len) return 0;
    const token_count: usize = (@as(usize, raw[rp]) << 16) | (@as(usize, raw[rp + 1]) << 8) | raw[rp + 2];
    rp += 3;
    if (rp + token_count > raw.len) return 0;
    const tokens = raw[rp..][0..token_count];
    rp += token_count;

    // cmd_stream2_offset: present for sub-chunks > 64KB
    var cmd_stream2_data: ?[2]u8 = null;
    if (src_size > 0x10000) {
        if (rp + 2 > raw.len) return 0;
        cmd_stream2_data = raw[rp..][0..2].*;
        rp += 2;
    }

    // Parse off16 stream
    if (rp + 2 > raw.len) return 0;
    const off16_count: usize = @as(usize, std.mem.readInt(u16, raw[rp..][0..2], .little));
    rp += 2;
    const off16_bytes = off16_count * 2;
    if (rp + off16_bytes > raw.len) return 0;
    const off16_data = raw[rp..][0..off16_bytes];
    rp += off16_bytes;

    // Parse off32 header (3 bytes) + data
    if (rp + 3 > raw.len) return 0;
    const off32_hdr = raw[rp..][0..3];
    const off32_packed: u32 = @as(u32, off32_hdr[0]) | (@as(u32, off32_hdr[1]) << 8) | (@as(u32, off32_hdr[2]) << 16);
    rp += 3;
    var off32_c1: usize = (off32_packed >> 12) & 0xFFF;
    var off32_c2: usize = off32_packed & 0xFFF;
    var off32_extra: usize = 0;
    if (off32_c1 >= 4095) { if (rp + 2 > raw.len) return 0; off32_c1 = std.mem.readInt(u16, raw[rp..][0..2], .little); rp += 2; off32_extra += 2; }
    if (off32_c2 >= 4095) { if (rp + 2 > raw.len) return 0; off32_c2 = std.mem.readInt(u16, raw[rp..][0..2], .little); rp += 2; off32_extra += 2; }
    // Scan off32 entries to compute actual byte count (3 or 4 bytes per entry)
    const off32_byte_count: usize = blk: {
        var count: usize = 0;
        var scan = rp;
        const total_entries = off32_c1 + off32_c2;
        for (0..total_entries) |_| {
            if (scan + 3 > raw.len) break :blk count;
            if (raw[scan + 2] >= 0xC0) {
                if (scan + 4 > raw.len) break :blk count;
                count += 4;
                scan += 4;
            } else {
                count += 3;
                scan += 3;
            }
        }
        break :blk count;
    };
    if (rp + off32_byte_count > raw.len) return 0;
    rp += off32_byte_count;

    // Remaining = length stream
    const length_data = raw[rp..];

    // Encode to dst

    // Literals: raw memcpy
    const lit_n = entropy_enc.encodeArrayU8Memcpy(dst[wp..], literals) catch return 0;
    wp += lit_n;

    const tok_n = entropy_enc.encodeArrayU8(allocator, dst[wp..], tokens, options, speed_tradeoff, null, 0, null) catch return 0;
    wp += tok_n;

    // Write cmd_stream2_offset if present
    if (cmd_stream2_data) |cs2d| {
        if (wp + 2 > dst.len) return 0;
        dst[wp] = cs2d[0];
        dst[wp + 1] = cs2d[1];
        wp += 2;
    }

    // Off16: split hi/lo, entropy encode each half
    if (off16_count >= 32) {
        const split_buf = allocator.alloc(u8, off16_count * 2) catch return 0;
        defer allocator.free(split_buf);
        const lo_bytes = split_buf[0..off16_count];
        const hi_bytes = split_buf[off16_count..][0..off16_count];
        for (0..off16_count) |i| {
            lo_bytes[i] = off16_data[i * 2];
            hi_bytes[i] = off16_data[i * 2 + 1];
        }
        const split_enc = allocator.alloc(u8, off16_bytes + 512) catch return 0;
        defer allocator.free(split_enc);
        const hi_n = entropy_enc.encodeArrayU8(allocator, split_enc, hi_bytes, options, speed_tradeoff, null, 0, null) catch return 0;
        const lo_n = entropy_enc.encodeArrayU8(allocator, split_enc[hi_n..], lo_bytes, options, speed_tradeoff, null, 0, null) catch return 0;
        const split_total = hi_n + lo_n;
        if (split_total < off16_bytes) {
            if (wp + 2 + split_total > dst.len) return 0;
            std.mem.writeInt(u16, dst[wp..][0..2], fast_constants.entropy_coded_16_marker, .little);
            wp += 2;
            @memcpy(dst[wp..][0..split_total], split_enc[0..split_total]);
            wp += split_total;
        } else {
            if (wp + 2 + off16_bytes > dst.len) return 0;
            std.mem.writeInt(u16, dst[wp..][0..2], @intCast(off16_count), .little);
            wp += 2;
            @memcpy(dst[wp..][0..off16_bytes], off16_data);
            wp += off16_bytes;
        }
    } else {
        if (wp + 2 + off16_bytes > dst.len) return 0;
        std.mem.writeInt(u16, dst[wp..][0..2], @intCast(off16_count), .little);
        wp += 2;
        @memcpy(dst[wp..][0..off16_bytes], off16_data);
        wp += off16_bytes;
    }

    // Off32: copy raw header + data
    if (wp + 3 + off32_extra + off32_byte_count > dst.len) return 0;
    @memcpy(dst[wp..][0..3], off32_hdr);
    wp += 3;
    if (off32_extra > 0) {
        @memcpy(dst[wp..][0..off32_extra], raw[rp - off32_byte_count - off32_extra ..][0..off32_extra]);
        wp += off32_extra;
    }
    if (off32_byte_count > 0) {
        @memcpy(dst[wp..][0..off32_byte_count], raw[rp - off32_byte_count ..][0..off32_byte_count]);
        wp += off32_byte_count;
    }

    // Length stream: copy raw
    if (wp + length_data.len > dst.len) return 0;
    @memcpy(dst[wp..][0..length_data.len], length_data);
    wp += length_data.len;

    return wp;
}

// Also need the High-framed path for L6+ dispatch inside compressFramedOne.
const high_framed = @import("high_framed.zig");

const areAllBytesEqual = block_header.areAllBytesEqual;

// ────────────────────────────────────────────────────────────
//  Parallel SC Fast compress (L1)
// ────────────────────────────────────────────────────────────

const FastScContext = struct {
    src: []const u8,
    effective_src: []const u8,
    level: u8,
    sc_flag_bits: u8,
    greedy_hash_bits: u6,
    hasher_k: u32,
    parser_config: fast_enc.ParserConfig,
    entropy_options: entropy_enc.EntropyOptions,
    dict_len: usize,
    tmp_bufs: []const []u8,
    written: []usize,
    next_group: std.atomic.Value(usize),
    error_flag: std.atomic.Value(u32),
    captured_err: std.atomic.Value(u16),
    num_chunks: usize,
    num_groups: usize,
    sc_group_size: usize,
};

fn fastScWorker(shared: *FastScContext) void {
    var greedy_hasher: ?FastMatchHasher(u32) = null;
    var chain_hasher_local: ?MatchHasher2 = null;
    defer if (greedy_hasher) |*h| h.deinit();
    defer if (chain_hasher_local) |*h| h.deinit();

    // Uses c_allocator directly (not the caller's allocator) because parallel
    // workers need thread-safe allocation. Invisible to test allocator leak
    // detection — verified manually.
    if (shared.level == 5) {
        chain_hasher_local = MatchHasher2.init(std.heap.c_allocator, shared.greedy_hash_bits) catch {
            _ = shared.captured_err.cmpxchgStrong(0, @intFromError(error.OutOfMemory), .release, .monotonic);
            _ = shared.error_flag.store(1, .release);
            return;
        };
    } else {
        greedy_hasher = FastMatchHasher(u32).init(std.heap.c_allocator, .{
            .hash_bits = shared.greedy_hash_bits,
            .min_match_length = shared.hasher_k,
        }) catch {
            _ = shared.captured_err.cmpxchgStrong(0, @intFromError(error.OutOfMemory), .release, .monotonic);
            _ = shared.error_flag.store(1, .release);
            return;
        };
    }

    while (true) {
        const group_idx = shared.next_group.fetchAdd(1, .monotonic);
        if (group_idx >= shared.num_groups) return;
        if (shared.error_flag.load(.acquire) != 0) return;

        const first_chunk = group_idx * shared.sc_group_size;
        const last_chunk = @min(first_chunk + shared.sc_group_size, shared.num_chunks);

        // Reset hasher once per group — persists across chunks within the
        // group so the parser can find matches across the group window.
        const group_start_off = shared.dict_len + first_chunk * lz_constants.chunk_size;
        const window_base_ptr: [*]const u8 = shared.effective_src.ptr + group_start_off;

        if (greedy_hasher) |*h| {
            h.reset();
            if (shared.dict_len > 0) h.preloadDictionary(shared.effective_src.ptr, shared.dict_len);
        }
        if (chain_hasher_local) |*h| {
            h.reset();
            h.setSrcBase(window_base_ptr);
            h.setBaseWithoutPreload(0);
            if (shared.dict_len > 0) h.preloadDictionary(shared.effective_src.ptr, shared.dict_len);
        }

        var ci: usize = first_chunk;
        while (ci < last_chunk) : (ci += 1) {
        const chunk_idx = ci;
        const src_off = shared.dict_len + chunk_idx * lz_constants.chunk_size;
        const block_src_len = @min(shared.effective_src.len - src_off, lz_constants.chunk_size);
        const block_src = shared.effective_src[src_off..][0..block_src_len];

        var out = shared.tmp_bufs[chunk_idx];
        var wpos: usize = 0;

        const flags0: u8 = 0x05 | shared.sc_flag_bits | 0x40;
        out[wpos] = flags0;
        out[wpos + 1] = @intFromEnum(block_header.CodecType.fast);
        wpos += 2;

        if (areAllBytesEqual(block_src)) {
            const memset_hdr: u32 = lz_constants.chunk_size_mask | (@as(u32, 1) << lz_constants.chunk_type_shift);
            std.mem.writeInt(u32, out[wpos..][0..4], memset_hdr, .little);
            wpos += 4;
            out[wpos] = block_src[0];
            wpos += 1;
            shared.written[chunk_idx] = wpos;
            continue;
        }

        const chunk_hdr_pos = wpos;
        wpos += 4;
        const chunk_payload_start = wpos;

        const speed_tradeoff = cost_coeffs.speedTradeoffFor(
            cost_coeffs.default_space_speed_tradeoff_bytes,
            shared.level >= 5,
        );
        var total_cost: f32 = 0;
        var sub_off: usize = 0;

        while (sub_off < block_src_len) {
            const round_bytes = @min(block_src_len - sub_off, fast_constants.sub_chunk_size);
            const sub_src = shared.effective_src[src_off + sub_off ..][0..round_bytes];
            const round_f: f32 = @floatFromInt(round_bytes);
            const sub_memset_cost: f32 = (round_f * cost_coeffs.memset_per_byte + cost_coeffs.memset_base) * speed_tradeoff + round_f + 3.0;

            if (round_bytes >= 32) {
                if (areAllBytesEqual(sub_src)) {
                    block_header.writeBE24(out[wpos..].ptr, @intCast(round_bytes));
                    @memcpy(out[wpos + 3 ..][0..round_bytes], sub_src);
                    wpos += round_bytes + 3;
                    total_cost += @floatFromInt(round_bytes + 3);
                    sub_off += round_bytes;
                    continue;
                }

                const sub_hdr_pos = wpos;
                wpos += 3;
                const sub_payload_start = wpos;
                const start_pos = src_off + sub_off;

                const result = switch (shared.level) {
                    5 => fast_enc.encodeSubChunkEntropyChain(4, std.heap.c_allocator, &chain_hasher_local.?, sub_src, window_base_ptr, out[sub_payload_start..], start_pos, shared.entropy_options, shared.parser_config) catch |err| {
                        _ = shared.captured_err.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
                        _ = shared.error_flag.store(1, .release);
                        return;
                    },
                    3 => fast_enc.encodeSubChunkEntropy(1, std.heap.c_allocator, &greedy_hasher.?, sub_src, window_base_ptr, out[sub_payload_start..], start_pos, shared.entropy_options, shared.parser_config) catch |err| {
                        _ = shared.captured_err.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
                        _ = shared.error_flag.store(1, .release);
                        return;
                    },
                    4 => fast_enc.encodeSubChunkEntropy(2, std.heap.c_allocator, &greedy_hasher.?, sub_src, window_base_ptr, out[sub_payload_start..], start_pos, shared.entropy_options, shared.parser_config) catch |err| {
                        _ = shared.captured_err.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
                        _ = shared.error_flag.store(1, .release);
                        return;
                    },
                    2 => fast_enc.encodeSubChunkRaw(-1, u32, std.heap.c_allocator, &greedy_hasher.?, sub_src, window_base_ptr, out[sub_payload_start..], start_pos, shared.parser_config) catch |err| {
                        _ = shared.captured_err.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
                        _ = shared.error_flag.store(1, .release);
                        return;
                    },
                    else => fast_enc.encodeSubChunkRaw(-2, u32, std.heap.c_allocator, &greedy_hasher.?, sub_src, window_base_ptr, out[sub_payload_start..], start_pos, shared.parser_config) catch |err| {
                        _ = shared.captured_err.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
                        _ = shared.error_flag.store(1, .release);
                        return;
                    },
                };

                const lz_cost: f32 = result.cost + 3.0;
                const lz_wins = !result.bail and lz_cost < sub_memset_cost and result.bytes_written > 0 and result.bytes_written < round_bytes;

                if (lz_wins) {
                    const hdr: u32 = @as(u32, @intCast(result.bytes_written)) |
                        (@as(u32, @intFromEnum(result.chunk_type)) << lz_constants.sub_chunk_type_shift) |
                        lz_constants.chunk_header_compressed_flag;
                    block_header.writeBE24(out[sub_hdr_pos..].ptr, hdr);
                    wpos = sub_payload_start + result.bytes_written;
                    total_cost += lz_cost;
                } else {
                    wpos = sub_hdr_pos;
                    const hdr: u32 = @as(u32, @intCast(round_bytes)) | lz_constants.chunk_header_compressed_flag;
                    block_header.writeBE24(out[wpos..].ptr, hdr);
                    @memcpy(out[wpos + 3 ..][0..round_bytes], sub_src);
                    wpos += 3 + round_bytes;
                    total_cost += sub_memset_cost;
                }
            } else {
                const hdr: u32 = @as(u32, @intCast(round_bytes)) | lz_constants.chunk_header_compressed_flag;
                block_header.writeBE24(out[wpos..].ptr, hdr);
                @memcpy(out[wpos + 3 ..][0..round_bytes], sub_src);
                wpos += 3 + round_bytes;
                total_cost += sub_memset_cost;
            }
            sub_off += round_bytes;
        }

        const chunk_compressed_size = wpos - chunk_payload_start;
        const block_f: f32 = @floatFromInt(block_src_len);
        const block_memset_cost: f32 = (block_f * cost_coeffs.memset_per_byte + cost_coeffs.memset_base) * speed_tradeoff + block_f + 4.0;
        if (chunk_compressed_size >= block_src_len or total_cost > block_memset_cost) {
            wpos = 0;
            const unc_flags0: u8 = 0x05 | 0x80 | shared.sc_flag_bits | 0x40;
            out[wpos] = unc_flags0;
            out[wpos + 1] = @intFromEnum(block_header.CodecType.fast);
            wpos += 2;
            @memcpy(out[wpos..][0..block_src_len], block_src);
            wpos += block_src_len;
        } else {
            const raw: u32 = @as(u32, @intCast(chunk_compressed_size - 1));
            std.mem.writeInt(u32, out[chunk_hdr_pos..][0..4], raw, .little);
        }

        shared.written[chunk_idx] = wpos;
        } // end while (ci < last_chunk)
    }
}

fn compressFastChunksParallel(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    effective_src: []const u8,
    dst_payload: []u8,
    opts: Options,
    sc_flag_bits: u8,
    greedy_hash_bits: u6,
    resolved: encoder.ResolvedParams,
    parser_config: fast_enc.ParserConfig,
    num_threads: u32,
    sc_group_size: usize,
) CompressError!usize {
    const dict_len = if (opts.dictionary) |d| d.len else 0;
    // For the parallel path, sc_group_size is passed as chunks (integer).
    // chunk iteration in the parallel path still uses lz_constants.chunk_size.
    const num_chunks = (src.len + lz_constants.chunk_size - 1) / lz_constants.chunk_size;

    const tmp_bufs = try allocator.alloc([]u8, num_chunks);
    defer {
        for (tmp_bufs) |b| if (b.len != 0) allocator.free(b);
        allocator.free(tmp_bufs);
    }
    for (tmp_bufs) |*b| b.* = &[_]u8{};
    for (tmp_bufs, 0..) |*b, i| {
        const blk_off = i * lz_constants.chunk_size;
        const blen = @min(src.len - blk_off, lz_constants.chunk_size);
        b.* = try allocator.alloc(u8, compressBound(blen));
    }

    const written = try allocator.alloc(usize, num_chunks);
    defer allocator.free(written);
    @memset(written, 0);

    const num_groups = (num_chunks + sc_group_size - 1) / sc_group_size;

    var shared: FastScContext = .{
        .src = src,
        .effective_src = effective_src,
        .level = opts.level,
        .sc_flag_bits = sc_flag_bits,
        .greedy_hash_bits = greedy_hash_bits,
        .hasher_k = resolved.hasher_k,
        .parser_config = parser_config,
        .entropy_options = entropyOptionsForLevel(opts.level),
        .dict_len = dict_len,
        .tmp_bufs = tmp_bufs,
        .written = written,
        .next_group = std.atomic.Value(usize).init(0),
        .error_flag = std.atomic.Value(u32).init(0),
        .captured_err = std.atomic.Value(u16).init(0),
        .num_chunks = num_chunks,
        .num_groups = num_groups,
        .sc_group_size = sc_group_size,
    };

    const worker_count = @min(@as(usize, num_threads), num_groups);
    if (worker_count <= 1) {
        fastScWorker(&shared);
    } else {
        var group: std.Io.Group = .init;
        for (0..worker_count) |_| {
            group.concurrent(io, fastScWorker, .{&shared}) catch |err| switch (err) {
                error.ConcurrencyUnavailable => fastScWorker(&shared),
            };
        }
        // Worker errors are captured via atomic error_flag/captured_err;
        // group.await errors are redundant and safely discarded.
        group.await(io) catch {};
    }

    if (shared.error_flag.load(.acquire) != 0) {
        const code = shared.captured_err.load(.acquire);
        if (code != 0) {
            const any_err: anyerror = @errorFromInt(code);
            const narrow: CompressError = @errorCast(any_err);
            return narrow;
        }
        return error.DestinationTooSmall;
    }

    // Assemble chunk results + SC prefix table into dst_payload.
    var dst_pos: usize = 0;
    for (0..num_chunks) |i| {
        const n = written[i];
        if (dst_pos + n > dst_payload.len) return error.DestinationTooSmall;
        @memcpy(dst_payload[dst_pos..][0..n], tmp_bufs[i][0..n]);
        dst_pos += n;
    }

    // SC prefix table: (num_chunks - 1) * 8 bytes.
    var i: usize = 1;
    while (i < num_chunks) : (i += 1) {
        const chunk_start = i * lz_constants.chunk_size;
        if (chunk_start >= src.len) break;
        const copy_size = @min(@as(usize, 8), src.len - chunk_start);
        if (dst_pos + 8 > dst_payload.len) return error.DestinationTooSmall;
        @memset(dst_payload[dst_pos..][0..8], 0);
        @memcpy(dst_payload[dst_pos..][0..copy_size], src[chunk_start..][0..copy_size]);
        dst_pos += 8;
    }

    return dst_pos;
}

/// Single-piece compress — builds one Fast-codec SLZ1 frame from
/// `src` into `dst`.  For levels 6+, delegates to the High-codec
/// frame builder.  The public `compressFramed` wrapper handles
/// multi-piece OOM retry around this function.
pub fn compressFramedOne(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    dst: []u8,
    opts: Options,
) CompressError!usize {
    // Levels 6-11 use the High codec (optimal parser + hash-based /
    // BT4 match finder). Fork here so the Fast path below stays
    // byte-exact for L1-L5.
    if (opts.level >= 6) {
        return high_framed.compressFramedHigh(allocator, io, src, dst, opts);
    }

    // ── Frame header ────────────────────────────────────────────────────
    //
    // MapLevel maps unified level (1-11) to
    // (codec, codecLevel). For Fast levels 1-5 the mapping is 1→1, 2→2,
    // 3→3, 4→5 (Fast 4 is skipped), 5→6. The STORED level in the frame
    // header is the codec-level, not the unified level. Replicate that so
    // the written byte matches the format spec exactly.
    const codec_level: u8 = switch (opts.level) {
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 5,
        5 => 6,
        else => unreachable,
    };
    var pos: usize = 0;
    const sc_grp: f32 = if (opts.gpu_mode) 0.25 else if (opts.sc_group_size_override) |ov| ov else switch (opts.level) {
        1 => @as(f32, @floatFromInt(lz_constants.default_sc_group_size)),
        2, 3, 4 => @floatFromInt(@min(high_framed.computeAdaptiveGroupSize(src.len), 16)),
        5 => @floatFromInt(high_framed.computeAdaptiveGroupSize(src.len)),
        else => @as(f32, @floatFromInt(lz_constants.default_sc_group_size)),
    };
    const hdr_len = try frame.writeHeader(dst, .{
        .codec = .fast,
        .level = codec_level,
        .block_size = opts.block_size,
        .sc_group_size = sc_grp,
        .content_size = if (opts.include_content_size) @as(u64, @intCast(src.len)) else null,
        .dictionary_id = opts.dictionary_id,
        .content_checksum = false,
    });
    pos += hdr_len;

    // ── Dictionary preload ──────────────────────────────────────────────
    // If a dictionary is provided, create a combined window: dict ++ src.
    // The encoder sees dictionary bytes as prior output, enabling matches
    // against dictionary content from the very first chunk.
    const dict = opts.dictionary orelse &[_]u8{};
    const dict_len = dict.len;
    var combined_buf: ?[]u8 = null;
    const effective_src: []const u8 = if (dict_len > 0) blk: {
        const buf = allocator.alloc(u8, dict_len + src.len) catch return error.OutOfMemory;
        @memcpy(buf[0..dict_len], dict);
        @memcpy(buf[dict_len..], src);
        combined_buf = buf;
        break :blk buf;
    } else src;
    defer if (combined_buf) |buf| allocator.free(buf);

    // ── Resolve per-input parameters ───────────────────────────────────
    const resolved = resolveParams(src, opts);
    // Per-level engine hash-bit caps
    // where `maxHashBits = level switch { -3=>13, -2=>14, -1=>16, 0|1=>17, 2=>19, _ => adaptive }`.
    const engine_level_cap: u6 = switch (resolved.engine_level) {
        -3 => 13,
        -2 => 17,
        -1 => 18,
        0, 1 => 19,
        2 => 20,
        else => resolved.hash_bits,
    };
    const greedy_hash_bits: u6 = @min(resolved.hash_bits, engine_level_cap);

    // ── Allocate the persistent hasher(s) this level needs ────────────
    //
    //   engine level ≤ -2  → FastMatchHasher<ushort>   (user L1)
    //   engine level ∈ {-1,1,2} → FastMatchHasher<uint>   (user L2, L3, L4)
    //   engine level == 4  → MatchHasher2 chain hasher (user L5)
    // Slz.MapLevel skips Fast 4 entirely, so no MatchHasher2x bucket.
    var greedy_hasher_u32: ?FastMatchHasher(u32) = null;
    defer if (greedy_hasher_u32) |*h| h.deinit();
    var chain_hasher: ?MatchHasher2 = null;
    defer if (chain_hasher) |*h| h.deinit();

    switch (opts.level) {
        5 => {
            if (opts.gpu_mode) {
                greedy_hasher_u32 = FastMatchHasher(u32).init(allocator, .{
                    .hash_bits = greedy_hash_bits,
                    .min_match_length = resolved.hasher_k,
                }) catch |err| return if (err == error.HashBitsOutOfRange) error.BadLevel else @errorCast(err);
            } else {
                chain_hasher = try MatchHasher2.init(allocator, resolved.hash_bits);
            }
        },
        else => {
            // Fast 2 (engine -1), Fast 3 (engine 1), Fast 5 (engine 2).
            greedy_hasher_u32 = FastMatchHasher(u32).init(allocator, .{
                .hash_bits = greedy_hash_bits,
                .min_match_length = resolved.hasher_k,
            }) catch |err| return if (err == error.HashBitsOutOfRange) error.BadLevel else @errorCast(err);
        },
    }

    const speed_tradeoff = cost_coeffs.speedTradeoffFor(
        cost_coeffs.default_space_speed_tradeoff_bytes,
        resolved.use_entropy,
    );
    const parser_config: fast_enc.ParserConfig = .{
        // Parser mmlt uses the UN-bumped value (reads opts.min_match_length
        // regardless of the text bump).
        .minimum_match_length = resolved.parser_min_match_length,
        .dictionary_size = resolved.dict_size,
        .speed_tradeoff = speed_tradeoff,
    };

    // Reset all hashers ONCE at the top of compressFramed. The hash table
    // is cleared once per frame and then never re-cleared.
    //
    // The greedy parser uses the hash table with positions stored in
    // WHOLE-INPUT coordinates (measured from src.ptr). Stale entries from
    // sub-chunk N−1 read during sub-chunk N give huge offsets that fail
    // the `offset <= cursor - source_block_base` bound check.
    if (greedy_hasher_u32) |*h| h.reset();
    if (chain_hasher) |*h| {
        h.reset();
        h.setSrcBase(effective_src.ptr);
        h.setBaseWithoutPreload(0);
    }

    // Pre-fill hashers with dictionary positions so the first chunk
    // can find matches against dictionary content.
    if (dict_len > 0) {
        if (greedy_hasher_u32) |*h| {
            h.preloadDictionary(effective_src.ptr, dict_len);
        }
        if (chain_hasher) |*h| {
            h.preloadDictionary(effective_src.ptr, dict_len);
        }
    }

    // ── ONE frame block wraps all internal 256 KB chunks ───────────────
    // Empty source: the stream-based compress loop
    // never enters the body (`while (bytesRead > 0)`), so it writes only the
    // frame header + end mark and returns. Match that here to keep parity
    // on zero-byte inputs.
    if (src.len == 0) {
        if (pos + 4 > dst.len) return error.DestinationTooSmall;
        frame.writeEndMark(dst[pos..]);
        pos += 4;
        return pos;
    }

    // The framed compressor calls the block compressor
    // which produces a single buffer of concatenated internal blocks, and
    // wraps the whole thing in ONE frame block header. Match that.
    const frame_block_hdr_pos: usize = pos;
    pos += 8;
    const frame_block_start: usize = pos;

    const can_compress = src.len > fast_constants.min_source_length;

    const self_contained: bool = opts.self_contained or opts.two_phase or (opts.level >= 1 and opts.level <= 4);
    const sc_flag_bit: u8 = if (self_contained) 0x10 else 0;
    const two_phase_flag_bit: u8 = if (opts.two_phase) 0x20 else 0;

    const eff_chunk_sz: usize = frame.scGroupSizeToBytes(sc_grp);
    const num_chunks: usize = if (can_compress) (src.len + @min(eff_chunk_sz, lz_constants.chunk_size) - 1) / @min(eff_chunk_sz, lz_constants.chunk_size) else 0;
    const effective_threads: u32 = if (opts.num_threads == 0) @intCast(@max(1, std.Thread.getCpuCount() catch 1)) else opts.num_threads;
    if (self_contained and can_compress and effective_threads > 1 and num_chunks > 1 and sc_grp >= 1.0 and !opts.gpu_mode) {
        const parallel_sc_grp: usize = @max(1, @as(usize, @intFromFloat(sc_grp)));
        const parallel_payload_size = try compressFastChunksParallel(
            allocator,
            io,
            src,
            effective_src,
            dst[frame_block_start..],
            opts,
            sc_flag_bit | two_phase_flag_bit,
            greedy_hash_bits,
            resolved,
            parser_config,
            effective_threads,
            parallel_sc_grp,
        );

        frame.writeBlockHeader(dst[frame_block_hdr_pos..], .{
            .compressed_size = @intCast(parallel_payload_size),
            .decompressed_size = @intCast(src.len),
            .uncompressed = false,
            .parallel_decode_metadata = false,
        });
        pos = frame_block_start + parallel_payload_size;

        // End mark.
        if (pos + 4 > dst.len) return error.DestinationTooSmall;
        frame.writeEndMark(dst[pos..]);
        pos += 4;
        return pos;
    }

    // Block/chunk/sub-chunk structure:
    //   per 256 KB block:
    //     if all-equal → 2-byte block hdr + 5-byte memset chunk hdr, done
    //     else: 2-byte block hdr + 4-byte chunk hdr placeholder + sub-chunks
    //            per sub-chunk:
    //              < 32 → 3-byte raw sub-chunk hdr + memcpy
    //              all-equal → raw memcpy (no compressed flag)
    //              else → LZ; compare lzCost vs memsetCost
    //            if totalCost > blockMemsetCost → rewind, emit uncompressed block
    //            else backfill chunk hdr
    // ── GPU compress path ────────────────────────────────────────
    // When gpu_mode and CUDA is available, compress all chunks on GPU
    // and assemble the frame from GPU results.
    if (opts.gpu_mode and can_compress) gpu_compress: {
        const use_gpu_enc = comptime blk: {
            break :blk @hasDecl(@import("build_options"), "gpu") and @import("build_options").gpu;
        };
        if (!use_gpu_enc) break :gpu_compress;

        const gpu_enc = @import("fast/gpu_encoder.zig");
        if (!gpu_enc.isAvailable()) break :gpu_compress;

        const data_src = effective_src[dict_len..];
        const eff_chunk = @min(frame.scGroupSizeToBytes(sc_grp), lz_constants.chunk_size);
        const sub_chunk_cap: usize = lz_constants.sub_chunk_size;
        const gpu_block: usize = @min(eff_chunk, sub_chunk_cap);
        const n_chunks = (data_src.len + eff_chunk - 1) / eff_chunk;
        if (n_chunks == 0) break :gpu_compress;

        // Count total GPU blocks (sub-chunks)
        var n_gpu_blocks: usize = 0;
        for (0..n_chunks) |ci| {
            const chunk_size = @min(eff_chunk, data_src.len - ci * eff_chunk);
            n_gpu_blocks += (chunk_size + gpu_block - 1) / gpu_block;
        }

        var descs = allocator.alloc(gpu_enc.CompressChunkDesc, n_gpu_blocks) catch break :gpu_compress;
        defer allocator.free(descs);
        const comp_sizes = allocator.alloc(u32, n_gpu_blocks) catch break :gpu_compress;
        defer allocator.free(comp_sizes);

        const per_block_cap = gpu_block * 3;
        var gpu_out = allocator.alloc(u8, n_gpu_blocks * per_block_cap) catch break :gpu_compress;
        defer allocator.free(gpu_out);

        // Build descriptors for each GPU block (sub-chunk)
        var bi: usize = 0;
        for (0..n_chunks) |ci| {
            const chunk_start = ci * eff_chunk;
            const chunk_size = @min(eff_chunk, data_src.len - chunk_start);
            const n_subs = (chunk_size + gpu_block - 1) / gpu_block;
            for (0..n_subs) |si| {
                const sub_start = chunk_start + si * gpu_block;
                const sub_size = @min(gpu_block, data_src.len - sub_start);
                descs[bi] = .{
                    .src_offset = @intCast(dict_len + sub_start),
                    .src_size = @intCast(sub_size),
                    .dst_offset = @intCast(bi * per_block_cap),
                    .dst_capacity = @intCast(per_block_cap),
                    .is_first = if (ci == 0 and si == 0 and dict_len == 0) @as(u32, 1) else 0,
                };
                bi += 1;
            }
        }

        if (!gpu_enc.gpuCompress(effective_src, gpu_out, descs, comp_sizes, io, opts.level))
            break :gpu_compress;

        const gpu_entropy_opts = entropyOptionsForLevel(opts.level);
        // Entropy re-encoding scratch
        const gpu_speed_tradeoff = cost_coeffs.speedTradeoffFor(
            cost_coeffs.default_space_speed_tradeoff_bytes,
            true,
        );
        const entropy_scratch: ?[]u8 = null; // GPU decoder only handles raw sub-chunks
        defer if (entropy_scratch) |s| allocator.free(s);

        // Assemble frame from GPU-compressed sub-chunks grouped into chunks
        var soff: usize = dict_len;
        var gpu_bi: usize = 0;
        for (0..n_chunks) |ci| {
            const chunk_size = @min(eff_chunk, data_src.len - ci * eff_chunk);
            const n_subs = (chunk_size + gpu_block - 1) / gpu_block;

            if (pos + 2 > dst.len) return error.DestinationTooSmall;
            const flags0: u8 = 0x05 | sc_flag_bit | two_phase_flag_bit | 0x40;

            // Check if ANY sub-chunk failed to compress
            var total_comp: usize = 0;
            var all_ok = true;
            for (0..n_subs) |si| {
                const sub_size = @min(gpu_block, chunk_size - si * gpu_block);
                const cs = comp_sizes[gpu_bi + si];
                if (cs >= sub_size) { all_ok = false; break; }
                total_comp += cs + 3;
            }

            if (!all_ok) {
                dst[pos] = flags0 | 0x80;
                dst[pos + 1] = @intFromEnum(block_header.CodecType.fast);
                pos += 2;
                @memcpy(dst[pos..][0..chunk_size], effective_src[soff..][0..chunk_size]);
                pos += chunk_size;
            } else {
                dst[pos] = flags0;
                dst[pos + 1] = @intFromEnum(block_header.CodecType.fast);
                pos += 2;

                // Reserve 4 bytes for chunk header (backfill after sub-chunks)
                if (pos + 4 > dst.len) return error.DestinationTooSmall;
                const chunk_hdr_pos = pos;
                pos += 4;

                // Write each sub-chunk (with entropy re-encoding)
                for (0..n_subs) |si| {
                    const raw_cs = comp_sizes[gpu_bi + si];
                    const raw_payload = gpu_out[(gpu_bi + si) * per_block_cap ..][0..raw_cs];
                    const init_bytes: usize = if (ci == 0 and si == 0 and dict_len == 0) 8 else 0;

                    var use_payload = raw_payload;
                    var use_cs: usize = raw_cs;

                    if (entropy_scratch) |scratch| {
                        const sub_src_size = @min(gpu_block, chunk_size - si * gpu_block);
                        const ent_cs = reencodeGpuWithEntropy(
                            allocator, raw_payload, scratch,
                            gpu_entropy_opts, gpu_speed_tradeoff, init_bytes, sub_src_size,
                        ) catch 0;
                        if (ent_cs > 0 and ent_cs < raw_cs) {
                            use_payload = scratch[0..ent_cs];
                            use_cs = ent_cs;
                        }
                    }

                    if (pos + 3 + use_cs > dst.len) return error.DestinationTooSmall;
                    const sc_hdr: u32 = @as(u32, @intCast(use_cs)) |
                        (@as(u32, 1) << lz_constants.sub_chunk_type_shift) |
                        lz_constants.chunk_header_compressed_flag;
                    block_header.writeBE24(dst[pos..].ptr, sc_hdr);
                    pos += 3;
                    @memcpy(dst[pos..][0..use_cs], use_payload);
                    pos += use_cs;
                }

                // Backfill chunk header
                const chunk_payload = pos - chunk_hdr_pos - 4;
                std.mem.writeInt(u32, dst[chunk_hdr_pos..][0..4], @intCast(chunk_payload - 1), .little);
            }
            soff += chunk_size;
            gpu_bi += n_subs;
        }

        // Tail prefix table: (n_chunks - 1) × 8 bytes, matching CPU format.
        if (self_contained and n_chunks > 1) {
            for (1..n_chunks) |ci| {
                const chunk_start = ci * eff_chunk;
                if (chunk_start >= data_src.len) break;
                const copy_size: usize = @min(@as(usize, 8), data_src.len - chunk_start);
                if (pos + 8 > dst.len) return error.DestinationTooSmall;
                @memset(dst[pos..][0..8], 0);
                @memcpy(dst[pos..][0..copy_size], data_src[chunk_start..][0..copy_size]);
                pos += 8;
            }
        }

        // Backfill frame block header (8 bytes at frame_block_hdr_pos)
        const block_payload_size = pos - frame_block_start;
        frame.writeBlockHeader(dst[frame_block_hdr_pos..], .{
            .compressed_size = @intCast(block_payload_size),
            .decompressed_size = @intCast(src.len),
            .uncompressed = false,
            .parallel_decode_metadata = false,
        });

        // End mark
        if (pos + 4 > dst.len) return error.DestinationTooSmall;
        frame.writeEndMark(dst[pos..]);
        pos += 4;
        return pos;
    }

    var src_off: usize = dict_len;
    var chunk_in_group: usize = 0;
    var group_start_off: usize = dict_len;
    while (can_compress and src_off < effective_src.len) {
        const sc_group_bytes: usize = frame.scGroupSizeToBytes(sc_grp);
        const block_src_len: usize = @min(sc_group_bytes, @min(effective_src.len - src_off, lz_constants.chunk_size));
        const block_src: []const u8 = effective_src[src_off..][0..block_src_len];

        // SC mode: reset hasher at group boundaries.
        if (self_contained and chunk_in_group == 0) {
            if (greedy_hasher_u32) |*h| h.reset();
            if (chain_hasher) |*h| {
                h.reset();
                h.setSrcBase(effective_src[src_off..].ptr);
                h.setBaseWithoutPreload(0);
            }
            group_start_off = src_off;
        }
        const window_base_ptr: [*]const u8 = if (self_contained) effective_src[group_start_off..].ptr else effective_src.ptr;

        const block_start: usize = pos;
        // For SC mode, EVERY block is a keyframe (independently decodable).
        // Keyframe when sc || first block in frame.
        const keyframe = self_contained or src_off == dict_len;

        // ── Write 2-byte block header (compressed) ──────────────────────
        if (pos + 2 > dst.len) return error.DestinationTooSmall;
        var flags0: u8 = 0x05 | sc_flag_bit | two_phase_flag_bit;
        if (keyframe) flags0 |= 0x40;
        dst[pos] = flags0;
        dst[pos + 1] = @intFromEnum(block_header.CodecType.fast);
        pos += 2;

        // ── Block-level AreAllBytesEqual → memset chunk header ─────────
        //
        if (areAllBytesEqual(block_src)) {
            if (pos + 4 + 1 > dst.len) return error.DestinationTooSmall;
            const memset_hdr: u32 = lz_constants.chunk_size_mask | (@as(u32, 1) << lz_constants.chunk_type_shift);
            std.mem.writeInt(u32, dst[pos..][0..4], memset_hdr, .little);
            pos += 4;
            dst[pos] = block_src[0];
            pos += 1;
            src_off += block_src_len;
            continue;
        }

        // ── 4-byte chunk header placeholder ────────────────────────────
        if (pos + 4 > dst.len) return error.DestinationTooSmall;
        const chunk_hdr_pos: usize = pos;
        pos += 4;
        const chunk_payload_start: usize = pos;

        // ── Iterate sub-chunks ─────────────────────────────────────────
        // Per sub-chunk:
        //   * < 32 bytes → raw-flag sub-chunk header + raw bytes
        //   * all-equal → raw memcpy (no compressed flag)
        //   * else → LZ encode + cost-based 3-way decision
        var total_cost: f32 = 0;
        var sub_off: usize = 0;

        while (sub_off < block_src_len) {
            const round_bytes: usize = @min(block_src_len - sub_off, fast_constants.sub_chunk_size);
            const sub_src: []const u8 = effective_src[src_off + sub_off ..][0..round_bytes];

            const round_f: f32 = @floatFromInt(round_bytes);
            const sub_memset_cost: f32 =
                (round_f * cost_coeffs.memset_per_byte + cost_coeffs.memset_base) *
                speed_tradeoff +
                round_f + 3.0;

            if (round_bytes >= 32) {
                if (areAllBytesEqual(sub_src)) {
                    // Plain memcpy: 3-byte BE header + raw bytes, with the
                    // compressed flag CLEAR (size ≤ 18 bits).
                    if (pos + round_bytes + 3 > dst.len) return error.DestinationTooSmall;
                    block_header.writeBE24(dst[pos..].ptr, @intCast(round_bytes));
                    @memcpy(dst[pos + 3 ..][0..round_bytes], sub_src);
                    pos += round_bytes + 3;
                    // Raw memcpy cost = count + 3.
                    total_cost += @floatFromInt(round_bytes + 3);
                    sub_off += round_bytes;
                    continue;
                }

                // ── LZ trial encode ────────────────────────────────────
                const sub_hdr_pos: usize = pos;
                pos += 3;
                const sub_payload_start: usize = pos;

                const start_position_for_sub: usize = src_off + sub_off;
                const entropy_options = entropyOptionsForLevel(opts.level);
                const result = try if (opts.gpu_mode) switch (opts.level) {
                    1 => fast_enc.encodeSubChunkRaw(-2, u32, allocator, &greedy_hasher_u32.?, sub_src, window_base_ptr, dst[sub_payload_start..], start_position_for_sub, parser_config),
                    2 => fast_enc.encodeSubChunkRaw(-1, u32, allocator, &greedy_hasher_u32.?, sub_src, window_base_ptr, dst[sub_payload_start..], start_position_for_sub, parser_config),
                    3 => fast_enc.encodeSubChunkRaw(1, u32, allocator, &greedy_hasher_u32.?, sub_src, window_base_ptr, dst[sub_payload_start..], start_position_for_sub, parser_config),
                    4 => fast_enc.encodeSubChunkRaw(2, u32, allocator, &greedy_hasher_u32.?, sub_src, window_base_ptr, dst[sub_payload_start..], start_position_for_sub, parser_config),
                    5 => fast_enc.encodeSubChunkRaw(4, u32, allocator, &greedy_hasher_u32.?, sub_src, window_base_ptr, dst[sub_payload_start..], start_position_for_sub, parser_config),
                    else => unreachable,
                } else switch (opts.level) {
                    1 => fast_enc.encodeSubChunkRaw(-2, u32, allocator, &greedy_hasher_u32.?, sub_src, window_base_ptr, dst[sub_payload_start..], start_position_for_sub, parser_config),
                    2 => fast_enc.encodeSubChunkRaw(-1, u32, allocator, &greedy_hasher_u32.?, sub_src, window_base_ptr, dst[sub_payload_start..], start_position_for_sub, parser_config),
                    3 => fast_enc.encodeSubChunkEntropy(1, allocator, &greedy_hasher_u32.?, sub_src, window_base_ptr, dst[sub_payload_start..], start_position_for_sub, entropy_options, parser_config),
                    4 => fast_enc.encodeSubChunkEntropy(2, allocator, &greedy_hasher_u32.?, sub_src, window_base_ptr, dst[sub_payload_start..], start_position_for_sub, entropy_options, parser_config),
                    5 => fast_enc.encodeSubChunkEntropyChain(4, allocator, &chain_hasher.?, sub_src, window_base_ptr, dst[sub_payload_start..], start_position_for_sub, entropy_options, parser_config),
                    else => unreachable,
                };
                const lz_cost: f32 = result.cost + 3.0; // +3 for sub-chunk header

                const lz_wins = !result.bail and
                    lz_cost < sub_memset_cost and
                    result.bytes_written > 0 and
                    result.bytes_written < round_bytes;

                if (lz_wins) {
                    // LZ path: backfill 3-byte sub-chunk header.
                    const hdr: u32 = @as(u32, @intCast(result.bytes_written)) |
                        (@as(u32, @intFromEnum(result.chunk_type)) << lz_constants.sub_chunk_type_shift) |
                        lz_constants.chunk_header_compressed_flag;
                    block_header.writeBE24(dst[sub_hdr_pos..].ptr, hdr);
                    pos = sub_payload_start + result.bytes_written;
                    total_cost += lz_cost;
                } else {
                    // Memset (uncompressed) path: 3-byte header with
                    // compressed flag set, size = round_bytes, then raw
                    // bytes verbatim. Rewind the LZ-written payload.
                    pos = sub_hdr_pos;
                    if (pos + 3 + round_bytes > dst.len) return error.DestinationTooSmall;
                    const hdr: u32 = @as(u32, @intCast(round_bytes)) | lz_constants.chunk_header_compressed_flag;
                    block_header.writeBE24(dst[pos..].ptr, hdr);
                    @memcpy(dst[pos + 3 ..][0..round_bytes], sub_src);
                    pos += 3 + round_bytes;
                    total_cost += sub_memset_cost;
                }
            } else {
                // round_bytes < 32: too small to compress.
                if (pos + 3 + round_bytes > dst.len) return error.DestinationTooSmall;
                const hdr: u32 = @as(u32, @intCast(round_bytes)) | lz_constants.chunk_header_compressed_flag;
                block_header.writeBE24(dst[pos..].ptr, hdr);
                @memcpy(dst[pos + 3 ..][0..round_bytes], sub_src);
                pos += 3 + round_bytes;
                total_cost += sub_memset_cost;
            }

            sub_off += round_bytes;
        }

        // ── Block-level cost decision ──────────────────────────────────
        // Rewrite
        // the whole block as uncompressed if either the compressed chunk
        // didn't shrink the payload OR its cost exceeded the block-level
        // memset cost.
        const chunk_compressed_size: usize = pos - chunk_payload_start;
        const block_f: f32 = @floatFromInt(block_src_len);
        const block_memset_cost: f32 =
            (block_f * cost_coeffs.memset_per_byte + cost_coeffs.memset_base) *
            speed_tradeoff +
            block_f + 4.0; // +4 = ChunkHeaderSize
        const should_bail = chunk_compressed_size >= block_src_len or total_cost > block_memset_cost;
        if (should_bail) {
            pos = block_start;
            if (pos + 2 + block_src_len > dst.len) return error.DestinationTooSmall;
            var unc_flags0: u8 = 0x05 | 0x80 | sc_flag_bit | two_phase_flag_bit;
            if (keyframe) unc_flags0 |= 0x40;
            dst[pos] = unc_flags0;
            dst[pos + 1] = @intFromEnum(block_header.CodecType.fast);
            pos += 2;
            @memcpy(dst[pos..][0..block_src_len], block_src);
            pos += block_src_len;
        } else {
            // v2 chunk header: bits [17:0] = compressed_size - 1,
            // bits [19:18] = type (0 = normal), bit [20] = has_cross_chunk_match.
            // We conservatively write the bit as 0 ("may have cross-chunk refs")
            // for now — a true-value computation would require tracking
            // min_match_src across the Fast parser variants and plumbing it up
            // into EncodeResult. Decoders treat 0 as "check the sidecar"; a
            // future encoder can set the bit to 1 when it knows the chunk is
            // cross-chunk-free, enabling a fast dispatch path.
            const has_cross_chunk_match_bit: u32 = 0;
            const raw: u32 = @as(u32, @intCast(chunk_compressed_size - 1)) | has_cross_chunk_match_bit;
            std.mem.writeInt(u32, dst[chunk_hdr_pos..][0..4], raw, .little);
        }

        src_off += block_src_len;
        chunk_in_group += 1;
        const sc_grp_int: usize = @max(1, @as(usize, @intFromFloat(sc_grp)));
        if (chunk_in_group >= sc_grp_int) chunk_in_group = 0;
    }

    // SC mode: append a prefix table of (num_chunks - 1) * 8 bytes at the
    // end of the frame block payload. Each entry holds the first 8 bytes of
    // chunks 1..N-1. The parallel decompressor uses these to restore the
    // corrupted first 8 bytes of each per-worker-decoded chunk.
    if (self_contained and can_compress) {
        var i: usize = 1;
        while (i < num_chunks) : (i += 1) {
            const chunk_start = i * @min(eff_chunk_sz, lz_constants.chunk_size);
            if (chunk_start >= src.len) break;
            const copy_size: usize = @min(@as(usize, 8), src.len - chunk_start);
            if (pos + 8 > dst.len) return error.DestinationTooSmall;
            // Zero-fill the 8-byte slot before copying, so the trailing
            // bytes past `copy_size` are deterministic.
            @memset(dst[pos..][0..8], 0);
            @memcpy(dst[pos..][0..copy_size], src[chunk_start..][0..copy_size]);
            pos += 8;
        }
    }

    // Frame block fallback: if compressed total didn't beat uncompressed
    // (or input too small), rewrite the frame block as one uncompressed
    // frame block.
    const frame_block_compressed_size = pos - frame_block_start;
    if (!can_compress or frame_block_compressed_size >= src.len) {
        pos = frame_block_start;
        if (pos + src.len > dst.len) return error.DestinationTooSmall;
        frame.writeBlockHeader(dst[frame_block_hdr_pos..], .{
            .compressed_size = @intCast(src.len),
            .decompressed_size = @intCast(src.len),
            .uncompressed = true,
            .parallel_decode_metadata = false,
        });
        @memcpy(dst[pos..][0..src.len], src);
        pos += src.len;
    } else {
        frame.writeBlockHeader(dst[frame_block_hdr_pos..], .{
            .compressed_size = @intCast(frame_block_compressed_size),
            .decompressed_size = @intCast(src.len),
            .uncompressed = false,
            .parallel_decode_metadata = false,
        });
    }

    // ── v2 parallel-decode sidecar block ───────────────────────────────
    //
    // For Fast L1-L4 compressed data (opts.level 1..4), compute the
    // cross-sub-chunk closure from the just-written compressed bytes
    // and append a sidecar block that parallel decoders can consume.
    // The sidecar lives BETWEEN the last compressed block and the end
    // mark, with the new `parallel_decode_metadata_flag` on its outer
    // block header so serial decoders skip it cleanly.
    //
    // buildPpocSidecar walks the just-written frame (header + blocks)
    // using `src` as the reference for literal-leaf byte values — the
    // original input, which we already have. No decode pass needed.
    //
    // Encoder failures (OOM from the analyzer's byte_earliest /
    // producer_map allocations) are swallowed: we silently omit the
    // sidecar and leave the frame flag clear. The frame still decodes
    // correctly via the serial path; only parallel-decode acceleration
    // is forfeited for that specific compress call.
    // Sidecar emission is scoped to Fast L1-L4 (the levels the PoC
    // closure analysis and parallel-decode path were designed for).
    // L5 uses the lazy chain parser with a very different token
    // distribution — the closure frequently exceeds millions of
    // entries on text input, the sidecar grows to 40%+ of the main
    // payload, and the walker's overcopy heuristics are unreliable
    // on its token layout. L5 decompresses correctly via the serial
    // path; extending parallel-decode support to it is a separate
    // task (probably needs a distinct codepath).
    if (opts.emit_parallel_decode_metadata and !self_contained and opts.level >= 1 and opts.level <= 5 and can_compress) {
        const sidecar_result = if (dict_len > 0)
            cleanness.buildPpocSidecarWithDict(allocator, dst[0..pos], effective_src, dict_len)
        else
            cleanness.buildPpocSidecar(allocator, dst[0..pos], src);
        if (sidecar_result) |*sc| {
            var sidecar = sc.*;
            defer sidecar.deinit(allocator);
            if (sidecar.match_ops.items.len > 0 or sidecar.literal_bytes.items.len > 0) {
                // Convert the analyzer's ArrayLists to pdm's
                // slice-based view. The analyzer's MatchOp and
                // LiteralByte structs have the same layout as pdm's,
                // so we copy via field names (no bitcast).
                //
                // Match ops are already emitted by the analyzer in
                // cmd_stream order (= file position order, which is
                // monotonically increasing in target_start). That's
                // exactly what the v2 sidecar writer wants, so no
                // re-sort needed.
                var tmp_match_ops = try allocator.alloc(pdm.MatchOp, sidecar.match_ops.items.len);
                defer allocator.free(tmp_match_ops);
                for (sidecar.match_ops.items, 0..) |op, i| {
                    tmp_match_ops[i] = .{
                        .target_start = op.target_start,
                        .src_start = op.src_start,
                        .length = op.length,
                    };
                }

                // Literal bytes come from two unrelated sources in
                // the analyzer (the closure BFS's literal leaves and
                // the walker's overcopy leaves), so they're NOT in
                // sorted order. The v2 writer assumes sorted input
                // (for run detection), so we sort before emitting.
                var tmp_literal_bytes = try allocator.alloc(pdm.LiteralByte, sidecar.literal_bytes.items.len);
                defer allocator.free(tmp_literal_bytes);
                for (sidecar.literal_bytes.items, 0..) |lit, i| {
                    tmp_literal_bytes[i] = .{
                        .position = lit.position,
                        .byte_value = lit.byte_value,
                    };
                }
                std.mem.sort(pdm.LiteralByte, tmp_literal_bytes, {}, struct {
                    fn lessThan(_: void, a: pdm.LiteralByte, b: pdm.LiteralByte) bool {
                        return a.position < b.position;
                    }
                }.lessThan);

                // Per-chunk XOR folds for parallel decode integrity verification.
                const chunk_xor_folds = allocator.alloc(u64, num_chunks) catch null;
                defer if (chunk_xor_folds) |folds| allocator.free(folds);
                const xor_folds_slice: []const u64 = if (chunk_xor_folds) |folds| blk: {
                    for (folds, 0..) |*f, ci| {
                        const cs = ci * lz_constants.chunk_size;
                        const ce = @min(cs + lz_constants.chunk_size, src.len);
                        f.* = pdm.xorFoldChunk(src[cs..ce]);
                    }
                    break :blk folds;
                } else &[_]u64{};

                const body_size = pdm.serializedBodySize(tmp_match_ops, tmp_literal_bytes, xor_folds_slice);
                // Outer block header (8 bytes) + body.
                if (pos + 8 + body_size > dst.len) {
                    // Out of output budget — skip the sidecar rather
                    // than failing the whole compress. Frame is still
                    // valid without it.
                } else {
                    // Write the 8-byte outer block header.
                    frame.writeBlockHeader(dst[pos..], .{
                        .compressed_size = @intCast(body_size),
                        .decompressed_size = 0,
                        .uncompressed = false,
                        .parallel_decode_metadata = true,
                    });
                    pos += 8;

                    // Write the sidecar body.
                    const body_written = try pdm.writeBlockBody(
                        dst[pos..],
                        tmp_match_ops,
                        tmp_literal_bytes,
                        xor_folds_slice,
                    );
                    pos += body_written;

                    // Patch the frame header flags byte to advertise the
                    // sidecar's presence. The flags byte is at offset 5
                    // (after magic+version). We use a bitwise-or so we
                    // don't clobber the other flag bits the encoder set.
                    dst[5] |= @as(u8, 1) << 4; // parallel_decode_metadata_present
                }
            }
        } else |_| {
            // buildPpocSidecar failed (probably OOM from the 400+ MB
            // byte_earliest + producer_map). Silently continue without
            // a sidecar — the frame is still correct, just slower to
            // parallel-decode.
        }
    }

    // ── End mark ───────────────────────────────────────────────────────
    if (pos + 4 > dst.len) return error.DestinationTooSmall;
    frame.writeEndMark(dst[pos..]);
    pos += 4;

    return pos;
}

// ────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────

test "compressFramedOne: empty input roundtrip" {
    const allocator = std.testing.allocator;
    var dst: [256]u8 = undefined;
    const n = try compressFramedOne(allocator, std.testing.io, &.{}, &dst, .{ .level = 1 });
    try std.testing.expect(n > 0);
    try std.testing.expect(n < 64);
    const decoder = @import("../decode/streamlz_decoder.zig");
    var dec_buf: [64]u8 = undefined;
    const dec_n = try decoder.decompressFramed(dst[0..n], &dec_buf);
    try std.testing.expectEqual(@as(usize, 0), dec_n);
}

test "compressFramedOne: all-equal bytes compresses small" {
    const allocator = std.testing.allocator;
    const src = try allocator.alloc(u8, 4096);
    defer allocator.free(src);
    @memset(src, 0xAA);
    const bound = compressBound(src.len);
    const dst = try allocator.alloc(u8, bound);
    defer allocator.free(dst);
    const n = try compressFramedOne(allocator, std.testing.io, src, dst, .{ .level = 1 });
    try std.testing.expect(n < 200);
    const decoder = @import("../decode/streamlz_decoder.zig");
    const dec = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dec);
    const dec_n = try decoder.decompressFramed(dst[0..n], dec);
    try std.testing.expectEqual(src.len, dec_n);
    try std.testing.expectEqualSlices(u8, src, dec[0..dec_n]);
}
