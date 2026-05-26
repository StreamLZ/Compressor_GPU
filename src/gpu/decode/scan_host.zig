//! Host-side decode-scan: walks the compressed block on the CPU and
//! emits HuffDecChunkDesc / RawOff16Desc arrays for the GPU pre-decode
//! kernels.
//!
//! This is the scanner used by the H2D entry point (the caller's
//! compressed bytes start on host disk / in host memory). The D2D entry
//! point forces the GPU scan (`scan_gpu.zig`) because the CPU has no
//! readable copy of the bytes. Either scanner produces byte-equivalent
//! post-compaction descriptor arrays, so the rest of the pipeline does
//! not need to know which one ran.
//!
//! Also serves as the fallback if `scan_parse_fn` is missing or the GPU
//! scan returns null at runtime.

const std = @import("std");

const d = @import("descriptors.zig");

/// 24-bit big-endian read from a slice at `off`. Mirrors `readBE24` in
/// common/gpu_byteio.cuh. Caller must have verified `off + 3 <= src.len`.
fn readBE24(src: []const u8, off: u32) u32 {
    return (@as(u32, src[off]) << 16) |
        (@as(u32, src[off + 1]) << 8) |
        @as(u32, src[off + 2]);
}

pub fn scanForEntropyChunks(
    chunk_descs: []const d.ChunkDesc,
    compressed_block: []const u8,
    sub_chunk_cap: u32,
    raw_off16_descs: []d.RawOff16Desc,
    huff_lit_descs: []d.HuffDecChunkDesc,
    huff_tok_descs: []d.HuffDecChunkDesc,
    huff_off16hi_descs: []d.HuffDecChunkDesc,
    huff_off16lo_descs: []d.HuffDecChunkDesc,
    io: ?std.Io,
) d.ScanResult {
    var num_raw: u32 = 0;
    var num_huff_lit: u32 = 0;
    var num_huff_tok: u32 = 0;
    var num_huff_hi: u32 = 0;
    var num_huff_lo: u32 = 0;
    var cur_sub_idx: u32 = 0; // global sub-chunk index — mirrors driver prefix sum
    const cap_safe: u32 = if (sub_chunk_cap == 0) 65536 else sub_chunk_cap;

    // SLZ_E2E_TIMER: per-sub-chunk walk timing. Measured ~0.5-0.9 ms on
    // the dev desktop — the scan is not a decode bottleneck (decode is
    // PCIe-bound); SLZ_GPU_SCAN moves it to the GPU mainly for the
    // device-resident D2D path, not for speed.
    const scan_dbg = std.c.getenv("SLZ_E2E_TIMER") != null;
    const t_scan0 = if (scan_dbg)
        (if (io) |iv| std.Io.Clock.awake.now(iv) else null)
    else
        null;
    var dbg_subchunks: u32 = 0;

    for (chunk_descs) |ch| {
        // Compute expected n_subs for this chunk so cur_sub_idx stays in sync
        // with the driver's first_subchunk_idx prefix sum even if we skip.
        const n_subs_expected: u32 = if (ch.flags != 0 or ch.decomp_size == 0) 1
            else (ch.decomp_size + cap_safe - 1) / cap_safe;
        const chunk_first_sub: u32 = cur_sub_idx;
        // Default: advance by full expected count (we'll override only if we
        // successfully walk every sub-chunk).
        defer cur_sub_idx = chunk_first_sub + n_subs_expected;

        if (ch.flags != 0) continue;
        if (ch.decomp_size == 0) continue;
        if (ch.src_offset >= compressed_block.len) continue;

        const chunk_end = @min(ch.src_offset + ch.comp_size, @as(u32, @intCast(compressed_block.len)));
        const chunk_src = compressed_block[ch.src_offset..chunk_end];
        if (chunk_src.len < 3) continue;

        // Walk all sub-chunks within this chunk.
        var sub_pos: u32 = 0;
        var remaining_decomp: u32 = ch.decomp_size;
        var sub_local_idx: u32 = 0;
        while (remaining_decomp > 0 and sub_pos + 3 <= chunk_src.len) {
            const sub_hdr: u32 = readBE24(chunk_src, sub_pos);
            if ((sub_hdr & 0x800000) == 0) break; // not LZ
            const sc_comp_size: u32 = sub_hdr & 0x7FFFF;
            const sub_end = sub_pos + 3 + sc_comp_size;
            if (sub_end > chunk_src.len) break;
            const sub_decomp: u32 = @min(remaining_decomp, cap_safe);

            // Global sub-chunk index for this sub-chunk's entropy descriptors.
            // Slot is ENTROPY_SCRATCH_SLOT_BYTES (128KB) so the largest
            // sub-chunk fits; off16-lo lives at +OFF16_HILO_SPLIT_OFFSET
            // (64KB) within the slot.
            const sub_idx: u32 = chunk_first_sub + sub_local_idx;
            const sub_dst_off: usize = @as(usize, sub_idx) * @as(usize, @intCast(d.ENTROPY_SCRATCH_SLOT_BYTES));

            // 8 init bytes only on the very first sub-chunk of the FRAME
            // (sub_idx == 0). All later sub-chunks (including each chunk's
            // own first sub-chunk) get their first 8 bytes restored from
            // the SC prefix table post-decode, not from the stream.
            const init_b: u32 = if (sub_idx == 0) 8 else 0;
            var pos: u32 = sub_pos + 3 + init_b;
            const sub_payload_end: u32 = sub_end;

            walk: {
                if (pos >= sub_payload_end) break :walk;

                // ── Stream 1: Literals ──
                const lit_first = chunk_src[pos];
                const lit_type = (lit_first >> d.CHUNK_TYPE_SHIFT) & d.CHUNK_TYPE_MASK;
                if (lit_type == 4) {
                    // Huffman literal stream. Payload after the header is
                    // [128 B weights][9 B sub-header][4 streams].
                    parseHuffHeader(chunk_src, pos, ch.src_offset, sub_dst_off,
                                    huff_lit_descs, &num_huff_lit);
                }
                const lit_next = skipStreamHeader(chunk_src, pos);
                if (lit_next == null) break :walk;
                pos = lit_next.?;

                if (pos >= sub_payload_end) break :walk;

                // ── Stream 2: Tokens (command stream) ──
                const tok_first = chunk_src[pos];
                const tok_type = (tok_first >> d.CHUNK_TYPE_SHIFT) & d.CHUNK_TYPE_MASK;
                if (tok_type == 4) {
                    // Huffman token stream — record sub_dst_off; driver adds
                    // tok_offset later when merging into the unified descriptor array.
                    parseHuffHeader(chunk_src, pos, ch.src_offset, sub_dst_off,
                                    huff_tok_descs, &num_huff_tok);
                }
                const tok_next = skipStreamHeader(chunk_src, pos);
                if (tok_next == null) break :walk;
                pos = tok_next.?;

                if (pos >= sub_payload_end) break :walk;

                // Skip cmd_stream2_offset if sub-chunk > 64KB
                if (sub_decomp > 0x10000) {
                    if (pos + 2 > sub_payload_end) break :walk;
                    pos += 2;
                }

                // ── Off16 stream ──
                if (pos + 2 > sub_payload_end) break :walk;
                const off16_count: u32 = @as(u32, chunk_src[pos]) | (@as(u32, chunk_src[pos + 1]) << 8);
                if (off16_count != 0xFFFF) break :walk; // not entropy-coded
                pos += 2;

                if (pos >= sub_payload_end) break :walk;

                // ── Off16 hi stream ──
                const hi_first = chunk_src[pos];
                const hi_type = (hi_first >> d.CHUNK_TYPE_SHIFT) & d.CHUNK_TYPE_MASK;
                if (hi_type == 0) {
                    const raw_info = parseType0StreamInfo(chunk_src, pos);
                    if (raw_info.data_offset != 0 and num_raw < raw_off16_descs.len) {
                        raw_off16_descs[num_raw] = .{
                            .src_offset = ch.src_offset + raw_info.data_offset,
                            .size = raw_info.size,
                            .gpu_offset = @intCast(sub_dst_off),
                        };
                        num_raw += 1;
                    }
                } else if (hi_type == 4) {
                    parseHuffHeader(chunk_src, pos, ch.src_offset, sub_dst_off,
                                    huff_off16hi_descs, &num_huff_hi);
                }
                const hi_next = skipStreamHeader(chunk_src, pos);
                if (hi_next == null) break :walk;
                pos = hi_next.?;

                if (pos >= sub_payload_end) break :walk;

                // ── Off16 lo stream ──
                const lo_first = chunk_src[pos];
                const lo_type = (lo_first >> d.CHUNK_TYPE_SHIFT) & d.CHUNK_TYPE_MASK;
                if (lo_type == 0) {
                    const raw_info = parseType0StreamInfo(chunk_src, pos);
                    if (raw_info.data_offset != 0 and num_raw < raw_off16_descs.len) {
                        raw_off16_descs[num_raw] = .{
                            .src_offset = ch.src_offset + raw_info.data_offset,
                            .size = raw_info.size,
                            .gpu_offset = @intCast(sub_dst_off + d.OFF16_HILO_SPLIT_OFFSET),
                        };
                        num_raw += 1;
                    }
                } else if (lo_type == 4) {
                    // Huff lo stream → scratch slot + OFF16_HILO_SPLIT_OFFSET
                    // (lo half of off16 slot). Encode sub_dst_off + offset here;
                    // merge phase adds off16_offset.
                    parseHuffHeader(chunk_src, pos, ch.src_offset, sub_dst_off + d.OFF16_HILO_SPLIT_OFFSET,
                                    huff_off16lo_descs, &num_huff_lo);
                }
            } // walk

            // Advance to the next sub-chunk
            sub_pos = sub_end;
            remaining_decomp -= sub_decomp;
            sub_local_idx += 1;
            dbg_subchunks += 1;
        } // while sub-chunks
    } // for chunks

    if (t_scan0) |t0| if (io) |iv| {
        const el: i64 = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
        std.debug.print("  [scan] {d} chunks, {d} sub-chunks, {d} raw-off16, {d:.3} ms ({d:.0} ns/sub-chunk)\n", .{
            chunk_descs.len, dbg_subchunks, num_raw,
            @as(f64, @floatFromInt(el)) / 1e6,
            @as(f64, @floatFromInt(el)) / @as(f64, @floatFromInt(@max(dbg_subchunks, 1))),
        });
    };

    return .{
        .num_raw_off16 = num_raw,
        .num_huff_lit = num_huff_lit,
        .num_huff_tok = num_huff_tok,
        .num_huff_off16hi = num_huff_hi,
        .num_huff_off16lo = num_huff_lo,
    };
}

/// Parse a chunk_type=4 Huffman literal header at `lit_off` within chunk_src.
/// Writes a HuffDecChunkDesc whose in_offset points at the FULL payload
/// (128 B weights + 9 B sub-header + 4 streams). lut_offset is assigned by
/// index — each descriptor owns HUFF_LUT_ENTRIES contiguous LUT entries.
fn parseHuffHeader(
    chunk_src: []const u8,
    lit_off: u32,
    src_offset_base: u32,
    dst_offset: usize,
    huff_descs_out: []d.HuffDecChunkDesc,
    num_huff: *u32,
) void {
    if (lit_off >= chunk_src.len) return;
    const first_byte = chunk_src[lit_off];
    var comp_size: u32 = 0;
    var dst_size: u32 = 0;
    var payload_off: u32 = 0;
    if (first_byte >= 0x80) {
        if (lit_off + 3 > chunk_src.len) return;
        const bits: u32 = readBE24(chunk_src, lit_off);
        comp_size = bits & 0x3FF;
        dst_size = comp_size + ((bits >> 10) & 0x3FF) + 1;
        payload_off = src_offset_base + lit_off + 3;
    } else {
        if (lit_off + 5 > chunk_src.len) return;
        const bits: u32 = (@as(u32, chunk_src[lit_off + 1]) << 24) |
            (@as(u32, chunk_src[lit_off + 2]) << 16) |
            (@as(u32, chunk_src[lit_off + 3]) << 8) |
            @as(u32, chunk_src[lit_off + 4]);
        comp_size = bits & 0x3FFFF;
        dst_size = (((bits >> 18) | (@as(u32, chunk_src[lit_off]) << 14)) & 0x3FFFF) + 1;
        payload_off = src_offset_base + lit_off + 5;
    }
    if (num_huff.* >= huff_descs_out.len) return;
    huff_descs_out[num_huff.*] = .{
        .in_offset = payload_off,
        .in_size = comp_size,
        .out_offset = @intCast(dst_offset),
        .out_size = dst_size,
        .lut_offset = num_huff.* * @as(u32, @intCast(d.HUFF_LUT_ENTRIES)),
    };
    num_huff.* += 1;
}

/// Parse-helper return type for type 0 (memcpy) stream headers. Local
/// to scan_host (the only consumer; previously lived in descriptors.zig).
const Type0Info = struct { data_offset: u32, size: u32 };

/// Parse a type 0 (memcpy) stream header, returning the data offset
/// (relative to chunk start) and the raw byte count.
fn parseType0StreamInfo(chunk_src: []const u8, pos: u32) Type0Info {
    if (pos >= chunk_src.len) return .{ .data_offset = 0, .size = 0 };
    const first_byte = chunk_src[pos];
    if (first_byte >= 0x80) {
        if (pos + 2 > chunk_src.len) return .{ .data_offset = 0, .size = 0 };
        const sz: u32 = ((@as(u32, chunk_src[pos]) << 8) | @as(u32, chunk_src[pos + 1])) & 0xFFF;
        return .{ .data_offset = pos + 2, .size = sz };
    } else {
        if (pos + 3 > chunk_src.len) return .{ .data_offset = 0, .size = 0 };
        const sz: u32 = readBE24(chunk_src, pos);
        return .{ .data_offset = pos + 3, .size = sz };
    }
}

/// Skip past an entropy-coded stream header + payload, returning
/// the new position, or null if the data is truncated.
fn skipStreamHeader(chunk_src: []const u8, pos: u32) ?u32 {
    if (pos >= chunk_src.len) return null;
    const first_byte = chunk_src[pos];
    const ct = (first_byte >> d.CHUNK_TYPE_SHIFT) & d.CHUNK_TYPE_MASK;

    if (ct == 0) {
        // Type 0: memcpy, 2 or 3 byte header
        if (first_byte >= 0x80) {
            if (pos + 2 > chunk_src.len) return null;
            const sz: u32 = ((@as(u32, chunk_src[pos]) << 8) | @as(u32, chunk_src[pos + 1])) & 0xFFF;
            return pos + 2 + sz;
        } else {
            if (pos + 3 > chunk_src.len) return null;
            const sz: u32 = readBE24(chunk_src, pos);
            return pos + 3 + sz;
        }
    } else if (ct == 1 or ct == 2 or ct == 4 or ct == 6) {
        // Entropy chunk types — Huffman (4) is GPU-emitted; types 1, 2, 6
        // are legacy and parsed for forward compat. 3 or 5 byte header +
        // compressed payload.
        if (first_byte >= 0x80) {
            if (pos + 3 > chunk_src.len) return null;
            const bits: u32 = readBE24(chunk_src, pos);
            const comp_size = bits & 0x3FF;
            return pos + 3 + comp_size;
        } else {
            if (pos + 5 > chunk_src.len) return null;
            const bits: u32 = (@as(u32, chunk_src[pos + 1]) << 24) | (@as(u32, chunk_src[pos + 2]) << 16) | (@as(u32, chunk_src[pos + 3]) << 8) | @as(u32, chunk_src[pos + 4]);
            const comp_size = bits & 0x3FFFF;
            return pos + 5 + comp_size;
        }
    } else if (ct == 5) {
        // Paired-secondary marker: [0x50][countA:u24][countB:u24] = 7 bytes
        if (pos + 7 > chunk_src.len) return null;
        return pos + 7;
    } else if (ct == 7) {
        // Paired-primary marker: [0x70][countA:u24][embedded type-6 stream]
        if (pos + 4 > chunk_src.len) return null;
        return skipStreamHeader(chunk_src, pos + 4);
    }
    return null;
}
