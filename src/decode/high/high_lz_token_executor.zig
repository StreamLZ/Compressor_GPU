//! High decoder phase-2: LZ run processor for Type 0 (delta literals)
//! and Type 1 (raw literals).
//!
//! Type 0 is single-pass: walks cmd/offs/len/lit streams inline.
//! Type 1 is two-pass: resolve tokens (carousel + lengths) into a flat
//! array, then execute with match-source prefetching. Both paths are
//! simple and correct; SIMD AVX2 tail optimizations and binary-search
//! dstSafeEnd split land in phase 7.
//!
//! Note: correctness of small-offset match propagation depends on
//! `io.copy_helpers.wildCopy16` doing load-store-load-store (not both
//! loads first); see the comment on that function.

const std = @import("std");
const builtin = @import("builtin");
const constants = @import("../../format/streamlz_constants.zig");
const copy = @import("../../io/copy_helpers.zig");
const ptr_math = @import("../../io/ptr_math.zig");
const high = @import("high_lz_decoder.zig");

const safe_space = constants.safe_space;

pub const DecodeError = high.DecodeError;

pub fn processLzRuns(
    mode: u32,
    dst: [*]u8,
    dst_size: usize,
    base_offset: usize,
    lz: *const high.HighLzTable,
    scratch_free: [*]u8,
    scratch_end: [*]u8,
) DecodeError!void {
    return processLzRunsInner(mode, dst, dst_size, base_offset, base_offset, lz, scratch_free, scratch_end);
}

pub fn processLzRunsSc(
    mode: u32,
    dst: [*]u8,
    dst_size: usize,
    initial_copy_offset: usize,
    match_base_offset: usize,
    lz: *const high.HighLzTable,
    scratch_free: [*]u8,
    scratch_end: [*]u8,
) DecodeError!void {
    return processLzRunsInner(mode, dst, dst_size, initial_copy_offset, match_base_offset, lz, scratch_free, scratch_end);
}

fn processLzRunsInner(
    mode: u32,
    dst: [*]u8,
    dst_size: usize,
    initial_copy_offset: usize,
    match_base_offset: usize,
    lz: *const high.HighLzTable,
    scratch_free: [*]u8,
    scratch_end: [*]u8,
) DecodeError!void {
    if (dst_size == 0) return error.OutputTruncated;
    const dst_end: [*]u8 = dst + dst_size;
    const dst_start: [*]const u8 = @ptrFromInt(@intFromPtr(dst) - match_base_offset);
    const dst_run_start: [*]u8 = dst + (if (initial_copy_offset == 0) @as(usize, 8) else 0);

    switch (mode) {
        0 => try processLzRunsType0(lz, dst_run_start, dst_end, dst_start),
        1 => try processLzRunsType1(lz, dst_run_start, dst_end, dst_start, scratch_free, scratch_end),
        else => return error.BadMode,
    }
}

/// Execute Type 1 tokens that were pre-resolved during phase 1.
/// Skips resolveTokens (already done) but still executes both literal
/// copies and match copies via executeTokensType1 + trailing literals.
pub fn processLzRunsType1PreResolved(
    tokens: [*]const LzToken,
    token_count: u32,
    lz: *const high.HighLzTable,
    dst: [*]u8,
    dst_size: usize,
    base_offset: usize,
) DecodeError!void {
    if (dst_size == 0) return error.OutputTruncated;
    const dst_end: [*]u8 = dst + dst_size;
    const dst_start: [*]const u8 = @ptrFromInt(@intFromPtr(dst) - base_offset);
    const dst_run_start: [*]u8 = dst + (if (base_offset == 0) @as(usize, 8) else 0);

    var lit_stream = lz.lit_stream;
    const lit_stream_end = lz.lit_stream + lz.lit_stream_size;

    if (token_count > 0) {
        try executeTokensType1(tokens, token_count, dst_run_start, dst_end, dst_start, &lit_stream);
    }

    // Advance dst past all tokens for trailing literal calculation.
    var dst_after: [*]u8 = dst_run_start;
    if (token_count > 0) {
        const last = tokens[token_count - 1];
        dst_after = dst_run_start + @as(usize, @intCast(last.dst_pos + last.lit_len + last.match_length));
    }

    // Trailing literals.
    var trailing: usize = @intFromPtr(dst_end) - @intFromPtr(dst_after);
    if (trailing != @intFromPtr(lit_stream_end) - @intFromPtr(lit_stream)) return error.StreamMismatch;
    while (trailing >= 8) {
        copy.copy64(dst_after, lit_stream);
        dst_after += 8;
        lit_stream += 8;
        trailing -= 8;
    }
    while (trailing > 0) : (trailing -= 1) {
        dst_after[0] = lit_stream[0];
        dst_after += 1;
        lit_stream += 1;
    }
}

/// Fallback allocator for the Type 1 token array when the scratch buffer
/// is exhausted. Uses libc malloc/free via Zig's c_allocator — matches the
/// `NativeMemory.Alloc` and avoids the ~5µs syscall cost of page_allocator
/// (VirtualAlloc/VirtualFree). For a 100 MB L9 decode we hit the fallback
/// ~12,000 times; VTune Hotspots showed page_allocator at 13.8% of CPU.
const fallback_allocator = std.heap.c_allocator;

// ────────────────────────────────────────────────────────────
//  Type 0 — delta-coded literals, single-pass
// ────────────────────────────────────────────────────────────

fn processLzRunsType0(
    lz: *const high.HighLzTable,
    dst_in: [*]u8,
    dst_end: [*]u8,
    dst_start: [*]const u8,
) DecodeError!void {
    var cmd_stream = lz.cmd_stream;
    const cmd_stream_end = lz.cmd_stream + lz.cmd_stream_size;
    var len_stream: [*]align(1) const i32 = lz.len_stream;
    const len_stream_end: [*]align(1) const i32 = lz.len_stream + lz.len_stream_size;
    var lit_stream = lz.lit_stream;
    const lit_stream_end = lz.lit_stream + lz.lit_stream_size;
    var offs_stream: [*]align(1) const i32 = lz.offs_stream;
    const offs_stream_end: [*]align(1) const i32 = lz.offs_stream + lz.offs_stream_size;

    var dst = dst_in;
    const dst_safe_end: [*]u8 = if (@intFromPtr(dst_end) >= @intFromPtr(dst) + safe_space)
        dst_end - safe_space
    else
        dst_in;

    // Recent offsets carousel. Slots 3-5 are active; 0-2 are scratch for rotation.
    var recent_offsets: [7]i32 = @splat(0);
    const init_recent: i32 = -@as(i32, @intCast(constants.initial_recent_offset));
    recent_offsets[3] = init_recent;
    recent_offsets[4] = init_recent;
    recent_offsets[5] = init_recent;
    var last_offset: i32 = init_recent;

    while (@intFromPtr(cmd_stream) < @intFromPtr(cmd_stream_end)) {
        const command_byte: u32 = cmd_stream[0];
        cmd_stream += 1;

        var literal_length: u32 = command_byte & 0x3;
        const offset_index: u32 = command_byte >> 6;
        const match_length: u32 = (command_byte >> 2) & 0xF;

        if (literal_length == 3) {
            if (@intFromPtr(len_stream) >= @intFromPtr(len_stream_end)) return error.StreamMismatch;
            literal_length = @bitCast(len_stream[0]);
            len_stream += 1;
        }

        // Speculative offs read.
        recent_offsets[6] = offs_stream[0];

        const lit_delta_offset: i32 = last_offset;

        const picked: i32 = recent_offsets[offset_index + 3];
        recent_offsets[offset_index + 3] = recent_offsets[offset_index + 2];
        recent_offsets[offset_index + 2] = recent_offsets[offset_index + 1];
        recent_offsets[offset_index + 1] = recent_offsets[offset_index + 0];
        recent_offsets[3] = picked;
        last_offset = picked;

        if (offset_index == 3) offs_stream += 1;

        const actual_match_len: u32 = blk: {
            if (match_length != 15) break :blk match_length + 2;
            if (@intFromPtr(len_stream) >= @intFromPtr(len_stream_end)) return error.StreamMismatch;
            const extra: u32 = @bitCast(len_stream[0]);
            len_stream += 1;
            break :blk 14 + extra;
        };

        const lit_len_i: usize = @intCast(literal_length);
        const ml_i: usize = @intCast(actual_match_len);
        const off_i: i32 = picked;
        const lit_off_i: i32 = lit_delta_offset;

        // Prefetch the match source for THIS iteration — the literal copy
        // that follows gives ~10-100 cycles for the line to arrive in L1.
        //
        {
            const pre_addr: usize = @intFromPtr(dst) + lit_len_i +% @as(usize, @bitCast(@as(isize, off_i)));
            @prefetch(@as([*]const u8, @ptrFromInt(pre_addr)), .{ .rw = .read, .locality = 3, .cache = .data });
        }

        if (@intFromPtr(dst + lit_len_i + ml_i) >= @intFromPtr(dst_safe_end)) {
            @branchHint(.cold);
            if (@intFromPtr(dst + lit_len_i + ml_i) > @intFromPtr(dst_end)) return error.OutputTruncated;

            copyLiteralAddExact(dst, lit_stream, lit_off_i, lit_len_i);
            dst += lit_len_i;
            lit_stream += lit_len_i;

            const match_addr: usize = @intFromPtr(dst) +% @as(usize, @bitCast(@as(isize, off_i)));
            if (match_addr < @intFromPtr(dst_start)) return error.OutputTruncated;
            const match_ptr: [*]const u8 = @ptrFromInt(match_addr);
            copyMatchExact(dst, match_ptr, ml_i);
            dst += ml_i;
        } else {
            // Fast path — wide copies.
            const match_src_post_lit: usize = @intFromPtr(dst + lit_len_i) +% @as(usize, @bitCast(@as(isize, off_i)));
            if (match_src_post_lit < @intFromPtr(dst_start)) return error.OutputTruncated;

            // Cascading delta-literal copy: the 80% case has ≤ 2 literals,
            // so the first copy64Add handles it with zero branches taken.
            // 8- and 16-byte literals fall through to the nested `if`s. Only
            // literals > 24 bytes enter the 8-byte-stride loop — much rarer
            // than the short-token case, and then still wide rather than
            // byte-wise.
            const lit_src_ptr: [*]const u8 = ptr_math.offsetPtr([*]const u8, dst, @as(isize, lit_off_i));
            copy.copy64Add(dst, lit_stream, lit_src_ptr);
            if (literal_length > 8) {
                copy.copy64Add(dst + 8, lit_stream + 8, lit_src_ptr + 8);
                if (literal_length > 16) {
                    copy.copy64Add(dst + 16, lit_stream + 16, lit_src_ptr + 16);
                    if (literal_length > 24) {
                        var remaining: usize = lit_len_i;
                        var dd = dst;
                        var ss = lit_stream;
                        var ee = lit_src_ptr;
                        while (remaining > 24) {
                            copy.copy64Add(dd + 24, ss + 24, ee + 24);
                            remaining -= 8;
                            dd += 8;
                            ss += 8;
                            ee += 8;
                        }
                    }
                }
            }
            dst += lit_len_i;
            lit_stream += lit_len_i;

            // Match copy.
            const match_addr: usize = @intFromPtr(dst) +% @as(usize, @bitCast(@as(isize, off_i)));
            const match_ptr: [*]const u8 = @ptrFromInt(match_addr);
            copy.copy64(dst, match_ptr);
            copy.copy64(dst + 8, match_ptr + 8);
            if (match_length == 15 and ml_i > 16) {
                copy.wildCopy16(dst + 16, match_ptr + 16, dst + ml_i);
            }
            dst += ml_i;
        }
    }

    if (@intFromPtr(offs_stream) != @intFromPtr(offs_stream_end) or
        @intFromPtr(len_stream) != @intFromPtr(len_stream_end))
    {
        return error.StreamMismatch;
    }

    var trailing: usize = @intFromPtr(dst_end) - @intFromPtr(dst);
    if (trailing != @intFromPtr(lit_stream_end) - @intFromPtr(lit_stream)) return error.StreamMismatch;

    // Trailing literal copy with delta-add (exact).
    const off_usize: usize = @bitCast(@as(isize, last_offset));
    while (trailing >= 8) {
        const lit_src: [*]const u8 = @ptrFromInt(@intFromPtr(dst) +% off_usize);
        copy.copy64Add(dst, lit_stream, lit_src);
        dst += 8;
        lit_stream += 8;
        trailing -= 8;
    }
    while (trailing > 0) : (trailing -= 1) {
        const delta_ptr: [*]const u8 = @ptrFromInt(@intFromPtr(dst) +% off_usize);
        dst[0] = lit_stream[0] +% delta_ptr[0];
        dst += 1;
        lit_stream += 1;
    }
}

fn copyLiteralAddExact(dst_in: [*]u8, src_in: [*]const u8, delta_off: i32, length: usize) void {
    var dst = dst_in;
    var src = src_in;
    var remaining = length;
    const off_usize: usize = @bitCast(@as(isize, delta_off));
    while (remaining >= 8) {
        const delta_ptr: [*]const u8 = @ptrFromInt(@intFromPtr(dst) +% off_usize);
        copy.copy64Add(dst, src, delta_ptr);
        dst += 8;
        src += 8;
        remaining -= 8;
    }
    while (remaining > 0) : (remaining -= 1) {
        const delta_ptr: [*]const u8 = @ptrFromInt(@intFromPtr(dst) +% off_usize);
        dst[0] = src[0] +% delta_ptr[0];
        dst += 1;
        src += 1;
    }
}

fn copyMatchExact(dst_in: [*]u8, src_in: [*]const u8, length: usize) void {
    // Match copies may overlap when offset < length; must copy byte-by-byte
    // once we're past the overlap-safe zone.
    var dst = dst_in;
    var src = src_in;
    var remaining = length;
    while (remaining > 0) : (remaining -= 1) {
        dst[0] = src[0];
        dst += 1;
        src += 1;
    }
}

// ────────────────────────────────────────────────────────────
//  Type 1 — raw literals, two-phase
// ────────────────────────────────────────────────────────────

pub const LzToken = struct {
    dst_pos: i32,
    offset: i32,
    lit_len: i32,
    match_length: i32,
};

pub fn resolveTokens(
    lz: *const high.HighLzTable,
    tokens: [*]LzToken,
    dst_size: i32,
    offs_final_out: *[*]align(1) const i32,
    len_final_out: *[*]align(1) const i32,
) DecodeError!u32 {
    var cmd_stream = lz.cmd_stream;
    const cmd_stream_end = lz.cmd_stream + lz.cmd_stream_size;
    var len_stream: [*]align(1) const i32 = lz.len_stream;
    var offs_stream: [*]align(1) const i32 = lz.offs_stream;

    // 3-entry recent-offset LIFO held in registers (CMOV chain below).
    var recent0: i32 = -@as(i32, @intCast(constants.initial_recent_offset));
    var recent1: i32 = recent0;
    var recent2: i32 = recent0;

    var dst_pos: i32 = 0;
    var token_index: u32 = 0;

    if (comptime builtin.cpu.arch == .x86_64) {
        asm volatile (".p2align 6");
        asm volatile ("nop");
        asm volatile (".p2align 5");
    }
    while (@intFromPtr(cmd_stream) < @intFromPtr(cmd_stream_end)) {
        const command_byte: u32 = cmd_stream[0];
        cmd_stream += 1;
        var literal_length: u32 = command_byte & 0x3;
        const offset_index: u32 = command_byte >> 6;
        const match_length: u32 = (command_byte >> 2) & 0xF;
        std.debug.assert(offset_index <= 3);

        // Speculative long-literal decode.
        const speculative_long: u32 = @bitCast(len_stream[0]);
        if (literal_length == 3) {
            @branchHint(.unlikely);
            literal_length = speculative_long;
            len_stream += 1;
        }

        const new_off: i32 = offs_stream[0];
        const ro_arr = [4]i32{ recent0, recent1, recent2, new_off };
        const picked: i32 = ro_arr[offset_index];
        recent2 = if (offset_index < 2) recent2 else recent1;
        recent1 = if (offset_index == 0) recent1 else recent0;
        recent0 = picked;

        // Branchless offs_stream advance: (oi + 1) & 4 is 0 for oi in
        // {0,1,2} and 4 (= sizeof(i32)) for oi == 3. Saves the branch
        // VTune flagged at 11% of resolveTokens.
        const offs_adv: usize = (@as(usize, offset_index) + 1) & 4;
        offs_stream = @ptrFromInt(@intFromPtr(offs_stream) + offs_adv);

        const actual_match_len: i32 = blk: {
            if (match_length != 15) {
                @branchHint(.likely);
                break :blk @as(i32, @intCast(match_length + 2));
            }
            const extra: i32 = len_stream[0];
            len_stream += 1;
            break :blk 14 + extra;
        };

        tokens[token_index] = .{
            .dst_pos = dst_pos,
            .offset = picked,
            .lit_len = @intCast(literal_length),
            .match_length = actual_match_len,
        };
        token_index += 1;

        dst_pos += @as(i32, @intCast(literal_length)) + actual_match_len;
        if (dst_pos > dst_size) {
            @branchHint(.cold);
            return error.OutputTruncated;
        }
    }

    offs_final_out.* = offs_stream;
    len_final_out.* = len_stream;
    return token_index;
}

fn processLzRunsType1(
    lz: *const high.HighLzTable,
    dst_in: [*]u8,
    dst_end: [*]u8,
    dst_start: [*]const u8,
    scratch_free: [*]u8,
    scratch_end: [*]u8,
) DecodeError!void {
    var dst = dst_in;
    var lit_stream = lz.lit_stream;
    const lit_stream_end = lz.lit_stream + lz.lit_stream_size;
    const offs_stream_end = lz.offs_stream + lz.offs_stream_size;
    const len_stream_end = lz.len_stream + lz.len_stream_size;

    const token_count = lz.cmd_stream_size;
    var fallback_tokens: ?[]LzToken = null;
    defer if (fallback_tokens) |t| fallback_allocator.free(t);

    if (token_count > 0) {
        const token_bytes: usize = @as(usize, token_count) * @sizeOf(LzToken);
        const tokens: [*]LzToken = blk: {
            if (@intFromPtr(scratch_free) + token_bytes <= @intFromPtr(scratch_end)) {
                // Align scratch_free to 16 bytes.
                const aligned = (@intFromPtr(scratch_free) + 15) & ~@as(usize, 15);
                if (aligned + token_bytes > @intFromPtr(scratch_end)) {
                    // Fall through to allocator path.
                } else {
                    break :blk @ptrFromInt(aligned);
                }
            }
            const slice = fallback_allocator.alignedAlloc(LzToken, .fromByteUnits(16), token_count) catch return error.OutputTruncated;
            fallback_tokens = slice;
            break :blk slice.ptr;
        };

        var offs_final: [*]align(1) const i32 = undefined;
        var len_final: [*]align(1) const i32 = undefined;
        const resolved = try resolveTokens(lz, tokens, @intCast(@intFromPtr(dst_end) - @intFromPtr(dst)), &offs_final, &len_final);

        if (@intFromPtr(offs_final) != @intFromPtr(offs_stream_end) or
            @intFromPtr(len_final) != @intFromPtr(len_stream_end))
        {
            return error.StreamMismatch;
        }

        try executeTokensType1(tokens, resolved, dst, dst_end, dst_start, &lit_stream);

        if (resolved > 0) {
            const last = tokens[resolved - 1];
            const advance: usize = @intCast(last.dst_pos + last.lit_len + last.match_length);
            dst += advance;
        }
    }

    // Trailing literal copy (raw).
    var trailing: usize = @intFromPtr(dst_end) - @intFromPtr(dst);
    if (trailing != @intFromPtr(lit_stream_end) - @intFromPtr(lit_stream)) return error.StreamMismatch;

    while (trailing >= 64) {
        copy.copy64Bytes(dst, lit_stream);
        dst += 64;
        lit_stream += 64;
        trailing -= 64;
    }
    while (trailing >= 8) {
        copy.copy64(dst, lit_stream);
        dst += 8;
        lit_stream += 8;
        trailing -= 8;
    }
    while (trailing > 0) : (trailing -= 1) {
        dst[0] = lit_stream[0];
        dst += 1;
        lit_stream += 1;
    }
}

// ────────────────────────────────────────────────────────────
//  Forward-LZ diagnostic — global accumulator
//  Enable with: comptime fwd_diag_enabled = true
// ────────────────────────────────────────────────────────────

const fwd_diag_enabled = false;

const FwdDiag = struct {
    const Entry = struct {
        hash: u64 = 0,
        total_bytes: u64 = 0,
        occurrences: u32 = 0,
        match_length: u32 = 0,
    };

    const max_buckets = 1 << 20; // 1M buckets
    var buckets: [max_buckets]Entry = @splat(Entry{});
    var total_match_bytes: u64 = 0;
    var total_tokens: u64 = 0;
    var chunks_seen: u32 = 0;

    fn accumulate(tokens: [*]const LzToken, count: u32, dst_base: [*]const u8) void {
        chunks_seen += 1;
        for (0..count) |i| {
            const t = tokens[i];
            const ml: usize = @intCast(t.match_length);
            if (ml < 2) continue;

            total_match_bytes += ml;
            total_tokens += 1;

            const match_start: [*]const u8 = dst_base + @as(usize, @intCast(t.dst_pos)) + @as(usize, @intCast(t.lit_len));

            var h: u64 = @as(u64, ml) *% 0x9E3779B97F4A7C15;
            const hash_bytes = @min(ml, 16);
            for (0..hash_bytes) |b| {
                h ^= @as(u64, match_start[b]) *% (@as(u64, 0x100000001B3) +% @as(u64, b) * 7);
            }
            h ^= h >> 33;
            h *%= 0xFF51AFD7ED558CCD;
            h ^= h >> 29;

            const mask = max_buckets - 1;
            var probe: usize = @intCast(h & mask);
            for (0..128) |_| {
                if (buckets[probe].occurrences == 0) {
                    buckets[probe] = .{
                        .hash = h,
                        .total_bytes = ml,
                        .occurrences = 1,
                        .match_length = @intCast(ml),
                    };
                    break;
                }
                if (buckets[probe].hash == h and buckets[probe].match_length == ml) {
                    buckets[probe].total_bytes += ml;
                    buckets[probe].occurrences += 1;
                    break;
                }
                probe = (probe + 1) & mask;
            }
        }
    }

    fn report() void {
        // Collect all entries with occurrences >= 2 into a sortable list.
        // Use a fixed-size buffer — count first.
        var unique: u64 = 0;
        var repeated: u64 = 0;
        var repeated_bytes: u64 = 0;
        var occ_2_5: u64 = 0;
        var occ_2_5_bytes: u64 = 0;
        var occ_6_10: u64 = 0;
        var occ_6_10_bytes: u64 = 0;
        var occ_11_50: u64 = 0;
        var occ_11_50_bytes: u64 = 0;
        var occ_51_plus: u64 = 0;
        var occ_51_plus_bytes: u64 = 0;

        var sorted_bytes: [65536]u64 = undefined;
        var sorted_count: usize = 0;

        for (0..max_buckets) |b| {
            const occ = buckets[b].occurrences;
            if (occ >= 1) unique += 1;
            if (occ >= 2) {
                repeated += 1;
                repeated_bytes += buckets[b].total_bytes;
                if (sorted_count < 65536) {
                    sorted_bytes[sorted_count] = buckets[b].total_bytes;
                    sorted_count += 1;
                }
                if (occ <= 5) { occ_2_5 += 1; occ_2_5_bytes += buckets[b].total_bytes; }
                else if (occ <= 10) { occ_6_10 += 1; occ_6_10_bytes += buckets[b].total_bytes; }
                else if (occ <= 30) { occ_11_50 += 1; occ_11_50_bytes += buckets[b].total_bytes; }
                else { occ_51_plus += 1; occ_51_plus_bytes += buckets[b].total_bytes; }
            }
        }

        // Sort descending.
        const slice = sorted_bytes[0..sorted_count];
        std.sort.pdq(u64, slice, {}, struct {
            fn cmp(_: void, a: u64, b: u64) bool {
                return a > b;
            }
        }.cmp);

        std.debug.print("\n[FWD-LZ GLOBAL] {d} chunks, {d} tokens, {d} total match bytes\n", .{
            chunks_seen, total_tokens, total_match_bytes,
        });
        std.debug.print("  {d} unique patterns, {d} repeated (>= 2 occurrences)\n", .{ unique, repeated });
        std.debug.print("  Occurrence distribution:\n", .{});
        std.debug.print("    2-5x:   {d:>8} patterns, {d:>12} bytes\n", .{ occ_2_5, occ_2_5_bytes });
        std.debug.print("    6-10x:  {d:>8} patterns, {d:>12} bytes\n", .{ occ_6_10, occ_6_10_bytes });
        std.debug.print("    11-30x: {d:>8} patterns, {d:>12} bytes\n", .{ occ_11_50, occ_11_50_bytes });
        std.debug.print("    31+x:   {d:>8} patterns, {d:>12} bytes\n", .{ occ_51_plus, occ_51_plus_bytes });
        std.debug.print("  repeated patterns cover {d} bytes = {d:.1}% of match bytes\n", .{
            repeated_bytes,
            @as(f64, @floatFromInt(repeated_bytes)) / @as(f64, @floatFromInt(@max(total_match_bytes, 1))) * 100.0,
        });

        // Cumulative coverage at percentiles.
        std.debug.print("\n  Cumulative coverage (sorted by total bytes desc):\n", .{});
        const milestones = [_]usize{ 10, 50, 100, 500, 1000, 5000, 10000, 50000 };
        var cumul: u64 = 0;
        var mi: usize = 0;
        for (0..sorted_count) |i| {
            cumul += slice[i];
            if (mi < milestones.len and i + 1 == milestones[mi]) {
                std.debug.print("    top {d:>6} patterns: {d:>12} bytes = {d:.1}% of match, {d:.1}% of output\n", .{
                    milestones[mi], cumul,
                    @as(f64, @floatFromInt(cumul)) / @as(f64, @floatFromInt(@max(total_match_bytes, 1))) * 100.0,
                    @as(f64, @floatFromInt(cumul)) / @as(f64, @floatFromInt(@max(total_match_bytes + total_tokens * 2, 1))) * 100.0,
                });
                mi += 1;
            }
        }
        // Final total.
        std.debug.print("    all {d:>6} patterns: {d:>12} bytes = {d:.1}% of match\n", .{
            sorted_count, cumul,
            @as(f64, @floatFromInt(cumul)) / @as(f64, @floatFromInt(@max(total_match_bytes, 1))) * 100.0,
        });
    }
};

fn analyzeMatchPatterns(tokens: [*]const LzToken, count: u32, dst_base: [*]const u8) void {
    if (comptime fwd_diag_enabled) FwdDiag.accumulate(tokens, count, dst_base);
}

pub fn reportFwdDiag() void {
    if (comptime fwd_diag_enabled) {
        if (FwdDiag.chunks_seen > 0) FwdDiag.report();
    }
}

/// Prefetch this far ahead (in tokens) so the match source cache line is
/// resident in L1 by the time we reach it. Uses 128 based on an Arrow
/// Lake sweep (32→1376, 64→1547, 128→1541, 256→1514 MB/s). We match it.
const prefetch_ahead: u32 = 128;

fn executeTokensType1(
    tokens: [*]const LzToken,
    token_count: u32,
    dst_in: [*]u8,
    dst_end: [*]u8,
    dst_start: [*]const u8,
    lit_stream_inout: *[*]const u8,
) DecodeError!void {
    const dst_base = dst_in;
    var lit_stream = lit_stream_inout.*;

    const dst_safe_end: [*]u8 = if (@intFromPtr(dst_end) >= @intFromPtr(dst_in) + safe_space)
        dst_end - safe_space
    else
        dst_in;

    // Split the loop into a prefetch-safe main body (i in 0..safe_end)
    // and a no-prefetch tail (i in safe_end..token_count). Removes the
    // `if (prefetch_index < token_count)` branch from every iteration of
    // the hot path.
    const safe_end: u32 = if (token_count > prefetch_ahead) token_count - prefetch_ahead else 0;
    var i: u32 = 0;
    if (comptime builtin.cpu.arch == .x86_64) {
        asm volatile (".p2align 6");
        asm volatile ("nop");
        asm volatile (".p2align 5");
    }
    while (i < safe_end) : (i += 1) {
        // Match-source prefetch for a token prefetch_ahead steps ahead —
        // unconditional in this loop body.
        const pt = tokens[i + prefetch_ahead];
        const pre_base: usize = @intFromPtr(dst_base) + @as(usize, @intCast(pt.dst_pos)) + @as(usize, @intCast(pt.lit_len));
        const pre_addr: usize = pre_base +% @as(usize, @bitCast(@as(isize, pt.offset)));
        const p0: [*]const u8 = @ptrFromInt(pre_addr);
        @prefetch(p0, .{ .rw = .read, .locality = 3, .cache = .data });
        @prefetch(p0 + 64, .{ .rw = .read, .locality = 3, .cache = .data });

        try processOneToken(tokens[i], dst_base, dst_safe_end, dst_end, dst_start, &lit_stream);
    }
    while (i < token_count) : (i += 1) {
        try processOneToken(tokens[i], dst_base, dst_safe_end, dst_end, dst_start, &lit_stream);
    }
    lit_stream_inout.* = lit_stream;
}

/// One token's worth of literal + match copy. Shared by the two
/// loop bodies (prefetch-safe main + no-prefetch tail) so the compiler
/// emits a single body. `lit_stream_inout` is updated in place.
inline fn processOneToken(
    t: LzToken,
    dst_base: [*]u8,
    dst_safe_end: [*]u8,
    dst_end: [*]u8,
    dst_start: [*]const u8,
    lit_stream_inout: *[*]const u8,
) DecodeError!void {
    var lit_stream = lit_stream_inout.*;
    const lit_len: usize = @intCast(t.lit_len);
    const ml: usize = @intCast(t.match_length);
    const offset: i32 = t.offset;

    const dst_token_start: [*]u8 = dst_base + @as(usize, @intCast(t.dst_pos));
    const dst_after_all: [*]u8 = dst_token_start + lit_len + ml;

    if (@intFromPtr(dst_after_all) > @intFromPtr(dst_safe_end)) {
        @branchHint(.cold);
        if (@intFromPtr(dst_after_all) > @intFromPtr(dst_end)) return error.OutputTruncated;
        var d = dst_token_start;
        var lrem = lit_len;
        while (lrem > 0) : (lrem -= 1) {
            d[0] = lit_stream[0];
            d += 1;
            lit_stream += 1;
        }
        const match_addr_slow: usize = @intFromPtr(d) +% @as(usize, @bitCast(@as(isize, offset)));
        if (match_addr_slow < @intFromPtr(dst_start)) {
            @branchHint(.cold);
            return error.OutputTruncated;
        }
        var s: [*]const u8 = @ptrFromInt(match_addr_slow);
        var mrem = ml;
        while (mrem > 0) : (mrem -= 1) {
            d[0] = s[0];
            d += 1;
            s += 1;
        }
        lit_stream_inout.* = lit_stream;
        return;
    }

    // Fast path: 16-byte SIMD literal copies.
    var d = dst_token_start;
    copy.copy16(d, lit_stream);
    if (lit_len > 16) {
        copy.copy16(d + 16, lit_stream + 16);
        if (lit_len > 32) {
            var remaining = lit_len;
            var dd = d + 32;
            var ss = lit_stream + 32;
            while (remaining > 32) {
                copy.copy16(dd, ss);
                remaining -= 16;
                dd += 16;
                ss += 16;
            }
        }
    }
    d += lit_len;
    lit_stream += lit_len;

    const match_addr: usize = @intFromPtr(d) +% @as(usize, @bitCast(@as(isize, offset)));
    const match_ptr: [*]const u8 = @ptrFromInt(match_addr);
    copy.copy64(d, match_ptr);
    copy.copy64(d + 8, match_ptr + 8);
    if (ml > 16) {
        copy.wildCopy16(d + 16, match_ptr + 16, d + ml);
    }
    lit_stream_inout.* = lit_stream;
}

