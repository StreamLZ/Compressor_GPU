//! Dictionary trainer using the FASTCOVER algorithm (v4 #16 phase 0).
//!
//! Scans training samples for frequently-occurring d-mer patterns and
//! greedily selects k-byte segments that cover the most common ones.
//! Output is a raw byte dictionary suitable for LZ match preload: the
//! encoder treats dictionary bytes as reachable history at chunk
//! start, so records that share structure with the training set
//! compress against it instead of starting cold.
//!
//! The dictionary is filled BACKWARD: the highest-scoring segments
//! land at the END. That ordering matters for the wire format - a
//! match into dict depth q encodes as offset ~ position + (D - q), so
//! tail bytes are the cheapest to reference and the trainer puts the
//! most valuable content there.
//!
//! Ported from the CPU sibling (Compressor_Native src/dict/trainer.zig),
//! itself based on zstd's FASTCOVER (lib/dictBuilder/fastcover.c).
//! Host-only, dependency-free; no GPU interaction.

const std = @import("std");

const default_d: usize = 8;
const default_k: usize = 48;
const default_f: u5 = 20;
const default_epochs: usize = 32;
const max_zero_score_epochs: usize = 10;

pub const TrainParams = struct {
    /// Target dictionary size in bytes. The result may be smaller if
    /// the training data runs out of scoring segments first.
    dict_size: usize = 32768,
    /// d-mer width: the pattern unit whose frequency drives segment
    /// scoring. Must be >= 8 (the hash reads 8 bytes).
    d: usize = default_d,
    /// Segment size: dictionary content is selected in k-byte pieces.
    k: usize = default_k,
    /// log2 of the frequency-table size; the d-mer hash is folded to
    /// f bits. Larger = fewer hash collisions, more memory.
    f: u5 = default_f,
    /// Number of segment-selection rounds per dict_size (the trainer
    /// round-robins samples across rounds).
    epochs: usize = default_epochs,
};

pub const TrainResult = struct {
    dict: []u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *TrainResult) void {
        self.allocator.free(self.dict);
    }
};

fn hashDmer(src: [*]const u8, d: usize, f: u5) usize {
    std.debug.assert(d >= 8);
    const v = std.mem.readInt(u64, src[0..8], .little);
    const h = v *% 0x9E3779B97F4A7C15;
    const shift: u6 = @intCast(@as(u7, 64) - @as(u7, f));
    return @intCast(h >> shift);
}

/// Train a dictionary from `samples` (one slice per record). The
/// result is at most `params.dict_size` bytes and is owned by the
/// returned TrainResult (free with `deinit`). Deterministic for a
/// given sample list and params.
pub fn train(
    allocator: std.mem.Allocator,
    samples: []const []const u8,
    params: TrainParams,
) !TrainResult {
    if (samples.len == 0) return error.NoSamples;

    const d = params.d;
    const k = params.k;
    const f = params.f;
    const freq_size: usize = @as(usize, 1) << f;

    // Count d-mer frequencies across all samples.
    const freqs = try allocator.alloc(u32, freq_size);
    defer allocator.free(freqs);
    @memset(freqs, 0);

    var total_len: usize = 0;
    for (samples) |sample| {
        if (sample.len < d) continue;
        var pos: usize = 0;
        while (pos + d <= sample.len) : (pos += 1) {
            const idx = hashDmer(sample.ptr + pos, d, f);
            freqs[idx] += 1;
        }
        total_len += sample.len;
    }
    if (total_len < d) return error.NoSamples;

    // Per-segment frequency tracking (reset per segment evaluation).
    const seg_freqs = try allocator.alloc(u32, freq_size);
    defer allocator.free(seg_freqs);

    // Build the dictionary by selecting best segments, filled backward
    // so the highest-scoring content ends up at the tail (cheapest
    // offsets for the LZ encoder; see the module header).
    const dict = try allocator.alloc(u8, params.dict_size);
    errdefer allocator.free(dict);
    var dict_pos: usize = params.dict_size;

    const epoch_size = params.dict_size / params.epochs;
    if (epoch_size == 0) return error.NoSamples;

    var zero_score_count: usize = 0;
    var epoch_idx: usize = 0;

    while (dict_pos > 0 and zero_score_count < max_zero_score_epochs) {
        // Round-robin through samples.
        const sample_idx = epoch_idx % samples.len;
        const sample = samples[sample_idx];
        epoch_idx += 1;

        if (sample.len < k) continue;

        // Find the best segment of size k in this sample: slide a
        // k-wide window, scoring each position by the global frequency
        // of the d-mers it covers (counting each distinct d-mer once).
        @memset(seg_freqs, 0);

        const dmers_in_k = k - d + 1;
        var best_score: u64 = 0;
        var best_begin: usize = 0;
        var active_score: u64 = 0;
        var active_begin: usize = 0;

        var pos: usize = 0;
        while (pos + d <= sample.len) : (pos += 1) {
            const idx = hashDmer(sample.ptr + pos, d, f);

            if (seg_freqs[idx] == 0) {
                active_score += freqs[idx];
            }
            seg_freqs[idx] += 1;

            // Window is full - drop the oldest d-mer.
            if (pos - active_begin >= dmers_in_k) {
                const del_idx = hashDmer(sample.ptr + active_begin, d, f);
                seg_freqs[del_idx] -= 1;
                if (seg_freqs[del_idx] == 0) {
                    active_score -= freqs[del_idx];
                }
                active_begin += 1;
            }

            if (active_score > best_score and pos - active_begin + 1 >= dmers_in_k) {
                best_score = active_score;
                best_begin = active_begin;
            }
        }

        if (best_score == 0) {
            zero_score_count += 1;
            continue;
        }
        zero_score_count = 0;

        // Copy the best segment into the dictionary (filled backward).
        const seg_len = @min(k, dict_pos);
        const seg_start = best_begin;
        if (seg_start + seg_len > sample.len) continue;

        dict_pos -= seg_len;
        @memcpy(dict[dict_pos..][0..seg_len], sample[seg_start..][0..seg_len]);

        // Zero out the selected d-mers so they're not double-counted.
        var z: usize = 0;
        while (z + d <= seg_len) : (z += 1) {
            const idx = hashDmer(sample.ptr + seg_start + z, d, f);
            freqs[idx] = 0;
        }
    }

    // If the dictionary wasn't fully filled, shift content to the start
    // and shrink the allocation. On realloc failure the errdefer above
    // frees the original allocation and we propagate the error.
    if (dict_pos > 0) {
        const used = params.dict_size - dict_pos;
        std.mem.copyForwards(u8, dict[0..used], dict[dict_pos..][0..used]);
        const result = try allocator.realloc(dict, used);
        return .{ .dict = result[0..used], .allocator = allocator };
    }

    return .{ .dict = dict, .allocator = allocator };
}

// ── Tests ────────────────────────────────────────────────────────────

const testing = std.testing;

test "train rejects an empty sample list" {
    try testing.expectError(error.NoSamples, train(testing.allocator, &.{}, .{}));
}

test "train rejects samples with no d-mer-sized content" {
    const samples = [_][]const u8{ "tiny", "abc" };
    try testing.expectError(error.NoSamples, train(testing.allocator, &samples, .{}));
}

test "trained dictionary contains the dominant shared pattern" {
    // Every sample shares one long repeated phrase surrounded by
    // per-sample noise; FASTCOVER must select a segment covering it.
    const phrase = "\"avatar_url\":\"https://avatars.example.com/u/";
    var bufs: [8][256]u8 = undefined;
    var samples: [8][]const u8 = undefined;
    for (&bufs, 0..) |*buf, i| {
        samples[i] = std.fmt.bufPrint(
            buf,
            "id{d}{s}{d}xyz{s}{d}end-of-record-{d}",
            .{ i, phrase, i, phrase, i * 7, i },
        ) catch unreachable;
    }
    var result = try train(testing.allocator, &samples, .{ .dict_size = 1024, .k = 48 });
    defer result.deinit();
    try testing.expect(result.dict.len > 0);
    try testing.expect(result.dict.len <= 1024);
    try testing.expect(std.mem.indexOf(u8, result.dict, phrase[0..16]) != null);
}

test "train is deterministic and respects dict_size" {
    var prng = std.Random.DefaultPrng.init(0xd1c7);
    const random = prng.random();
    var bufs: [16][512]u8 = undefined;
    var samples: [16][]const u8 = undefined;
    for (&bufs, 0..) |*buf, i| {
        random.bytes(buf);
        // Plant a shared motif so scoring has something to find.
        @memcpy(buf[64..96], "shared-motif-shared-motif-shar32");
        samples[i] = buf;
    }
    var a = try train(testing.allocator, &samples, .{ .dict_size = 2048 });
    defer a.deinit();
    var b = try train(testing.allocator, &samples, .{ .dict_size = 2048 });
    defer b.deinit();
    try testing.expect(a.dict.len <= 2048);
    try testing.expectEqualSlices(u8, a.dict, b.dict);
}
