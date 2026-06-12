//! Preset-dictionary registry (v4 #16 phase 1).
//!
//! Built-in dictionaries are compiled into the binary via @embedFile
//! and identified by a well-known `dictionary_id` carried in the SLZ1
//! frame header (flag bit 3). The decoder resolves the ID through
//! this registry and rejects frames whose ID it cannot resolve
//! (`error.UnknownDictionary`) - the wire format carries the ID only,
//! never the dictionary bytes.
//!
//! The IDs, names, and dictionary BYTES are shared verbatim with the
//! CPU sibling project (Compressor_Native src/dict/) so frames are
//! dictionary-compatible across the two products. An ID permanently
//! identifies exact bytes: retraining a dictionary means assigning a
//! NEW ID, never changing the bytes behind an existing one - decoders
//! resolve by ID alone and a silent content change would corrupt
//! every existing frame that references it.
//!
//! Custom dictionaries use IDs >= `custom_id_base` and must be
//! provided to both encoder and decoder out of band (registration
//! API lands with the C ABI surface in a later phase).

const std = @import("std");

/// Metadata for a built-in (or, later, caller-registered) dictionary.
pub const DictInfo = struct {
    id: u32,
    name: []const u8,
    data: []const u8,
    /// File extensions (lowercase, with dot) that auto-select this
    /// dictionary in the CLI. Empty for fallback-only dictionaries.
    extensions: []const []const u8,
};

// Well-known built-in dictionary IDs (shared with the CPU sibling).
pub const id_json: u32 = 1;
pub const id_html: u32 = 2;
pub const id_text: u32 = 3;
pub const id_xml: u32 = 4;
pub const id_css: u32 = 5;
pub const id_js: u32 = 6;
pub const id_general: u32 = 7;
/// First StreamLZ-trained dictionary (not shared with the CPU
/// sibling): FASTCOVER over the even-indexed half of
/// assets/github_users.jsonl at the measured 2 KB ratio knee
/// (`zig build dict_gate0`). Also the worked example of per-corpus
/// dictionary training.
pub const id_github_users: u32 = 8;

/// First ID of the custom-dictionary range. Everything below is
/// reserved for registry built-ins.
pub const custom_id_base: u32 = 0x1000_0000;

/// Table of compiled-in dictionaries. Adding one is a new row plus a
/// `builtin/<name>.dict` asset (mirrored into srcVK/dict/builtin/).
pub const builtin_dicts: []const DictInfo = &.{
    .{
        .id = id_json,
        .name = "json",
        .data = @embedFile("builtin/json.dict"),
        .extensions = &.{ ".json", ".geojson", ".jsonl", ".ndjson" },
    },
    .{
        .id = id_html,
        .name = "html",
        .data = @embedFile("builtin/html.dict"),
        .extensions = &.{ ".html", ".htm", ".xhtml", ".svg" },
    },
    .{
        .id = id_text,
        .name = "text",
        .data = @embedFile("builtin/text.dict"),
        .extensions = &.{ ".txt", ".md", ".rst", ".log" },
    },
    .{
        .id = id_xml,
        .name = "xml",
        .data = @embedFile("builtin/xml.dict"),
        .extensions = &.{ ".xml", ".rss", ".atom", ".opml", ".pom", ".xsl" },
    },
    .{
        .id = id_css,
        .name = "css",
        .data = @embedFile("builtin/css.dict"),
        .extensions = &.{".css"},
    },
    .{
        .id = id_js,
        .name = "js",
        .data = @embedFile("builtin/js.dict"),
        .extensions = &.{ ".js", ".mjs", ".cjs", ".ts" },
    },
    .{
        .id = id_general,
        .name = "general",
        .data = @embedFile("builtin/general.dict"),
        .extensions = &.{},
    },
    .{
        .id = id_github_users,
        .name = "github-users",
        .data = @embedFile("builtin/github_users.dict"),
        .extensions = &.{},
    },
};

/// Look up a built-in dictionary by short name (e.g. "json", "html").
pub fn findByName(name: []const u8) ?*const DictInfo {
    for (builtin_dicts) |*d| {
        if (std.mem.eql(u8, d.name, name)) return d;
    }
    return null;
}

/// Look up a built-in dictionary by its well-known numeric ID.
pub fn findById(id: u32) ?*const DictInfo {
    for (builtin_dicts) |*d| {
        if (d.id == id) return d;
    }
    return null;
}

/// Select a built-in dictionary by file extension; falls back to
/// "general" for unknown or missing extensions.
pub fn findByExtension(path: []const u8) ?*const DictInfo {
    const ext = std.fs.path.extension(path);
    if (ext.len == 0) return findByName("general");
    var lower_buf: [16]u8 = undefined;
    const ext_lower = toLower(ext, &lower_buf) orelse return findByName("general");
    for (builtin_dicts) |*d| {
        for (d.extensions) |e| {
            if (std.mem.eql(u8, ext_lower, e)) return d;
        }
    }
    return findByName("general");
}

fn toLower(s: []const u8, buf: *[16]u8) ?[]const u8 {
    if (s.len > 16) return null;
    for (s, 0..) |c, i| {
        buf[i] = if (c >= 'A' and c <= 'Z') c + 32 else c;
    }
    return buf[0..s.len];
}

// ── Tests ────────────────────────────────────────────────────────────

const testing = std.testing;

test "every builtin resolves by id and by name, with non-empty data" {
    for (builtin_dicts) |*d| {
        try testing.expect(d.data.len > 0);
        try testing.expect(d.id < custom_id_base);
        try testing.expectEqual(d, findById(d.id).?);
        try testing.expectEqual(d, findByName(d.name).?);
    }
}

test "builtin ids are unique" {
    for (builtin_dicts, 0..) |*a, i| {
        for (builtin_dicts[i + 1 ..]) |*b| {
            try testing.expect(a.id != b.id);
        }
    }
}

test "unknown lookups return null" {
    try testing.expectEqual(@as(?*const DictInfo, null), findById(999));
    try testing.expectEqual(@as(?*const DictInfo, null), findById(custom_id_base));
    try testing.expectEqual(@as(?*const DictInfo, null), findByName("nope"));
}

test "extension selection is case-insensitive and falls back to general" {
    try testing.expectEqual(findByName("json").?, findByExtension("records.JSON").?);
    try testing.expectEqual(findByName("js").?, findByExtension("a/b/app.mjs").?);
    try testing.expectEqual(findByName("general").?, findByExtension("data.bin").?);
    try testing.expectEqual(findByName("general").?, findByExtension("noext").?);
}
