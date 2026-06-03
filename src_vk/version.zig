//! Library version, exposed to both the C ABI (`slzVersionString`) and
//! the CLI (`--version`). One source of truth so the two surfaces
//! cannot drift.
//!
//! VK PORT NOTE: kept verbatim from src/version.zig — the Vulkan-backed
//! library reports the same version string as the CUDA one so client
//! tooling (file headers, --version output) interprets both binaries as
//! interchangeable codecs.

pub const string: [:0]const u8 = "3.0.0";
