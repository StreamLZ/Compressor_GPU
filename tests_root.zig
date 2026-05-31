//! Test-executable root for harnesses that need to import from both
//! `src/` (CUDA production code) and `src_vulkan/` (Vulkan port) AND
//! `tests/` (out-of-tree integration tests).
//!
//! Zig 0.16 enforces that a module's relative `@import` calls stay
//! within the directory of the module's root source file. The M9
//! cross-backend conformance harness lives under `tests/` but needs to
//! reach `src/encode/streamlz_encoder.zig` and friends; rooting the test
//! module here at the repo root widens the "package root" to the whole
//! repository, so `src/...` and `tests/...` imports both resolve. The
//! file is intentionally tiny — only an empty `test` block whose body
//! imports the actual test file. The Zig test runner picks up every
//! `test` declaration reachable from this root, including those nested
//! one level down inside `tests/cross_backend_tests.zig`.
//!
//! Add new test entry-points by appending another `_ = @import("...")`
//! line below (one per top-level test file).

test {
    _ = @import("tests/cross_backend_tests.zig");
}
