//! NEW per Exception 3 (no CUDA counterpart).
//!
//! CUDA↔VK cross-backend round-trip matrix: encode via CUDA, decode via
//! VK; encode via VK, decode via CUDA. Levels 1-2 full; levels 3-5 once
//! Huffman lands. Test bodies added by the fleshout agent.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
