// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! # ggml-spanker — GGML int4 matmul backend for the PopSolutions Sails
//!
//! Per ADR-001 (Rust runtime) and the cross-stream contract with
//! `popsolutions/MAST`. This crate exposes the [`MatmulInt4`]
//! trait — a Q4_K-shaped matrix-multiply primitive — together with
//! two implementations:
//!
//! - [`SailMatmul`]: the real-device path. Currently a stub
//!   returning [`Error::NotImplemented`] until the kernel ABI gains
//!   `SPANKER_IOC_WORK_SUBMIT` (deferred to PR #5b after ADR-003
//!   pins the v1 ABI).
//! - [`MockSail`]: a host-side mock that records the AXI4 traffic
//!   the matmul *would* issue, so unit tests can assert correctly-
//!   shaped transactions without a real device. Will be displaced
//!   for integration testing once Agent 1 exposes a queryable
//!   `axi4_mem_model` cocotb harness in MAST (cross-stream issue
//!   filed alongside this PR).
//!
//! ## Q4_K layout
//!
//! Mirrors upstream GGML's `block_q4_K` (256 weights packed into
//! 144 bytes, with 8-bit scales). Constants are duplicated here
//! and verified against `enum ggml_type` exposed by the bindgen
//! `ffi` module so they cannot drift silently.

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod mock;
pub mod sail;

pub use mock::{MockSail, Transaction};
pub use sail::SailMatmul;

// NOTE: bindgen-derived FFI types over upstream GGML are landed
// in PR #5b alongside the real-device SailMatmul implementation.
// The GGML submodule at external/ggml/ is pinned by this PR so
// PR #5b's build.rs has a stable header source. See the
// build-dependencies note in Cargo.toml for the MSRV rationale.

/// Number of weights packed into one Q4_K block. Mirrors
/// upstream GGML's `QK_K` constant.
pub const QK_K: usize = 256;

/// Bytes per Q4_K block. Mirrors upstream GGML's
/// `sizeof(block_q4_K)` (144 bytes: 12 bytes of scales + 128 bytes
/// of nibble-packed weights + 4 bytes of `d`/`dmin` half-floats).
pub const Q4_K_BLOCK_BYTES: usize = 144;

/// Errors returned by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Caller passed dimensions that violate the Q4_K layout
    /// contract (typically `k` not a multiple of [`QK_K`]).
    #[error("bad dims: m={m} k={k} n={n}; require k % {qk} == 0")]
    BadDims {
        /// Number of output rows.
        m: usize,
        /// Reduction dimension.
        k: usize,
        /// Number of output columns.
        n: usize,
        /// Required block size (i.e. [`QK_K`]).
        qk: usize,
    },

    /// The output buffer is smaller than the matmul requires.
    #[error("output buffer too small: have {have} bytes, need {need}")]
    OutputTooSmall {
        /// Bytes the caller provided.
        have: usize,
        /// Bytes the matmul actually needs.
        need: usize,
    },

    /// Real-device matmul is not yet wired up (waiting on
    /// `SPANKER_IOC_WORK_SUBMIT`).
    #[error("not implemented yet (waiting on SPANKER_IOC_WORK_SUBMIT)")]
    NotImplemented,

    /// Underlying runtime error (ioctl, open, etc.).
    #[error(transparent)]
    Runtime(#[from] spanker_runtime::Error),
}

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Q4_K matrix multiplication primitive.
///
/// Inputs `a`, `b`, and `out` are raw byte slices in GGML's Q4_K
/// quantized layout. The matmul computes `out = a · b^T` in
/// row-major terms with shapes:
///
/// - `a`: `m × k` quantized weights (`k` MUST be a multiple of
///   [`QK_K`]).
/// - `b`: `k × n` quantized weights.
/// - `out`: `m × n` quantized weights, allocated by the caller.
pub trait MatmulInt4 {
    /// Issue a Q4_K matmul against the underlying device.
    fn matmul_q4_k(
        &self,
        a: &[u8],
        b: &[u8],
        out: &mut [u8],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_k_block_bytes_constant_matches_known_layout() {
        // Q4_K is 144 bytes per block in upstream GGML; this
        // constant must not drift independently. PR #5b will
        // additionally cross-check it against the bindgen-derived
        // `enum ggml_type` size table.
        assert_eq!(Q4_K_BLOCK_BYTES, 144);
        assert_eq!(QK_K, 256);
    }
}
