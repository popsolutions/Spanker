// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! Real-device implementation of [`crate::MatmulInt4`].
//!
//! Currently a stub returning [`crate::Error::NotImplemented`]
//! because the kernel ABI does not yet expose a work-submission
//! ioctl. The real path lands once the kernel-driver PR adds
//! `SPANKER_IOC_WORK_SUBMIT` to the UAPI header — that PR is
//! itself gated on the DDR3 work-dispatch backend (cross-stream
//! Spanker #9). Until then the bindgen-derived `crate::ffi`
//! module exposes only PING + GET_VERSION; once WORK_SUBMIT
//! lands in the header, bindgen picks it up automatically and
//! the implementation below is fleshed out.

use spanker_runtime::SpankerControl;

use crate::{Error, MatmulInt4, Result};

/// Real-device implementation of [`MatmulInt4`].
///
/// Holds a [`SpankerControl`] handle for issuing future work-submit
/// ioctls. Construct via [`SailMatmul::new`] after opening the
/// control device with `SpankerControl::open()`.
pub struct SailMatmul {
    // Held now so the public constructor signature is stable from
    // PR #5 forward — the real implementation will start using it
    // once SPANKER_IOC_WORK_SUBMIT exists.
    #[allow(dead_code)]
    ctl: SpankerControl,
}

impl SailMatmul {
    /// Construct a new `SailMatmul` from an open [`SpankerControl`].
    pub fn new(ctl: SpankerControl) -> Self {
        Self { ctl }
    }
}

impl MatmulInt4 for SailMatmul {
    fn matmul_q4_k(
        &self,
        _a: &[u8],
        _b: &[u8],
        _out: &mut [u8],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<()> {
        // The real-device path will eventually issue:
        //   1. DMA writes of A and B into the device's DDR3 region,
        //   2. a SPANKER_IOC_WORK_SUBMIT ioctl with a matmul
        //      descriptor (m, k, n, A/B/OUT addrs),
        //   3. a DMA read of OUT.
        // Step 2's ABI is not yet defined (kernel UAPI header has
        // only PING + GET_VERSION today). Returning a clearly
        // labelled error keeps callers honest in the meantime.
        Err(Error::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the deferred-stub contract: `SailMatmul::matmul_q4_k`
    /// MUST return [`Error::NotImplemented`] until the kernel
    /// driver gains `SPANKER_IOC_WORK_SUBMIT` (cross-stream
    /// Spanker #9). When that lands and this test fails, replace
    /// it with the corresponding success-path assertions — do not
    /// just delete it.
    ///
    /// Uses a freshly-created temp file under `std::env::temp_dir()`
    /// for the underlying handle. The stub short-circuits with
    /// `NotImplemented` before issuing any ioctl, so the file's
    /// type doesn't matter — only that `open(2)` succeeds.
    /// Previous revisions used `/dev/null`, which is read-only in
    /// some hermetic sandboxes and caused the test to silently
    /// `return` (passing in CI summaries while never asserting
    /// anything). A temp file removes that silent-skip path.
    #[test]
    fn matmul_q4_k_returns_not_implemented() {
        // Unique-per-process path so concurrent test runs don't
        // collide. `cargo test` runs each integration binary in
        // its own process, but unit tests inside one binary share
        // a PID — `line!()` keeps this filename distinct from any
        // sibling test that might adopt the same pattern later.
        let tmp_path = std::env::temp_dir().join(format!(
            "spanker-sail-stub-{}-{}.tmp",
            std::process::id(),
            line!()
        ));
        // `create` truncates if the file already exists from a
        // previous run that was killed before cleanup ran.
        std::fs::File::create(&tmp_path).expect("temp file should be creatable");

        let ctl = SpankerControl::open_path(&tmp_path)
            .expect("freshly-created temp file should open r+w");
        let sail = SailMatmul::new(ctl);
        let err = sail
            .matmul_q4_k(&[], &[], &mut [], 0, 0, 0)
            .expect_err("SailMatmul must return NotImplemented today");
        assert!(
            matches!(err, Error::NotImplemented),
            "expected NotImplemented, got {err:?}"
        );

        // Best-effort cleanup; a leaked file in /tmp is harmless
        // but tidiness is cheap.
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn not_implemented_display_names_blocker_ioctl() {
        // Callers grepping logs need to find the right cross-stream
        // issue; the Display impl must mention SPANKER_IOC_WORK_SUBMIT.
        let msg = format!("{}", Error::NotImplemented);
        assert!(
            msg.contains("SPANKER_IOC_WORK_SUBMIT"),
            "NotImplemented Display should name the blocker ioctl: got {msg:?}"
        );
    }
}
