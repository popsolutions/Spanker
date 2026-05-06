// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! Real-device implementation of [`crate::MatmulInt4`].
//!
//! Currently a stub returning [`crate::Error::NotImplemented`]
//! because the kernel ABI does not yet expose a work-submission
//! ioctl. Lands fully in PR #5b after ADR-003 pins the v1 ABI and
//! the kernel module gains `SPANKER_IOC_WORK_SUBMIT`.

use spanker_runtime::SpankerControl;

use crate::{Error, MatmulInt4, Result};

/// Real-device implementation of [`MatmulInt4`].
///
/// Holds a [`SpankerControl`] handle for issuing future work-submit
/// ioctls. Construct via [`SailMatmul::new`] after opening the
/// control device with `SpankerControl::open()`.
pub struct SailMatmul {
    // Held now so the public constructor signature is stable from
    // PR #5 forward — PR #5b will start using it.
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
        // PR #5b will replace this with: dma writes for A/B, a
        // SPANKER_IOC_WORK_SUBMIT ioctl carrying the matmul
        // descriptor, and a dma read for OUT.
        Err(Error::NotImplemented)
    }
}
