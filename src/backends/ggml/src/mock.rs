// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! Host-side mock implementation of [`crate::MatmulInt4`].
//!
//! Records the AXI4 traffic the matmul *would* issue so unit tests
//! can assert correctly-shaped transactions without a real device.
//! Will be displaced for real integration testing once Agent 1
//! exposes a queryable `axi4_mem_model` cocotb harness in MAST
//! (cross-stream issue filed alongside this PR).

use std::sync::Mutex;

use crate::{
    expected_a_bytes, expected_b_bytes, expected_out_bytes, Error, MatmulInt4, Result, QK_K,
};

/// One AXI4 transaction recorded by [`MockSail`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transaction {
    /// Write `len` bytes to `dev_addr`. The `label` field is a
    /// human-readable hint for assertions ("A", "B", etc.).
    Write {
        /// Target device-side BAR offset.
        dev_addr: u64,
        /// Payload length in bytes.
        len: usize,
        /// Operand label.
        label: &'static str,
    },
    /// Submit a compute command at `dev_addr` (i.e. the cmd FIFO).
    ComputeSubmit {
        /// Target device-side cmd-FIFO offset.
        dev_addr: u64,
        /// Operation kind, e.g. `"matmul_q4_k"`.
        kind: &'static str,
        /// Output rows.
        m: usize,
        /// Reduction dimension.
        k: usize,
        /// Output columns.
        n: usize,
    },
    /// Read `len` bytes from `dev_addr`.
    Read {
        /// Source device-side BAR offset.
        dev_addr: u64,
        /// Payload length in bytes.
        len: usize,
        /// Operand label.
        label: &'static str,
    },
}

// Mock BAR offsets — opaque to the test surface; tests check
// transaction *shape* via `Transaction` matching, not raw addrs.
// Real driver exposes these via a future `SPANKER_IOC_BAR_INFO`
// ioctl (PR #5b).
const MOCK_BAR_A: u64 = 0x10000;
const MOCK_BAR_B: u64 = 0x20000;
const MOCK_BAR_OUT: u64 = 0x30000;
const MOCK_CMD_FIFO: u64 = 0x4000;

/// Host-side mock that records AXI4 transactions a real
/// `SailMatmul` would issue.
#[derive(Default)]
pub struct MockSail {
    txns: Mutex<Vec<Transaction>>,
}

impl MockSail {
    /// Construct an empty mock with no recorded transactions.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the recorded transactions in submission order.
    pub fn transactions(&self) -> Vec<Transaction> {
        self.txns
            .lock()
            .expect("MockSail txn lock poisoned")
            .clone()
    }

    fn push(&self, txn: Transaction) {
        self.txns
            .lock()
            .expect("MockSail txn lock poisoned")
            .push(txn);
    }
}

impl MatmulInt4 for MockSail {
    fn matmul_q4_k(
        &self,
        a: &[u8],
        b: &[u8],
        out: &mut [u8],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        // Reject unaligned/zero k up front — the helpers return
        // None for these too, but a dedicated branch gives a
        // clearer error path.
        if k == 0 || k % QK_K != 0 {
            return Err(Error::BadDims { m, k, n, qk: QK_K });
        }

        // Cross-check operand slice lengths against the declared
        // `m × k × n` shape under Q4_K. m=0 or n=0 are accepted
        // as explicit no-ops (consistent with NumPy's empty-tensor
        // semantics); the helpers correctly return Some(0) there.
        let need_a = expected_a_bytes(m, k).ok_or(Error::BadDims { m, k, n, qk: QK_K })?;
        let need_b = expected_b_bytes(k, n).ok_or(Error::BadDims { m, k, n, qk: QK_K })?;
        let need_out = expected_out_bytes(m, n).ok_or(Error::BadDims { m, k, n, qk: QK_K })?;

        if a.len() != need_a || b.len() != need_b {
            return Err(Error::BadDims { m, k, n, qk: QK_K });
        }
        if out.len() < need_out {
            return Err(Error::OutputTooSmall {
                have: out.len(),
                need: need_out,
            });
        }

        // m=0 / n=0 with valid k is a no-op: no traffic needed.
        if m == 0 || n == 0 {
            return Ok(());
        }

        self.push(Transaction::Write {
            dev_addr: MOCK_BAR_A,
            len: a.len(),
            label: "A",
        });
        self.push(Transaction::Write {
            dev_addr: MOCK_BAR_B,
            len: b.len(),
            label: "B",
        });
        self.push(Transaction::ComputeSubmit {
            dev_addr: MOCK_CMD_FIFO,
            kind: "matmul_q4_k",
            m,
            k,
            n,
        });
        self.push(Transaction::Read {
            dev_addr: MOCK_BAR_OUT,
            len: out.len(),
            label: "OUT",
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Q4_K_BLOCK_BYTES;

    /// Build correctly-sized (m, k, n) operand buffers for tests.
    fn alloc_operands(m: usize, k: usize, n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let blocks = k / QK_K;
        let a = vec![0u8; m * blocks * Q4_K_BLOCK_BYTES];
        let b = vec![0u8; n * blocks * Q4_K_BLOCK_BYTES];
        let out = vec![0u8; m * n * crate::OUTPUT_ELEM_BYTES];
        (a, b, out)
    }

    #[test]
    fn mock_records_four_transactions_in_order() {
        let m = 4;
        let k = QK_K;
        let n = 4;
        let (a, b, mut out) = alloc_operands(m, k, n);

        let mock = MockSail::new();
        mock.matmul_q4_k(&a, &b, &mut out, m, k, n)
            .expect("mock matmul should succeed on aligned k with consistent slice lens");

        let txns = mock.transactions();
        assert_eq!(txns.len(), 4);
        assert!(matches!(&txns[0], Transaction::Write { label: "A", .. }));
        assert!(matches!(&txns[1], Transaction::Write { label: "B", .. }));
        assert!(matches!(
            &txns[2],
            Transaction::ComputeSubmit {
                kind: "matmul_q4_k",
                ..
            }
        ));
        assert!(matches!(&txns[3], Transaction::Read { label: "OUT", .. }));
    }

    #[test]
    fn mock_rejects_unaligned_k() {
        let mock = MockSail::new();
        let mut out = [0u8; 16];
        let err = mock
            .matmul_q4_k(&[], &[], &mut out, 1, 100, 1)
            .expect_err("expected BadDims");
        assert!(matches!(err, Error::BadDims { qk: QK_K, .. }));
    }

    #[test]
    fn mock_rejects_undersized_a_slice() {
        let mock = MockSail::new();
        let (mut a, b, mut out) = alloc_operands(4, QK_K, 4);
        a.pop(); // make A one byte short of the declared m × k shape.
        let err = mock
            .matmul_q4_k(&a, &b, &mut out, 4, QK_K, 4)
            .expect_err("expected BadDims for short A");
        assert!(matches!(err, Error::BadDims { qk: QK_K, .. }));
    }

    #[test]
    fn mock_rejects_undersized_b_slice() {
        let mock = MockSail::new();
        let (a, mut b, mut out) = alloc_operands(4, QK_K, 4);
        b.pop(); // make B one byte short of the declared k × n shape.
        let err = mock
            .matmul_q4_k(&a, &b, &mut out, 4, QK_K, 4)
            .expect_err("expected BadDims for short B");
        assert!(matches!(err, Error::BadDims { qk: QK_K, .. }));
    }

    #[test]
    fn mock_returns_output_too_small_when_out_undersized() {
        let mock = MockSail::new();
        let (a, b, _) = alloc_operands(4, QK_K, 4);
        let mut out = vec![0u8; 4]; // way short of m*n*4 = 64
        let err = mock
            .matmul_q4_k(&a, &b, &mut out, 4, QK_K, 4)
            .expect_err("expected OutputTooSmall");
        match err {
            Error::OutputTooSmall { have, need } => {
                assert_eq!(have, 4);
                assert_eq!(need, 4 * 4 * crate::OUTPUT_ELEM_BYTES);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn mock_accepts_m_zero_as_noop() {
        let mock = MockSail::new();
        let (a, b, mut out) = alloc_operands(0, QK_K, 4);
        mock.matmul_q4_k(&a, &b, &mut out, 0, QK_K, 4)
            .expect("m=0 with valid k is a no-op");
        assert!(
            mock.transactions().is_empty(),
            "no AXI4 traffic should be issued for an empty matmul"
        );
    }

    #[test]
    fn mock_accepts_n_zero_as_noop() {
        let mock = MockSail::new();
        let (a, b, mut out) = alloc_operands(4, QK_K, 0);
        mock.matmul_q4_k(&a, &b, &mut out, 4, QK_K, 0)
            .expect("n=0 with valid k is a no-op");
        assert!(mock.transactions().is_empty());
    }
}
