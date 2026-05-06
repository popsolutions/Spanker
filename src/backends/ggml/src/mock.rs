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

use crate::{Error, MatmulInt4, Result, QK_K};

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
        if k == 0 || k % QK_K != 0 {
            return Err(Error::BadDims { m, k, n, qk: QK_K });
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

    #[test]
    fn mock_records_four_transactions_in_order() {
        let mock = MockSail::new();
        let mut out = [0u8; 64];
        mock.matmul_q4_k(&[0u8; 32], &[0u8; 64], &mut out, 4, QK_K, 4)
            .expect("mock matmul should succeed on aligned k");

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
}
