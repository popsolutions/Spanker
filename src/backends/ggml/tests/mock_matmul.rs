// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! Integration tests — `MockSail` asserts AXI4 transaction shape
//! for a Q4_K matmul.
//!
//! Real-device path (`SailMatmul`) is gated on the cross-stream
//! issue against MAST asking Agent 1 for a queryable
//! `axi4_mem_model` cocotb harness; until that lands the mock is
//! the source of truth for transaction shape.

use ggml_spanker::{Error, MatmulInt4, MockSail, Transaction, Q4_K_BLOCK_BYTES, QK_K};

#[test]
fn q4_k_matmul_issues_four_axi4_phases_in_order() {
    let m = 4;
    let k = QK_K; // exactly one Q4_K block per row of A
    let n = 4;

    // Q4_K input layout (skeleton-grade): m × (k / QK_K) blocks of
    // Q4_K_BLOCK_BYTES each. Real layouts will be honed in PR #5b.
    let a = vec![0u8; m * (k / QK_K) * Q4_K_BLOCK_BYTES];
    let b = vec![0u8; n * (k / QK_K) * Q4_K_BLOCK_BYTES];
    let mut out = vec![0u8; m * n * 4];

    let mock = MockSail::new();
    mock.matmul_q4_k(&a, &b, &mut out, m, k, n)
        .expect("mock matmul should succeed");

    let txns = mock.transactions();
    assert_eq!(
        txns.len(),
        4,
        "expected exactly 4 AXI4 transactions; got {txns:?}"
    );

    assert!(
        matches!(&txns[0], Transaction::Write { label: "A", len, .. } if *len == a.len()),
        "txn[0] should be a Write of A of len={}: {:?}",
        a.len(),
        txns[0]
    );
    assert!(
        matches!(&txns[1], Transaction::Write { label: "B", len, .. } if *len == b.len()),
        "txn[1] should be a Write of B of len={}: {:?}",
        b.len(),
        txns[1]
    );
    assert!(
        matches!(
            &txns[2],
            Transaction::ComputeSubmit {
                kind: "matmul_q4_k",
                m: txn_m,
                k: txn_k,
                n: txn_n,
                ..
            } if *txn_m == m && *txn_k == k && *txn_n == n
        ),
        "txn[2] should be a matmul_q4_k ComputeSubmit with m={m} k={k} n={n}: {:?}",
        txns[2]
    );
    assert!(
        matches!(&txns[3], Transaction::Read { label: "OUT", len, .. } if *len == out.len()),
        "txn[3] should be a Read of OUT of len={}: {:?}",
        out.len(),
        txns[3]
    );
}

#[test]
fn matmul_rejects_unaligned_k_dimension() {
    let mock = MockSail::new();
    let mut out = [0u8; 16];
    let err = mock
        .matmul_q4_k(&[], &[], &mut out, 1, 100, 1)
        .expect_err("expected BadDims");
    assert!(matches!(err, Error::BadDims { qk, .. } if qk == QK_K));
}
