// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! Integration tests — `MockSail` asserts AXI4 transaction shape
//! for a Q4_K matmul.
//!
//! Real-device path (`SailMatmul`) is gated on the cross-stream
//! issue against MAST asking Agent 1 for a queryable
//! `axi4_mem_model` cocotb harness; until that lands the mock is
//! the source of truth for transaction shape.

use ggml_spanker::{
    Error, MatmulInt4, MockSail, Transaction, OUTPUT_ELEM_BYTES, Q4_K_BLOCK_BYTES, QK_K,
};

fn alloc_operands(m: usize, k: usize, n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // `checked_mul` mirrors the production helpers
    // (`expected_a_bytes`, `expected_b_bytes`, `expected_out_bytes`)
    // so a future contributor copying this allocator inherits the
    // overflow-safe idiom rather than the bare `*` shortcut.
    let blocks = k / QK_K;
    let a_len = m
        .checked_mul(blocks)
        .and_then(|v| v.checked_mul(Q4_K_BLOCK_BYTES))
        .expect("A bytes overflow usize");
    let b_len = n
        .checked_mul(blocks)
        .and_then(|v| v.checked_mul(Q4_K_BLOCK_BYTES))
        .expect("B bytes overflow usize");
    let out_len = m
        .checked_mul(n)
        .and_then(|v| v.checked_mul(OUTPUT_ELEM_BYTES))
        .expect("OUT bytes overflow usize");
    let a = vec![0u8; a_len];
    let b = vec![0u8; b_len];
    let out = vec![0u8; out_len];
    (a, b, out)
}

#[test]
fn q4_k_matmul_issues_four_axi4_phases_in_order() {
    let m = 4;
    let k = QK_K; // exactly one Q4_K block per row of A
    let n = 4;

    let (a, b, mut out) = alloc_operands(m, k, n);

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

#[test]
fn matmul_rejects_mismatched_a_slice_length() {
    let mock = MockSail::new();
    let (a, b, mut out) = alloc_operands(4, QK_K, 4);
    let short_a = &a[..a.len() - 1];
    let err = mock
        .matmul_q4_k(short_a, &b, &mut out, 4, QK_K, 4)
        .expect_err("undersized A must trigger BadDims");
    assert!(matches!(err, Error::BadDims { qk, .. } if qk == QK_K));
}

#[test]
fn matmul_returns_output_too_small_when_out_undersized() {
    let mock = MockSail::new();
    let (a, b, _) = alloc_operands(4, QK_K, 4);
    let mut undersized = vec![0u8; 4]; // need 4*4*4 = 64
    let err = mock
        .matmul_q4_k(&a, &b, &mut undersized, 4, QK_K, 4)
        .expect_err("OutputTooSmall expected");
    match err {
        Error::OutputTooSmall { have, need } => {
            assert_eq!(have, 4);
            assert_eq!(need, 4 * 4 * OUTPUT_ELEM_BYTES);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn matmul_accepts_m_zero_as_noop() {
    let mock = MockSail::new();
    let (a, b, mut out) = alloc_operands(0, QK_K, 4);
    mock.matmul_q4_k(&a, &b, &mut out, 0, QK_K, 4)
        .expect("m=0 is a documented no-op");
    assert!(
        mock.transactions().is_empty(),
        "no AXI4 traffic for an empty matmul"
    );
}
