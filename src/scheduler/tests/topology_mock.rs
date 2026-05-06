// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! Integration tests — `Topology<MockSail>` exercises the
//! collective-ops contract end-to-end.
//!
//! The real-device path on `Topology<SpankerControl>` returns
//! `Error::NotImplemented` until the kernel ABI gains
//! `SPANKER_IOC_WORK_SUBMIT` (PR #6b). Inter-card link bandwidth /
//! latency characterisation comes from the cross-stream issue
//! against MAST filed alongside this PR.

use spanker_scheduler::{
    AllGather, AllReduce, MockSail, ReduceOp, Topology, INTERCARD_BUS_WIDTH, INTERCARD_LANES,
    INTERCARD_LANE_WIDTH,
};

#[test]
fn all_reduce_sum_two_cards_yields_sum() {
    let t = Topology::<MockSail>::with_mock(2);
    let mut per_card = vec![vec![1.0f32, 2.0, 3.0], vec![10.0, 20.0, 30.0]];
    t.all_reduce_f32(&mut per_card, ReduceOp::Sum)
        .expect("AllReduce sum on 2 cards");
    for v in &per_card {
        assert_eq!(v, &[11.0, 22.0, 33.0]);
    }
}

#[test]
fn all_reduce_sum_four_cards_yields_total() {
    let t = Topology::<MockSail>::with_mock(4);
    let mut per_card = vec![vec![1.0f32], vec![2.0], vec![3.0], vec![4.0]];
    t.all_reduce_f32(&mut per_card, ReduceOp::Sum)
        .expect("AllReduce sum on 4 cards");
    for v in &per_card {
        assert_eq!(v, &[10.0]);
    }
}

#[test]
fn all_reduce_avg_four_cards_yields_mean() {
    let t = Topology::<MockSail>::with_mock(4);
    let mut per_card = vec![vec![1.0f32], vec![2.0], vec![3.0], vec![4.0]];
    t.all_reduce_f32(&mut per_card, ReduceOp::Avg)
        .expect("AllReduce avg on 4 cards");
    for v in &per_card {
        assert_eq!(v, &[2.5]);
    }
}

#[test]
fn all_reduce_max_picks_per_index_max() {
    let t = Topology::<MockSail>::with_mock(3);
    let mut per_card = vec![
        vec![1.0f32, 5.0, 9.0],
        vec![7.0, 2.0, 8.0],
        vec![3.0, 6.0, 4.0],
    ];
    t.all_reduce_f32(&mut per_card, ReduceOp::Max)
        .expect("AllReduce max on 3 cards");
    for v in &per_card {
        assert_eq!(v, &[7.0, 6.0, 9.0]);
    }
}

#[test]
fn all_reduce_min_picks_per_index_min() {
    let t = Topology::<MockSail>::with_mock(3);
    let mut per_card = vec![
        vec![1.0f32, 5.0, 9.0],
        vec![7.0, 2.0, 8.0],
        vec![3.0, 6.0, 4.0],
    ];
    t.all_reduce_f32(&mut per_card, ReduceOp::Min)
        .expect("AllReduce min on 3 cards");
    for v in &per_card {
        assert_eq!(v, &[1.0, 2.0, 4.0]);
    }
}

#[test]
fn all_gather_four_cards_concatenates_in_order() {
    let t = Topology::<MockSail>::with_mock(4);
    let per_card = vec![
        vec![1.0f32, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ];
    let gathered = t.all_gather_f32(&per_card).expect("AllGather on 4 cards");
    assert_eq!(gathered, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn intercard_constants_match_mast_14_contract() {
    assert_eq!(INTERCARD_LANES, 4);
    assert_eq!(INTERCARD_LANE_WIDTH, 32);
    assert_eq!(INTERCARD_BUS_WIDTH, 128);
    assert_eq!(INTERCARD_LANES * INTERCARD_LANE_WIDTH, INTERCARD_BUS_WIDTH);
}
