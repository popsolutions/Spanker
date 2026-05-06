// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! Collective-ops trait surfaces and host-side mock impls.
//!
//! [`AllReduce`] and [`AllGather`] are the data-path collective
//! ops the scheduler exposes to the runtime. Real device impls
//! land alongside the inter-card link protocol (ADR-014, MAST
//! cross-stream issue filed alongside this PR).
//!
//! [`TensorParallel`] and [`ModelParallel`] are minimal marker
//! traits today — they expose `shard_count` so the runtime can
//! plan partitioning without committing to a tensor type. Bodies
//! grow when the GGML matmul (PR #5b) needs concrete shard
//! geometry.

use spanker_runtime::SpankerControl;

use crate::topology::{MockSail, Topology};
use crate::{Error, Result};

/// Reduction operation for [`AllReduce`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise sum across cards.
    Sum,
    /// Element-wise maximum across cards.
    Max,
    /// Element-wise minimum across cards.
    Min,
    /// Element-wise average across cards.
    Avg,
}

/// Reduce a per-card array across all sails using `op`. After
/// the call every per-card buffer contains the same reduced
/// result.
pub trait AllReduce {
    /// Reduce `per_card[i]` across all `i` using `op`. All
    /// buffers must have the same length; mismatched shapes
    /// produce [`Error::ShapeMismatch`]. The number of buffers
    /// must equal the topology size.
    fn all_reduce_f32(&self, per_card: &mut [Vec<f32>], op: ReduceOp) -> Result<()>;
}

/// Gather per-card buffers into a single flat result.
pub trait AllGather {
    /// Concatenate `per_card[0..n]` in topology-index order and
    /// return the flat buffer. The number of buffers must equal
    /// the topology size.
    fn all_gather_f32(&self, per_card: &[Vec<f32>]) -> Result<Vec<f32>>;
}

/// Tensor-parallel partitioning interface.
///
/// Skeleton today: only exposes the shard count. Real geometry
/// arrives with PR #5b.
pub trait TensorParallel {
    /// Number of shards a tensor would be split into.
    fn shard_count(&self) -> usize;
}

/// Model-parallel (layer-sharded) partitioning interface.
///
/// Skeleton today: only exposes the shard count.
pub trait ModelParallel {
    /// Number of shards a model would be split across.
    fn shard_count(&self) -> usize;
}

impl<H> TensorParallel for Topology<H> {
    fn shard_count(&self) -> usize {
        self.n_sails()
    }
}

impl<H> ModelParallel for Topology<H> {
    fn shard_count(&self) -> usize {
        self.n_sails()
    }
}

// -- mock impls (host-side simulation) --

fn validate_uniform<T>(per_card: &[Vec<T>], n_sails: usize) -> Result<usize> {
    if per_card.len() != n_sails {
        return Err(Error::TopologyMismatch {
            expected: n_sails,
            actual: per_card.len(),
        });
    }
    let stride = per_card.first().map(|v| v.len()).unwrap_or(0);
    for (i, v) in per_card.iter().enumerate() {
        if v.len() != stride {
            return Err(Error::ShapeMismatch {
                sail: i,
                expected: stride,
                actual: v.len(),
            });
        }
    }
    Ok(stride)
}

impl AllReduce for Topology<MockSail> {
    fn all_reduce_f32(&self, per_card: &mut [Vec<f32>], op: ReduceOp) -> Result<()> {
        let stride = validate_uniform(per_card, self.n_sails())?;
        if stride == 0 {
            return Ok(());
        }

        let n = per_card.len() as f32;
        let mut reduced = vec![0.0f32; stride];
        for i in 0..stride {
            let initial = per_card[0][i];
            let acc = match op {
                ReduceOp::Sum | ReduceOp::Avg => {
                    let mut s = 0.0f32;
                    for v in per_card.iter() {
                        s += v[i];
                    }
                    if matches!(op, ReduceOp::Avg) {
                        s / n
                    } else {
                        s
                    }
                }
                ReduceOp::Max => {
                    let mut m = initial;
                    for v in per_card.iter().skip(1) {
                        if v[i] > m {
                            m = v[i];
                        }
                    }
                    m
                }
                ReduceOp::Min => {
                    let mut m = initial;
                    for v in per_card.iter().skip(1) {
                        if v[i] < m {
                            m = v[i];
                        }
                    }
                    m
                }
            };
            reduced[i] = acc;
        }

        for v in per_card.iter_mut() {
            v.clone_from(&reduced);
        }
        Ok(())
    }
}

impl AllGather for Topology<MockSail> {
    fn all_gather_f32(&self, per_card: &[Vec<f32>]) -> Result<Vec<f32>> {
        let stride = validate_uniform(per_card, self.n_sails())?;
        let mut out = Vec::with_capacity(stride * self.n_sails());
        for v in per_card {
            out.extend_from_slice(v);
        }
        Ok(out)
    }
}

// -- real-device impls (deferred) --

impl AllReduce for Topology<SpankerControl> {
    fn all_reduce_f32(&self, _per_card: &mut [Vec<f32>], _op: ReduceOp) -> Result<()> {
        Err(Error::NotImplemented(
            "AllReduce on real device requires SPANKER_IOC_WORK_SUBMIT (PR #6b)",
        ))
    }
}

impl AllGather for Topology<SpankerControl> {
    fn all_gather_f32(&self, _per_card: &[Vec<f32>]) -> Result<Vec<f32>> {
        Err(Error::NotImplemented(
            "AllGather on real device requires SPANKER_IOC_WORK_SUBMIT (PR #6b)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_shard_count_matches_n_sails() {
        let t = Topology::<MockSail>::with_mock(3);
        assert_eq!(<Topology<MockSail> as TensorParallel>::shard_count(&t), 3);
        assert_eq!(<Topology<MockSail> as ModelParallel>::shard_count(&t), 3);
    }

    #[test]
    fn all_reduce_sum_topology_mismatch() {
        let t = Topology::<MockSail>::with_mock(2);
        let mut per_card = vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let err = t
            .all_reduce_f32(&mut per_card, ReduceOp::Sum)
            .expect_err("expected TopologyMismatch");
        assert!(matches!(
            err,
            Error::TopologyMismatch {
                expected: 2,
                actual: 3
            }
        ));
    }

    #[test]
    fn all_reduce_sum_shape_mismatch() {
        let t = Topology::<MockSail>::with_mock(2);
        let mut per_card = vec![vec![1.0f32, 2.0], vec![3.0]];
        let err = t
            .all_reduce_f32(&mut per_card, ReduceOp::Sum)
            .expect_err("expected ShapeMismatch");
        assert!(matches!(
            err,
            Error::ShapeMismatch {
                sail: 1,
                expected: 2,
                actual: 1
            }
        ));
    }
}
