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
///
/// Marked `#[non_exhaustive]` so future variants
/// (e.g. `Product`, `LogSumExp`, `BitwiseOr`) can be added
/// without a major-version semver bump. Downstream `match` arms
/// must include a `_ =>` catch-all.
///
/// # Regression guard
///
/// The following doctest fails to compile *because* `ReduceOp`
/// is `#[non_exhaustive]`: a downstream `match` that names every
/// current variant is rejected without a `_ =>` arm. If someone
/// removes the `#[non_exhaustive]` attribute the doctest will
/// start to compile, the `compile_fail` will fail, and CI will
/// catch the silent semver-evolution regression.
///
/// ```compile_fail
/// use spanker_scheduler::ReduceOp;
/// fn name(op: ReduceOp) -> &'static str {
///     match op {
///         ReduceOp::Sum => "sum",
///         ReduceOp::Max => "max",
///         ReduceOp::Min => "min",
///         ReduceOp::Avg => "avg",
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
///
/// # Buffer-layout contract
///
/// Callers MUST provide one contiguous `Vec<f32>` per card,
/// indexed in topology order. Implementations are free to
/// stream each card's buffer sequentially before moving to the
/// next, so per-card contiguity is the public layout contract
/// for both the host-side mock and the real-device path. Do NOT
/// pass interleaved or strided slices: the real-device DMA path
/// (PR #6b) will assume per-card contiguity to map each `Vec`
/// to a single scatter-gather descriptor.
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

        // Cache-friendly layout: hoist the `match op` outside the
        // per-element loop and transpose the loops so each card's
        // contiguous Vec is read sequentially (outer = card,
        // inner = element). For n_sails=4, stride=1M this turns
        // a 4-way Vec stride per element into a single linear
        // sweep per card — at least one order-of-magnitude fewer
        // L1 misses on the host-side mock and (more importantly)
        // honours the per-card-contiguous buffer contract that
        // the real-device DMA path (PR #6b) will rely on.
        //
        // Pick the per-op combinator once up-front as a `fn`
        // pointer, then run a single generic outer-card /
        // inner-element fold. This both eliminates the previous
        // `let rest = ...` fragility (a single `iter().skip(1)`
        // was consumed inside one match arm; future variants
        // would have surfaced a "value used after move" error)
        // AND keeps the dispatch a single decision per call.
        // NB: `ReduceOp` is `#[non_exhaustive]`, but in-crate
        // matches are still exhaustive — when adding a new
        // variant (Product, LogSumExp, BitwiseOr…), extend
        // this `match` instead of falling back to a `_` arm
        // (which would silently drop unimplemented ops).
        let combine: fn(&mut f32, f32) = match op {
            ReduceOp::Sum | ReduceOp::Avg => |acc, x| *acc += x,
            ReduceOp::Max => |acc, x| {
                if x > *acc {
                    *acc = x;
                }
            },
            ReduceOp::Min => |acc, x| {
                if x < *acc {
                    *acc = x;
                }
            },
        };

        // Bit-exact-identical to the prior per-op-arm impl:
        // initialise `reduced` from card 0, then fold cards
        // 1..n into it. Avg divides after the fold so the
        // accumulation order matches the previous Sum path.
        let n_cards = per_card.len();
        let mut reduced = per_card[0].clone();
        for v in per_card.iter().skip(1) {
            for (r, x) in reduced.iter_mut().zip(v.iter()) {
                combine(r, *x);
            }
        }
        if matches!(op, ReduceOp::Avg) {
            let n = n_cards as f32;
            for r in reduced.iter_mut() {
                *r /= n;
            }
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

    // NB: the `*_non_exhaustive_requires_catchall_downstream`
    // unit tests previously lived here. They were removed in
    // PR #11 because `#[allow(unreachable_patterns)]` made them
    // pass even if `#[non_exhaustive]` was removed from the
    // enum, so they did not actually guard the attribute. The
    // load-bearing guards are the `compile_fail` doctests on
    // [`ReduceOp`] and [`Error`] (in `lib.rs`), which run
    // `rustc` and assert downstream-shape matches without a
    // `_ =>` arm fail to compile.

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
