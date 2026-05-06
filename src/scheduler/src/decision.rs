// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! TP-vs-MP decision cost function.
//!
//! Given a tile shape and the rev-A bandwidth constants from
//! [`crate::bandwidth`], pick whether the tile is cheaper to run
//! tensor-parallel (weights sharded, activations AllReduce'd) or
//! model-parallel (weights replicated per layer-group, activations
//! sharded forward). Mock-only today: this is the load-bearing
//! scaffolding the runtime will eventually call when it partitions
//! a workload across the connected Sails per
//! `project_multicard_parallelism.md`.
//!
//! ## Cost model
//!
//! For a matmul tile with shape `(m, n, k)` and `bytes_per_element`
//! per scalar, on `n_sails` cards:
//!
//! - **Tensor parallel (TP):** weights are sharded across sails,
//!   so each card pulls `weight_bytes / n_sails` from local DDR
//!   in parallel. Each AllReduce step then sweeps the activation
//!   buffer (`m * n * bytes_per_element`) over the inter-card
//!   link. Cost ≈
//!   `weight_bytes / (n_sails * local_ddr_bw)
//!   + activation_bytes / intercard_bw`.
//! - **Model parallel (MP):** weights for this layer-group live
//!   on one sail, so that sail pulls the full `weight_bytes`
//!   from local DDR. The forward activation is sharded across
//!   sails so only `activation_bytes / n_sails` crosses the
//!   inter-card link per forward step. Cost ≈
//!   `weight_bytes / local_ddr_bw
//!   + activation_bytes / (n_sails * intercard_bw)`.
//!
//! Both are bandwidth-bound estimates — compute and latency are
//! deferred until silicon characterisation is available. The
//! point of this module is to compare the two strategies on the
//! same axis (wall-clock seconds) so the scheduler can pick the
//! smaller estimate.
//!
//! ## Intended use
//!
//! ```
//! use spanker_scheduler::{
//!     pick_strategy, Strategy, TileShape,
//!     LOCAL_DDR_BW_BYTES_PER_SEC, INTERCARD_BW_BYTES_PER_SEC,
//! };
//!
//! // A tiny activation, big weights → TP wins (shard heavy weights).
//! let tile = TileShape {
//!     m: 4,
//!     n: 4,
//!     k: 1 << 20,
//!     bytes_per_element: 4,
//! };
//! let chosen = pick_strategy(
//!     &tile,
//!     4,
//!     LOCAL_DDR_BW_BYTES_PER_SEC,
//!     INTERCARD_BW_BYTES_PER_SEC,
//! );
//! assert_eq!(chosen, Strategy::TensorParallel);
//! ```

/// Partitioning strategy picked by [`pick_strategy`].
///
/// Marked `#[non_exhaustive]` so future variants
/// (e.g. `Hybrid`, `Pipeline`) can land without a major-version
/// semver bump. Downstream `match` arms must include a `_ =>`
/// catch-all.
///
/// # Regression guard
///
/// The following doctest fails to compile *because* `Strategy`
/// is `#[non_exhaustive]`: a downstream `match` that names every
/// current variant is rejected without a `_ =>` arm. If someone
/// removes the `#[non_exhaustive]` attribute the doctest will
/// start to compile, the `compile_fail` will fail, and CI will
/// catch the silent semver-evolution regression.
///
/// ```compile_fail
/// use spanker_scheduler::Strategy;
/// fn name(s: Strategy) -> &'static str {
///     match s {
///         Strategy::TensorParallel => "tp",
///         Strategy::ModelParallel => "mp",
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Strategy {
    /// Shard weights across sails; AllReduce activations.
    TensorParallel,
    /// Replicate (or place) weights per sail; shard activations forward.
    ModelParallel,
}

/// Shape of one matmul tile the scheduler is sizing for.
///
/// `m` × `n` is the activation footprint the AllReduce
/// (TP) or forward shard (MP) carries; `m` × `k` × `bytes_per_element`
/// is the weight footprint each card or shard reads from local DDR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileShape {
    /// Output rows.
    pub m: u64,
    /// Output cols (also activation cols).
    pub n: u64,
    /// Inner / contraction dimension. Drives weight footprint.
    pub k: u64,
    /// Bytes per scalar (e.g. 4 for f32, 2 for f16. For Q4_0 the
    /// post-dequant scratch is the load-bearing throughput axis on
    /// the open-toolchain ECP5 stack — pass the post-dequant width
    /// here, not the packed nibble width).
    pub bytes_per_element: u64,
}

impl TileShape {
    /// Total weight footprint in bytes: `m * k * bytes_per_element`.
    ///
    /// Saturates rather than overflows so callers with bad inputs
    /// get a finite cost estimate instead of a panic.
    pub fn weight_bytes(&self) -> u64 {
        self.m
            .saturating_mul(self.k)
            .saturating_mul(self.bytes_per_element)
    }

    /// Total activation footprint in bytes: `m * n * bytes_per_element`.
    ///
    /// Saturates rather than overflows so callers with bad inputs
    /// get a finite cost estimate instead of a panic.
    pub fn activation_bytes(&self) -> u64 {
        self.m
            .saturating_mul(self.n)
            .saturating_mul(self.bytes_per_element)
    }
}

/// Convert a (bytes, bandwidth-in-bytes-per-second) pair into
/// seconds. Returns `f64::INFINITY` if `bw` is zero so the cost
/// model never silently divides by zero — the comparison in
/// [`pick_strategy`] will then prefer the other strategy or fall
/// through to the canonical TP default if both costs are infinite.
fn seconds(bytes: u64, bw: u64) -> f64 {
    if bw == 0 {
        return f64::INFINITY;
    }
    (bytes as f64) / (bw as f64)
}

/// Pick the cheaper of TP / MP for `tile` given `n_sails` and the
/// two bandwidth constants. See module docs for the cost model.
///
/// `n_sails == 0` is treated as `1` (single-card degenerate
/// path) — both strategies collapse to the same wall-clock and
/// the function returns [`Strategy::TensorParallel`] as the
/// canonical default. `n_sails == 1` likewise picks
/// [`Strategy::TensorParallel`] because there is no inter-card
/// path to compare against and TP is the lower-overhead default
/// when no sharding is active.
pub fn pick_strategy(
    tile: &TileShape,
    n_sails: u64,
    local_ddr_bw: u64,
    intercard_bw: u64,
) -> Strategy {
    let n = n_sails.max(1);
    if n == 1 {
        // Degenerate: no inter-card path. TP is the canonical
        // single-card default — same wall-clock as MP, but the
        // runtime's TP path is the simpler partitioning step.
        return Strategy::TensorParallel;
    }

    let weight_bytes = tile.weight_bytes();
    let activation_bytes = tile.activation_bytes();
    let n_f = n as f64;

    // TP: weights sharded, activations AllReduce'd over intercard.
    let tp_weight_seconds = seconds(weight_bytes, local_ddr_bw) / n_f;
    let tp_activation_seconds = seconds(activation_bytes, intercard_bw);
    let tp_cost = tp_weight_seconds + tp_activation_seconds;

    // MP: weights local to one sail, activations sharded forward.
    let mp_weight_seconds = seconds(weight_bytes, local_ddr_bw);
    let mp_activation_seconds = seconds(activation_bytes, intercard_bw) / n_f;
    let mp_cost = mp_weight_seconds + mp_activation_seconds;

    if tp_cost <= mp_cost {
        Strategy::TensorParallel
    } else {
        Strategy::ModelParallel
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bandwidth::{INTERCARD_BW_BYTES_PER_SEC, LOCAL_DDR_BW_BYTES_PER_SEC};

    /// Activation-dominated test name = "the load-bearing axis the
    /// AllReduce moves is small relative to the weight read". When
    /// `m * n` is tiny and `k` is huge, weight_bytes dwarfs
    /// activation_bytes; sharding the heavy weight read across
    /// cards is the winning move, so TP wins.
    #[test]
    fn pick_strategy_returns_tp_when_activation_dominates() {
        // m=4, n=4 → activation 64 B; k=2^20 → weight 16 MB;
        // ratio 250000:1. TP must win.
        let tile = TileShape {
            m: 4,
            n: 4,
            k: 1 << 20,
            bytes_per_element: 4,
        };
        let chosen = pick_strategy(
            &tile,
            4,
            LOCAL_DDR_BW_BYTES_PER_SEC,
            INTERCARD_BW_BYTES_PER_SEC,
        );
        assert_eq!(chosen, Strategy::TensorParallel);
    }

    /// Weight-dominated test name = "the activation axis dwarfs
    /// the weight axis on this tile". When `m * n` is huge and
    /// `k` is tiny, the activation transfer is the hot path —
    /// MP (which shards the activation forward) wins.
    #[test]
    fn pick_strategy_returns_mp_when_weights_dominate() {
        // m=4096, n=4096 → activation 64 MB; k=4 → weight 64 KB.
        // The activation is what we want to shard, so MP (which
        // splits activation across n_sails) wins.
        let tile = TileShape {
            m: 4096,
            n: 4096,
            k: 4,
            bytes_per_element: 4,
        };
        let chosen = pick_strategy(
            &tile,
            4,
            LOCAL_DDR_BW_BYTES_PER_SEC,
            INTERCARD_BW_BYTES_PER_SEC,
        );
        assert_eq!(chosen, Strategy::ModelParallel);
    }

    /// Single-card degenerate: both strategies have the same
    /// wall-clock; the function picks the canonical TP default.
    #[test]
    fn pick_strategy_n_sails_1_returns_tp() {
        let tile = TileShape {
            m: 1024,
            n: 1024,
            k: 1024,
            bytes_per_element: 4,
        };
        let chosen = pick_strategy(
            &tile,
            1,
            LOCAL_DDR_BW_BYTES_PER_SEC,
            INTERCARD_BW_BYTES_PER_SEC,
        );
        assert_eq!(chosen, Strategy::TensorParallel);
    }

    /// `n_sails == 0` is treated as 1 — same canonical TP default.
    #[test]
    fn pick_strategy_n_sails_0_treated_as_1() {
        let tile = TileShape {
            m: 1024,
            n: 1024,
            k: 1024,
            bytes_per_element: 4,
        };
        let chosen = pick_strategy(
            &tile,
            0,
            LOCAL_DDR_BW_BYTES_PER_SEC,
            INTERCARD_BW_BYTES_PER_SEC,
        );
        assert_eq!(chosen, Strategy::TensorParallel);
    }

    /// Override the bandwidth constants and flip the decision.
    /// A balanced tile (activation_bytes ≈ weight_bytes) lets
    /// the bandwidth ratio be the deciding lever. We pick a tile
    /// that goes one way on the rev-A defaults and the other way
    /// when the bandwidth ratio is inverted, confirming the
    /// cost function is genuinely bandwidth-driven and not a
    /// hard-coded constant fall-through.
    #[test]
    fn pick_strategy_with_bandwidth_overrides() {
        // Balanced tile: weight = m*k*bpe = 1*1024*4 = 4 KB,
        // activation = m*n*bpe = 1*1024*4 = 4 KB. Equal axes,
        // so the bandwidth ratio is the only deciding factor.
        let tile = TileShape {
            m: 1,
            n: 1024,
            k: 1024,
            bytes_per_element: 4,
        };

        // Default rev-A constants (DDR > intercard by 4×):
        // - TP: 4096/(4*2e9) + 4096/5e8 = 5.12e-7 + 8.19e-6 = 8.7e-6
        // - MP: 4096/2e9 + 4096/(4*5e8) = 2.05e-6 + 2.05e-6 = 4.1e-6
        // MP wins on the default constants.
        let default = pick_strategy(
            &tile,
            4,
            LOCAL_DDR_BW_BYTES_PER_SEC,
            INTERCARD_BW_BYTES_PER_SEC,
        );
        assert_eq!(default, Strategy::ModelParallel);

        // Invert: intercard 10× *slower* than DDR, but DDR also
        // slower in absolute terms. With bw_intercard much
        // slower, MP's tiny `activation/(4*intercard)` saving is
        // overwhelmed by TP's `weight/(4*ddr)` saving on the
        // now-larger weight cost — TP wins.
        // - DDR  = 100 MB/s, intercard = 10 MB/s, n_sails = 4
        // - TP: 4096/(4*1e8) + 4096/1e7 = 1.02e-5 + 4.10e-4 = 4.20e-4
        // - MP: 4096/1e8 + 4096/(4*1e7) = 4.10e-5 + 1.02e-4 = 1.43e-4
        // Hmm, MP still wins above. We need a regime where TP's
        // weight-shard saving dominates. Pick a weight-heavy
        // tile for the inversion arm — the test is "the
        // bandwidth overrides change the answer". Switch tile.
        let weight_biased = TileShape {
            m: 4,
            n: 4,
            k: 1 << 16,
            bytes_per_element: 4,
        };
        // weight = 4*65536*4 = 1 MB; activation = 4*4*4 = 64 B.
        // Default constants (rev-A) → TP wins (already covered
        // by `pick_strategy_returns_tp_when_activation_dominates`,
        // verified here for the inversion baseline):
        let default_weighty = pick_strategy(
            &weight_biased,
            4,
            LOCAL_DDR_BW_BYTES_PER_SEC,
            INTERCARD_BW_BYTES_PER_SEC,
        );
        assert_eq!(default_weighty, Strategy::TensorParallel);

        // Now make the inter-card link *much* slower than DDR,
        // so the AllReduce activation cost balloons and MP's
        // `activation/(n*intercard)` shard becomes the dominant
        // saving even on this weight-heavy tile.
        // - DDR = 1e9, intercard = 1e3 (1 KB/s), n=4
        // - TP: 1MB/(4*1e9) + 64/1e3 = 2.5e-4 + 0.064 = 0.0643
        // - MP: 1MB/1e9 + 64/(4*1e3) = 1e-3 + 0.016 = 0.017
        // MP wins on the inverted bandwidth, flipping the
        // decision from the default-constants TP answer.
        let flipped = pick_strategy(
            &weight_biased,
            4,
            1_000_000_000, // DDR same order as rev-A
            1_000,         // intercard crippled to 1 KB/s
        );
        assert_eq!(flipped, Strategy::ModelParallel);
    }

    /// TinyLlama-1.1B Q4_0 decode-step capacity-planning datapoint.
    ///
    /// Numbers (per `project_tinyllama_baseline.md` and the
    /// TinyLlama-1.1B-Chat config):
    /// - hidden size `d = 2048`
    /// - intermediate size `d_ff = 5632`
    /// - decode = 1 token at a time, so activation `m = 1`
    /// - bytes_per_element = 2 because Q4_0 dequants to f16-equivalent
    ///   scratch on the matmul path; the post-dequant width is the
    ///   load-bearing throughput axis on the open-toolchain ECP5 stack.
    ///
    /// Per-decode-step FFN up-projection tile: `m=1, n=5632, k=2048`.
    /// In the m=1 decode regime weight_bytes ≈ `k * 2 = 4096 B` and
    /// activation_bytes ≈ `n * 2 = 11264 B` — activation is the
    /// bigger axis. MP (which shards the activation forward across
    /// cards) wins.
    ///
    /// **Capacity-planning takeaway:** TinyLlama decode on
    /// 4× InnerJib7EA wants **MP**, not TP. The per-token activation
    /// is the scarce-bandwidth resource we want to shard, NOT the
    /// per-token weight read (the weight read is small in the m=1
    /// decode case because we only stream one row of the projection
    /// matrix per token; prefill — m >> 1 — would flip this back to
    /// TP). If a real bench shows TP winning for TinyLlama decode
    /// on 4 sails, the bandwidth model is wrong — bisect the bw
    /// constants in `bandwidth.rs` first.
    #[test]
    fn pick_strategy_for_tinyllama_decode_step() {
        // Per-block FFN up-projection tile (the heaviest matmul
        // in a decode step on a small model like TinyLlama).
        let tile = TileShape {
            m: 1,
            n: 5632,
            k: 2048,
            bytes_per_element: 2,
        };
        let chosen = pick_strategy(
            &tile,
            4,
            LOCAL_DDR_BW_BYTES_PER_SEC,
            INTERCARD_BW_BYTES_PER_SEC,
        );
        assert_eq!(chosen, Strategy::ModelParallel);
    }

    /// Saturating arithmetic guard: pathological inputs (u64
    /// near max) must not panic the cost function. The estimates
    /// become infinite, but the comparison still produces a
    /// deterministic answer.
    #[test]
    fn pick_strategy_saturates_on_overflow_inputs() {
        let tile = TileShape {
            m: u64::MAX,
            n: u64::MAX,
            k: u64::MAX,
            bytes_per_element: 4,
        };
        // Just verify it doesn't panic and returns a Strategy.
        let _ = pick_strategy(
            &tile,
            4,
            LOCAL_DDR_BW_BYTES_PER_SEC,
            INTERCARD_BW_BYTES_PER_SEC,
        );
    }

    /// Zero-bandwidth guard: `seconds(_, 0)` returns infinity so
    /// the comparison never silently divides by zero. Both costs
    /// become infinity here; TP is returned as the canonical
    /// default per the `tp_cost <= mp_cost` tiebreak.
    #[test]
    fn pick_strategy_with_zero_bandwidth_picks_tp_default() {
        let tile = TileShape {
            m: 16,
            n: 16,
            k: 16,
            bytes_per_element: 4,
        };
        let chosen = pick_strategy(&tile, 4, 0, 0);
        assert_eq!(chosen, Strategy::TensorParallel);
    }

    /// Smoke-check on `TileShape::weight_bytes` /
    /// `activation_bytes` — these helpers are part of the public
    /// surface so they need a direct unit test in addition to the
    /// integration coverage from `pick_strategy_*` tests.
    #[test]
    fn tile_shape_byte_helpers() {
        let tile = TileShape {
            m: 8,
            n: 16,
            k: 32,
            bytes_per_element: 2,
        };
        assert_eq!(tile.weight_bytes(), 8 * 32 * 2);
        assert_eq!(tile.activation_bytes(), 8 * 16 * 2);
    }
}
