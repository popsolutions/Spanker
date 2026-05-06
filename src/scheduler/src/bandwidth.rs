// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! Bandwidth model constants for the rev-A topology.
//!
//! Numbers are realistic ceilings on the open-toolchain ECP5 stack,
//! NOT theoretical maxima. Sources:
//!
//! - [`LOCAL_DDR_BW_BYTES_PER_SEC`]: production references
//!   (OrangeCrab, Trellis Board, Versa-ECP5) per Stays
//!   `docs/upstream-contributions/2026-05-06-litedram-ecp5.md` —
//!   measured 192–300 MT/s in the wild on the open ECP5 toolchain
//!   even though `litedram/phy/ecp5ddrphy.py` advertises 800 MT/s.
//!   Realistic ceiling 1.5–2.4 GB/s; midpoint 2.0 GB/s.
//! - [`INTERCARD_BW_BYTES_PER_SEC`]: InnerJib7EA pinout §9 — 4 lanes
//!   × ~1.25 Gbps × 8b/10b = ~4 Gbps effective ≈ 500 MB/s per
//!   direction.
//!
//! ## Why this module exists
//!
//! The Spanker scheduler's TP-vs-MP partitioning logic must compare
//! local DDR throughput to inter-card link throughput when deciding
//! whether a tile is cheaper to fetch from a peer card or to recompute
//! locally. Hard-coding theoretical numbers (the stale ADR-001
//! "DDR3-1600 = 12.8 GB/s" claim) would over-shard tensor-parallel
//! workloads that are actually inter-card-fetch bound on rev-A.
//!
//! ## When to bump these constants
//!
//! - **`LOCAL_DDR_BW_BYTES_PER_SEC`:** when rev-B targets a faster
//!   ECP5 build (e.g. sys_clk_freq pushed to 100 MHz / DRAM 200 MHz =
//!   400 MT/s on x16 = 800 MB/s, or rev-C moves to DDR4) — bump in
//!   lockstep with the platform ADR.
//! - **`INTERCARD_BW_BYTES_PER_SEC`:** when ADR-014 (line coding)
//!   chooses 64b/66b instead of 8b/10b, this becomes ~600 MB/s. Bump
//!   in the same PR that lands ADR-014.
//!
//! Cross-references:
//! - MAST issue #32 — ADR-001 amendment to correct the 12.8 GB/s claim
//! - Stays PR #26 (merged 2026-05-06) — LiteDRAM ECP5 recon
//! - InnerJib7EA PR #11 — connector pinout §9 bandwidth math

/// Realistic local DDR3 bandwidth ceiling on rev-A
/// (ECP5-85F + open toolchain + DDR3L SO-DIMM), in bytes per second.
///
/// **Value:** `2_000_000_000` (2.0 GB/s).
///
/// This is the midpoint of the 1.5–2.4 GB/s range observed across the
/// three production reference designs that drive `ECP5DDRPHY`
/// (OrangeCrab, Trellis Board, Lattice Versa-ECP5). It is **not** the
/// theoretical DDR3-1600 ceiling (12.8 GB/s) — that number assumed an
/// IO clock the open ECP5 toolchain cannot meet on production
/// silicon, see Stays
/// `docs/upstream-contributions/2026-05-06-litedram-ecp5.md`.
///
/// Used by the scheduler's TP-vs-MP decision logic to estimate the
/// per-card data-feed budget. Pair with
/// [`INTERCARD_BW_BYTES_PER_SEC`] for collective-op cost models.
pub const LOCAL_DDR_BW_BYTES_PER_SEC: u64 = 2_000_000_000;

/// Per-direction inter-card link bandwidth ceiling on rev-A
/// (4 lanes × 1.25 Gbps × 8b/10b coding), in bytes per second.
///
/// **Value:** `500_000_000` (500 MB/s).
///
/// Per InnerJib7EA `docs/hw/intercard-connector-pinout.md` §9. When
/// ADR-014 lands and selects 64b/66b coding the effective rate becomes
/// ~600 MB/s — bump this constant in the ADR-014 PR.
///
/// Used by the scheduler's collective-ops cost model
/// (AllReduce, AllGather) to decide whether sharding a tile across
/// cards is cheaper than recomputing it locally.
pub const INTERCARD_BW_BYTES_PER_SEC: u64 = 500_000_000;

#[cfg(test)]
mod tests {
    // These tests are deliberately constant-vs-constant comparisons:
    // the `bandwidth` module's contract is "these specific u64 values
    // are the rev-A ceilings", so the load-bearing guard *is* the
    // assertion-on-constants. Without this allow, clippy would force
    // us to refactor each guard through `core::hint::black_box` or
    // similar opacity tricks, which would only obscure intent without
    // changing the behaviour: the tests would still fail at the
    // exact same boundary if someone reverted the constants.
    #![allow(clippy::assertions_on_constants)]

    use super::*;

    /// Regression guard: the stale ADR-001 number was 12.8 GB/s
    /// (1.6 GT/s × 8 bytes on x64 DDR3-1600). If anyone reverts this
    /// constant to that value — or anywhere near half of it — the
    /// TP/MP cost model will over-shard tensor-parallel workloads on
    /// rev-A. This test is the load-bearing artefact preventing a
    /// silent regression to the theoretical-peak number.
    ///
    /// We assert `LOCAL_DDR_BW < 4 GB/s` because:
    /// - the realistic ceiling range is 1.5–2.4 GB/s (Stays recon)
    /// - the 800 MT/s PHY-header ceiling on x64 = 6.4 GB/s, so any
    ///   value above 4 GB/s implies someone reverted to a
    ///   not-yet-validated regime
    /// - 4 GB/s leaves headroom for an honest rev-B bump to ~3 GB/s
    ///   without tripping the guard.
    #[test]
    fn local_ddr_bw_is_not_the_stale_12_8_gb_per_sec_value() {
        // Stale theoretical value the scheduler must never silently
        // adopt — see MAST #32 (ADR-001 amendment).
        const STALE_DDR3_1600_THEORETICAL_BYTES_PER_SEC: u64 = 12_800_000_000;
        assert!(
            LOCAL_DDR_BW_BYTES_PER_SEC < STALE_DDR3_1600_THEORETICAL_BYTES_PER_SEC,
            "LOCAL_DDR_BW_BYTES_PER_SEC ({LOCAL_DDR_BW_BYTES_PER_SEC}) must be the realistic \
             ECP5+open-toolchain ceiling, not the theoretical DDR3-1600 number"
        );
        // Generous upper bound that allows future rev-B bumps but
        // rejects the 12.8 GB/s and 6.4 GB/s theoretical numbers.
        assert!(
            LOCAL_DDR_BW_BYTES_PER_SEC < 4_000_000_000,
            "LOCAL_DDR_BW_BYTES_PER_SEC ({LOCAL_DDR_BW_BYTES_PER_SEC}) is above the \
             realistic-ceiling envelope; rev-B bumps must update this guard explicitly"
        );
    }

    /// Regression guard: the realistic ceiling range is 1.5–2.4 GB/s
    /// per the three production reference designs. The midpoint
    /// 2.0 GB/s is the chosen modelling value.
    #[test]
    fn local_ddr_bw_is_inside_observed_range() {
        // 1.5 GB/s lower bound (OrangeCrab @ 192 MT/s on x64).
        assert!(LOCAL_DDR_BW_BYTES_PER_SEC >= 1_500_000_000);
        // 2.4 GB/s upper bound (production-validated 300 MT/s on x64).
        assert!(LOCAL_DDR_BW_BYTES_PER_SEC <= 2_400_000_000);
    }

    /// Pin the inter-card constant to the doc-stated 500 MB/s number
    /// (8b/10b coding baseline, see InnerJib7EA pinout §9). Bump in
    /// lockstep with ADR-014 when 64b/66b is chosen.
    #[test]
    fn intercard_bw_matches_innerjib7ea_pinout_section_9() {
        assert_eq!(INTERCARD_BW_BYTES_PER_SEC, 500_000_000);
    }

    /// Pin the topology of the bandwidth model: local DDR is the
    /// higher-throughput resource per card by a comfortable margin.
    /// The `≥4×` factor is what the TP-vs-MP decision logic relies on
    /// — for any tile that fits in local DDR but would require
    /// inter-card fetch, the local path is at least 4× faster.
    ///
    /// On rev-A the constants land at exactly 4× (2.0 GB/s vs
    /// 500 MB/s with 8b/10b coding); the assertion uses `>=` so that
    /// the rev-A baseline passes the guard. If a future bump (rev-B
    /// faster intercard, rev-C slower DDR) breaks this invariant,
    /// the TP/MP heuristics need to be re-derived; the test forces
    /// that conversation rather than letting it slip silently.
    #[test]
    fn local_ddr_dominates_intercard_by_at_least_4x() {
        assert!(
            LOCAL_DDR_BW_BYTES_PER_SEC >= 4 * INTERCARD_BW_BYTES_PER_SEC,
            "LOCAL_DDR_BW ({LOCAL_DDR_BW_BYTES_PER_SEC}) must be >= 4 × INTERCARD_BW \
             ({INTERCARD_BW_BYTES_PER_SEC}); TP-vs-MP heuristics depend on this margin"
        );
    }

    /// Sanity: both constants are non-zero. A zero would silently
    /// turn any throughput-divided cost estimate into a divide-by-zero
    /// or an infinity, producing nonsense scheduling decisions.
    #[test]
    fn constants_are_positive() {
        assert!(LOCAL_DDR_BW_BYTES_PER_SEC > 0);
        assert!(INTERCARD_BW_BYTES_PER_SEC > 0);
    }
}
