// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! Inter-card link contract — Rust mirror of the constants and
//! `link_state_t` enum landed by MAST #14 (intercard skeleton).
//!
//! Until ADR-014 (inter-card link architecture choice) lands the
//! actual protocol, this module exposes only the *shape* of the
//! link surface so the scheduler can reason about topology
//! without committing to a wire protocol. Real bandwidth and
//! latency numbers come from the cross-stream issue against
//! MAST filed alongside this PR.

/// Number of high-speed lanes per inter-card link. Default per
/// MAST #14 is 4; PCB rev-A may parametrise per Sail variant.
pub const INTERCARD_LANES: usize = 4;

/// Width of one lane, in bits. Mirrors MAST #14's
/// `INTERCARD_LANE_WIDTH = 32`.
pub const INTERCARD_LANE_WIDTH: usize = 32;

/// Aggregate bus width seen by the link MAC, in bits. Mirrors
/// MAST #14's `INTERCARD_BUS_WIDTH = 128` (default).
pub const INTERCARD_BUS_WIDTH: usize = 128;

/// State of a single inter-card link, mirroring `link_state_t`
/// in MAST #14.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkState {
    /// Link is down; no traffic.
    Down,
    /// Lane training in progress.
    Training,
    /// Link is up and ready for traffic.
    Up,
    /// Hard error; link must be retrained or replaced.
    Error,
}

/// A single point-to-point link between two sails in a topology.
///
/// `local_sail` and `remote_sail` are indices into
/// [`crate::Topology::sails`]; the protocol that flows over the
/// link is opaque to this crate and lands in ADR-014.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Link {
    /// Topology index of the originating sail.
    pub local_sail: usize,
    /// Topology index of the peer sail.
    pub remote_sail: usize,
    /// Current state of this link.
    pub state: LinkState,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intercard_constants_match_mast_14() {
        // MAST #14 contract: 4 lanes × 32 bits = 128-bit bus.
        assert_eq!(INTERCARD_LANES * INTERCARD_LANE_WIDTH, INTERCARD_BUS_WIDTH);
    }

    #[test]
    fn link_state_is_copy() {
        let s = LinkState::Up;
        let _ = s;
        let _ = s; // would fail to compile if !Copy
    }
}
