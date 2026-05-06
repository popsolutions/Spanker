// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! Topology over connected Sails plus their inter-card links.
//!
//! [`Topology`] is generic over the per-sail handle type so unit
//! tests can drive a [`MockSail`] vector without `/dev/spanker*`
//! being present. The real-device path is implemented on
//! `Topology<SpankerControl>`; the mock path is on
//! `Topology<MockSail>`.

use std::path::PathBuf;

use spanker_runtime::SpankerControl;

use crate::intercard::{Link, LinkState};
use crate::{Error, Result};

/// In-process mock for one Sail. Carries an `id` so collective-op
/// implementations on `Topology<MockSail>` can disambiguate cards
/// when reducing host-side. Tests should construct mock
/// topologies via [`Topology::with_mock`] rather than this type
/// directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MockSail {
    id: usize,
}

impl MockSail {
    /// Construct a new mock sail with the given index.
    pub fn new(id: usize) -> Self {
        Self { id }
    }

    /// Topology index of this mock sail.
    pub fn id(&self) -> usize {
        self.id
    }
}

/// Topology over connected Sails plus their inter-card link graph.
///
/// Generic over the per-sail handle type; defaults to
/// [`SpankerControl`] for the real-device path. Tests use
/// [`Topology<MockSail>`] via [`Topology::with_mock`].
pub struct Topology<H = SpankerControl> {
    sails: Vec<H>,
    links: Vec<Link>,
}

impl Topology<SpankerControl> {
    /// Walk `/dev/spanker0..N` opening each as a [`SpankerControl`]
    /// per ADR-002. Stops at the first index where `/dev/spankerN`
    /// does not exist; returns [`Error::NoSails`] if none were
    /// found.
    ///
    /// Inter-card link discovery is a follow-up — until ADR-014
    /// pins the link protocol, the returned `Topology` has an
    /// empty `links` vector even when multiple sails are present.
    pub fn enumerate() -> Result<Self> {
        let mut sails = Vec::new();
        for index in 0..256 {
            let path = PathBuf::from(format!("/dev/spanker{index}"));
            if !path.exists() {
                break;
            }
            sails.push(SpankerControl::open_path(&path)?);
        }
        if sails.is_empty() {
            return Err(Error::NoSails);
        }
        Ok(Self {
            sails,
            links: Vec::new(),
        })
    }
}

impl Topology<MockSail> {
    /// Construct a fully-meshed mock topology with `n_sails`
    /// cards, all inter-card links in [`LinkState::Up`].
    pub fn with_mock(n_sails: usize) -> Self {
        let sails = (0..n_sails).map(MockSail::new).collect();
        // n*(n-1) directed edges in a fully-meshed graph
        // (each of n nodes has a directed link to each of the
        // n-1 other nodes).
        let mut links = Vec::with_capacity(n_sails * n_sails.saturating_sub(1));
        for local in 0..n_sails {
            for remote in 0..n_sails {
                if local == remote {
                    continue;
                }
                links.push(Link {
                    local_sail: local,
                    remote_sail: remote,
                    state: LinkState::Up,
                });
            }
        }
        Self { sails, links }
    }
}

impl<H> Topology<H> {
    /// Number of sails in this topology.
    pub fn n_sails(&self) -> usize {
        self.sails.len()
    }

    /// Inter-card links in this topology.
    pub fn links(&self) -> &[Link] {
        &self.links
    }
}

#[cfg(test)]
impl<H> Topology<H> {
    /// Per-sail handles in topology-index order. Test-only
    /// accessor used by the topology's own unit tests; the
    /// public surface deliberately keeps the handle vector
    /// opaque so future refactors can change its representation.
    pub(crate) fn sails_for_tests(&self) -> &[H] {
        &self.sails
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_reports_no_sails_when_dev_missing() {
        // /dev/spanker0 won't exist on a host without spanker.ko
        // loaded; the cargo test runner is not root, so we cannot
        // create test nodes. This guards the empty-iteration path.
        // Use match instead of expect_err to avoid requiring Debug
        // on Topology<SpankerControl>'s success type.
        let path = std::path::Path::new("/dev/spanker0");
        if path.exists() {
            // CI host happens to have the module loaded — skip.
            return;
        }
        match Topology::enumerate() {
            Err(Error::NoSails) => {}
            Err(other) => panic!("expected NoSails, got {other:?}"),
            Ok(_) => panic!("expected NoSails, got Ok"),
        }
    }

    #[test]
    fn with_mock_zero_sails_has_no_links() {
        let t = Topology::<MockSail>::with_mock(0);
        assert_eq!(t.n_sails(), 0);
        assert!(t.links().is_empty());
    }

    #[test]
    fn with_mock_two_sails_has_two_links() {
        let t = Topology::<MockSail>::with_mock(2);
        assert_eq!(t.n_sails(), 2);
        // Fully-meshed directed: (0->1) and (1->0).
        assert_eq!(t.links().len(), 2);
        for link in t.links() {
            assert_eq!(link.state, LinkState::Up);
            assert_ne!(link.local_sail, link.remote_sail);
        }
    }

    #[test]
    fn with_mock_four_sails_has_twelve_links() {
        let t = Topology::<MockSail>::with_mock(4);
        assert_eq!(t.n_sails(), 4);
        // n*(n-1) = 12 directed links in a fully-meshed topology.
        assert_eq!(t.links().len(), 12);
    }

    #[test]
    fn mock_sail_carries_id() {
        let t = Topology::<MockSail>::with_mock(3);
        for (i, sail) in t.sails_for_tests().iter().enumerate() {
            assert_eq!(sail.id(), i);
        }
    }
}
