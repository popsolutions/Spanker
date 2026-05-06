// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! # spanker-scheduler — distributed scheduler for the PopSolutions Sails
//!
//! Per `project_multicard_parallelism.md` (multi-card parallelism is
//! a first-class architectural requirement) and the cross-stream
//! contract with `popsolutions/MAST` (intercard skeleton, MAST #14)
//! and `popsolutions/Stays` (PCB connector pinout).
//!
//! ## What this crate exposes
//!
//! - [`Topology`]: opaque handle over the connected Sails plus
//!   their inter-card link graph. Generic over the per-sail
//!   handle type so unit tests can drive a [`MockSail`] vector
//!   without `/dev/spanker*` present.
//! - [`AllReduce`], [`AllGather`]: collective-ops trait surfaces
//!   for the multi-card data path. Implemented host-side on
//!   [`Topology<MockSail>`] for now (real implementations land
//!   when the kernel ABI gains work-submission ioctls and the
//!   inter-card link protocol is specified per ADR-014).
//! - [`TensorParallel`], [`ModelParallel`]: marker / shape traits
//!   the runtime consumes when partitioning workloads. Bodies
//!   land alongside real-device matmul (PR #5b) and inter-card
//!   link bandwidth characterisation (cross-stream issue against
//!   MAST filed alongside this PR).
//! - Inter-card constants imported from MAST #14: see
//!   [`intercard`] module.
//! - Bandwidth-model constants for rev-A capacity planning: see
//!   [`bandwidth`] module. These supersede the stale ADR-001
//!   "DDR3-1600 = 12.8 GB/s" number with realistic
//!   ECP5+open-toolchain ceilings (cross-stream MAST #32, this
//!   crate's issues #14 and #21). Three tiers are modelled:
//!   local DDR > inter-card > host-link.
//!
//! ## Bandwidth-model usage
//!
//! ```
//! use spanker_scheduler::{
//!     LOCAL_DDR_BW_BYTES_PER_SEC,
//!     INTERCARD_BW_BYTES_PER_SEC,
//!     HOST_LINK_BW_BYTES_PER_SEC,
//! };
//! // Three-tier hierarchy: local DDR > inter-card > host link.
//! // The TP-vs-MP per-token decision logic relies on the first
//! // inequality; session-level cost-budget logic relies on the
//! // second to recognise the host link as the slowest hop.
//! assert!(LOCAL_DDR_BW_BYTES_PER_SEC > INTERCARD_BW_BYTES_PER_SEC);
//! assert!(INTERCARD_BW_BYTES_PER_SEC > HOST_LINK_BW_BYTES_PER_SEC);
//! ```

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod bandwidth;
pub mod collective;
pub mod decision;
pub mod intercard;
pub mod topology;

pub use bandwidth::{
    HOST_LINK_BW_BYTES_PER_SEC, INTERCARD_BW_BYTES_PER_SEC, LOCAL_DDR_BW_BYTES_PER_SEC,
};
pub use collective::{AllGather, AllReduce, ModelParallel, ReduceOp, TensorParallel};
pub use decision::{pick_strategy, Strategy, TileShape};
pub use intercard::{Link, LinkState, INTERCARD_BUS_WIDTH, INTERCARD_LANES, INTERCARD_LANE_WIDTH};
pub use topology::{MockSail, Topology};

/// Errors returned by this crate.
///
/// Marked `#[non_exhaustive]` because library `Error` enums
/// almost always grow new variants; downstream `match` arms
/// must include a `_ =>` catch-all so we can extend without a
/// major-version semver bump.
///
/// # Regression guard
///
/// The following doctest fails to compile *because* `Error` is
/// `#[non_exhaustive]`: a downstream exhaustive `match` is
/// rejected without a `_ =>` arm. If someone removes the
/// `#[non_exhaustive]` attribute the doctest will start to
/// compile, the `compile_fail` will fail, and CI will catch the
/// silent semver-evolution regression.
///
/// ```compile_fail
/// use spanker_scheduler::Error;
/// fn classify(e: Error) -> &'static str {
///     match e {
///         Error::NoSails => "no sails",
///         Error::TopologyMismatch { .. } => "topology",
///         Error::ShapeMismatch { .. } => "shape",
///         Error::NotImplemented(_) => "not impl",
///         Error::Runtime(_) => "runtime",
///     }
/// }
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// `Topology::enumerate()` found no `/dev/spanker*` device
    /// nodes (typical cause: `spanker.ko` is not loaded, or the
    /// driver's PCIe probe path has not yet been wired up to
    /// create per-Sail nodes).
    #[error("no Sails enumerated; spanker.ko may not be loaded or no devices probed yet")]
    NoSails,

    /// Per-card buffer count differs from the topology size.
    #[error("topology mismatch: expected {expected} sails, got {actual}")]
    TopologyMismatch {
        /// Number of sails the topology was built with.
        expected: usize,
        /// Number of per-card buffers the caller passed.
        actual: usize,
    },

    /// Per-card buffers have inconsistent shapes (collective ops
    /// require uniform shape across cards).
    #[error("buffer shape mismatch on sail {sail}: expected {expected} elems, got {actual}")]
    ShapeMismatch {
        /// Which sail's buffer disagrees.
        sail: usize,
        /// Shape of sail 0's buffer (the reference).
        expected: usize,
        /// Shape of the offending buffer.
        actual: usize,
    },

    /// Real-device collective op is not yet wired up.
    #[error("not implemented yet: {0}")]
    NotImplemented(&'static str),

    /// Underlying runtime error (open, ioctl).
    #[error(transparent)]
    Runtime(#[from] spanker_runtime::Error),
}

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;
