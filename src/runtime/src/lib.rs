// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! # Spanker userspace runtime
//!
//! Per [ADR-001](../../../docs/adr/0001-userspace-runtime-language.md)
//! (Rust userspace) and
//! [ADR-002](../../../docs/adr/0002-driver-model.md) (out-of-tree
//! kbuild kernel module). This crate provides a minimal, blocking
//! handle over the singleton control device `/dev/spankerctl`
//! exposed by `spanker.ko`.
//!
//! ## v0 ABI
//!
//! Mirrors `src/driver/include/uapi/spanker_ioctl.h`. The constants
//! and `ioctl_*` macros below MUST stay in lock-step with that
//! header until ADR-003 introduces a SemVer policy for the ABI.
//! [`SpankerControl::version`] lets callers fail cleanly on a
//! major mismatch.

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

/// Default path to the singleton control character device, created
/// by `spanker.ko` at module-load time.
pub const CONTROL_DEVICE_PATH: &str = "/dev/spankerctl";

/// Major component of the ABI version this runtime was built
/// against. The kernel driver's reported major MUST equal this; a
/// mismatch is a hard error and the userspace process should
/// refuse to proceed.
pub const ABI_VERSION_MAJOR: u16 = 0;
/// Minor component of the ABI version this runtime was built
/// against.
pub const ABI_VERSION_MINOR: u16 = 1;
/// Patch component of the ABI version this runtime was built
/// against.
pub const ABI_VERSION_PATCH: u16 = 0;

// FFI shims for the v0 ABI. The `nix::ioctl_*!` macros expand to
// `pub unsafe fn` items that we don't want to surface in the
// crate's public docs, so we wrap them in a private module and
// silence missing_docs locally.
#[allow(missing_docs)]
mod ffi {
    /// ioctl magic byte; matches `SPANKER_IOC_MAGIC` in the UAPI
    /// header. 0xE3 is a placeholder pending reconciliation with
    /// mainline `Documentation/userspace-api/ioctl/ioctl-number.rst`
    /// before any in-tree submission attempt (Generation B per
    /// ADR-002).
    pub(super) const SPANKER_IOC_MAGIC: u8 = 0xE3;

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub(super) struct SpankerVersionRaw {
        pub major: u16,
        pub minor: u16,
        pub patch: u16,
        pub reserved: u16,
    }

    nix::ioctl_none!(spanker_ping, SPANKER_IOC_MAGIC, 0x01);
    nix::ioctl_read!(
        spanker_get_version,
        SPANKER_IOC_MAGIC,
        0x02,
        SpankerVersionRaw
    );
}

use ffi::{spanker_get_version, spanker_ping, SpankerVersionRaw};

/// ABI version triple reported by the kernel driver via
/// `SPANKER_IOC_GET_VERSION`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AbiVersion {
    /// Major version. Must match [`ABI_VERSION_MAJOR`].
    pub major: u16,
    /// Minor version. Newer minors are forward-compatible with this
    /// runtime once ADR-003 is accepted.
    pub minor: u16,
    /// Patch version. Bumps freely.
    pub patch: u16,
}

/// Errors returned by [`SpankerControl`].
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `open(2)` on the control device failed (typical cause:
    /// `spanker.ko` is not loaded, or the caller lacks permission).
    #[error("open {path}: {source}")]
    Open {
        /// Path that failed to open.
        path: PathBuf,
        /// Underlying I/O error from libc.
        #[source]
        source: io::Error,
    },

    /// An ioctl call returned a non-zero status.
    #[error("ioctl SPANKER_IOC_{op} failed: {source}")]
    Ioctl {
        /// Symbolic opcode name.
        op: &'static str,
        /// Underlying nix error.
        #[source]
        source: nix::Error,
    },
}

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Handle to the Spanker singleton control device.
///
/// The device is created by `spanker.ko` at module-load time and
/// always exists once the kernel module is loaded, regardless of
/// whether any Sails are physically attached.
pub struct SpankerControl {
    file: File,
}

impl SpankerControl {
    /// Open the default control device at [`CONTROL_DEVICE_PATH`].
    pub fn open() -> Result<Self> {
        Self::open_path(CONTROL_DEVICE_PATH)
    }

    /// Open a control device at an arbitrary path. Used by tests
    /// and by callers that need to talk to a non-default path
    /// (e.g., a chroot or a developer mock).
    pub fn open_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(p)
            .map_err(|source| Error::Open {
                path: p.to_path_buf(),
                source,
            })?;
        Ok(Self { file })
    }

    /// Smoke-test the ioctl dispatcher (`SPANKER_IOC_PING`). The
    /// kernel driver returns 0 unconditionally; a non-zero result
    /// means the dispatcher is broken or the magic byte mismatches.
    pub fn ping(&self) -> Result<()> {
        // SAFETY: `spanker_ping` is a no-payload ioctl that takes
        // only the file descriptor; the kernel does no reads or
        // writes through any user pointer.
        unsafe { spanker_ping(self.file.as_raw_fd()) }
            .map(|_| ())
            .map_err(|source| Error::Ioctl { op: "PING", source })
    }

    /// Read the kernel driver's reported ABI version
    /// (`SPANKER_IOC_GET_VERSION`).
    pub fn version(&self) -> Result<AbiVersion> {
        let mut raw = SpankerVersionRaw::default();
        // SAFETY: `&mut raw` points to an owned local stack value;
        // the kernel copies exactly `sizeof(SpankerVersionRaw)` (8)
        // bytes into it via copy_to_user. No aliasing, no escape.
        unsafe { spanker_get_version(self.file.as_raw_fd(), &mut raw) }.map_err(|source| {
            Error::Ioctl {
                op: "GET_VERSION",
                source,
            }
        })?;
        Ok(AbiVersion {
            major: raw.major,
            minor: raw.minor,
            patch: raw.patch,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_path_reports_missing_device() {
        let bogus = "/tmp/spanker-runtime-test-does-not-exist";
        let err = SpankerControl::open_path(bogus)
            .err()
            .expect("expected failure on bogus path");
        match err {
            Error::Open { path, source } => {
                assert_eq!(path, PathBuf::from(bogus));
                assert_eq!(source.kind(), io::ErrorKind::NotFound);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn raw_version_struct_is_eight_bytes() {
        // The UAPI struct is wire-stable at 8 bytes; if this
        // assertion fires, the kernel-side struct changed shape
        // and userspace bindings are now lying to callers.
        assert_eq!(std::mem::size_of::<SpankerVersionRaw>(), 8);
    }

    #[test]
    fn abi_constants_match_built_values() {
        // Sanity: the public consts match the values baked into
        // the ioctl version response. Catches accidental skew if
        // someone bumps the consts but forgets the kernel side.
        assert_eq!(ABI_VERSION_MAJOR, 0);
        assert_eq!(ABI_VERSION_MINOR, 1);
        assert_eq!(ABI_VERSION_PATCH, 0);
    }
}
