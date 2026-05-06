// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! Integration smoke test for the runtime library.
//!
//! Only meaningful when `/dev/spankerctl` is present (i.e. when the
//! `spanker.ko` kernel module is loaded). Skips otherwise so that
//! `cargo test --workspace` passes on any developer host — CI does
//! not currently insmod the kernel module (deferred to a follow-up
//! that wires DKMS into the runner).

use spanker_runtime::{SpankerControl, ABI_VERSION_MAJOR, ABI_VERSION_MINOR, CONTROL_DEVICE_PATH};

#[test]
fn version_matches_built_constants_when_device_present() {
    if !std::path::Path::new(CONTROL_DEVICE_PATH).exists() {
        eprintln!("skipping: {CONTROL_DEVICE_PATH} not present (load spanker.ko first)");
        return;
    }

    let ctl = match SpankerControl::open() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: cannot open {CONTROL_DEVICE_PATH}: {e}");
            return;
        }
    };

    let v = ctl.version().expect("GET_VERSION ioctl failed");
    assert_eq!(v.major, ABI_VERSION_MAJOR, "ABI major mismatch");
    assert_eq!(v.minor, ABI_VERSION_MINOR, "ABI minor mismatch");

    ctl.ping().expect("PING ioctl failed");
}
