// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PopSolutions Cooperative

//! bindgen build script for `ggml-spanker`.
//!
//! Generates `$OUT_DIR/bindings.rs` from `wrapper.h`, which itself
//! `#include`s the spanker UAPI header at
//! `src/driver/include/uapi/spanker_ioctl.h`.
//!
//! See `wrapper.h` for the rationale on which symbols are bound
//! today vs deferred to the kernel-driver work-submit PR.

use std::env;
use std::path::PathBuf;

fn main() {
    // Resolve the spanker UAPI header relative to this crate. The
    // workspace layout is fixed (`src/backends/ggml/` ← crate,
    // `src/driver/include/uapi/` ← UAPI), so we walk up three
    // levels from CARGO_MANIFEST_DIR to land on the workspace
    // root and then descend into the driver's include path.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent() // src/backends
        .and_then(|p| p.parent()) // src
        .and_then(|p| p.parent()) // workspace root
        .expect("ggml-spanker is expected to live at <workspace>/src/backends/ggml")
        .to_path_buf();
    let uapi_dir = workspace_root.join("src/driver/include/uapi");
    let wrapper = manifest_dir.join("wrapper.h");

    // Tell cargo to re-run only when the inputs change.
    println!("cargo:rerun-if-changed={}", wrapper.display());
    println!(
        "cargo:rerun-if-changed={}",
        uapi_dir.join("spanker_ioctl.h").display()
    );

    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg(format!("-I{}", uapi_dir.display()))
        // Allow the spanker UAPI symbols only — keeps the binding
        // surface tight and avoids pulling in the host's libc /
        // kernel headers transitively pulled by `<linux/types.h>`
        // and `<linux/ioctl.h>`.
        .allowlist_type("spanker_.*")
        .allowlist_var("SPANKER_.*")
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
        // Suppress bindgen's compile-time size/align assertions.
        // Enabling them would require the test runner to link
        // against libclang at `cargo test` time (the generated
        // assertions reference C-side `sizeof`/`alignof` values),
        // which is heavier than this crate's UAPI surface
        // justifies. The cross-mirror test
        // `bindgen_uapi_constants_match_runtime_mirror` in
        // `src/lib.rs` covers `size_of::<ffi::spanker_version>()`
        // (the only struct currently bound) at unit-test time
        // instead. If WORK_SUBMIT or another non-trivial struct
        // is later added to the UAPI header, revisit this and
        // either flip layout_tests back on or extend the mirror.
        .layout_tests(false)
        .generate()
        .expect("bindgen failed to generate spanker UAPI bindings");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}
