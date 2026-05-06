<!--
SPDX-License-Identifier: CC-BY-SA-4.0
Copyright (c) 2026 PopSolutions Cooperative
-->

# ADR-001 — Userspace runtime language

- **Status:** Accepted
- **Date:** 2026-05-05
- **Stream:** stream-3 (Software Stack — Spanker)
- **Authors:** Agent 3 (Software Stack)
- **Supersedes:** —
- **Superseded by:** —

## Context

Spanker is the software counterpart to the PopSolutions Sails accelerator
boards. It is composed of four cooperating layers:

1. **Linux kernel driver** — PCIe enumerator, ioctl interface, DMA ring
   management.
2. **Userspace runtime** — orchestrator process; loads kernel binaries
   onto the device, manages buffers, dispatches work, surfaces telemetry.
3. **Inference backend** — GGML kernel ports targeting the Sails compute
   unit (RVV vector + custom matrix extension `Xpop_matmul`).
4. **Distributed scheduler** — enumerates connected Sails, partitions
   workloads, drives collective ops (TP / MP / AllReduce) across N cards.

The kernel driver is C either way (Linux licensing contract; GPL-2.0-only
when linking kernel headers). The decision in scope here is the language
for the **userspace** layers — runtime, inference backend, and
distributed scheduler.

Project constraints that bear on the decision:

- **Multi-card parallelism is first-class** (per project memory). The
  scheduler must coordinate concurrent ioctl streams, peer-to-peer DMA,
  and collective ops across multiple devices. Memory-corruption bugs in
  this layer are catastrophic in production inference.
- **Mission alignment** (per project memory): the cooperative ships for
  the Global South; "precarious-but-available beats premium-but-locked".
  We need a toolchain that is freely usable on the cheapest available
  developer hardware (Manjaro / Debian / Ubuntu LTS, x86_64 + aarch64),
  with no vendor-licensed compilers in the path.
- **Solo development mode** (per project memory): a small team cannot
  afford to chase memory-safety bugs that a borrow-checker would have
  blocked at compile time. Verification budget goes to RTL, not to
  reproducing UAFs.
- **GGML reuse**: upstream GGML
  ([`ggml-org/ggml`](https://github.com/ggml-org/ggml)) is C/C++. Whatever
  we pick must call into GGML cheaply, since porting it is out of scope
  for the MVP.
- **Cooperative reciprocity**: the runtime is consumed by downstream
  Sails projects and (eventually) external operators. The build and
  packaging story has to be reproducible by someone who has never
  touched our code.

## Decision

**Rust** (edition 2024, MSRV `1.85.0`) for the userspace runtime, the
GGML backend wrapper, and the distributed scheduler.

The repository will be organised as a Cargo workspace whose members are:

```
src/runtime/        — orchestrator binary + library crate
src/backends/ggml/  — `ggml-spanker` crate; bindgen FFI over upstream GGML
src/scheduler/      — created by PR #4; collective-ops API + multi-card enum
```

The kernel driver under `src/driver/` stays in C, governed by
**ADR-002 — Driver model** (next ADR). The Rust runtime communicates
with the driver through a stable ioctl ABI documented under
`docs/api/` (versioned per ADR-003 — Public interface contracts; opened
as a follow-up).

`rust-toolchain.toml` will pin the MSRV at `1.85.0` (the floor for
edition 2024, stabilised in Rust 1.85 / 2025-02). `Cargo.lock` will be checked in
for the runtime binary; library crates (`ggml-spanker`,
`spanker-scheduler`) will not check in `Cargo.lock`, per current Cargo
guidance.

## Considered alternatives

### C++ (rejected)

- ✅ Cheapest GGML embedding — no FFI, just `#include`.
- ✅ Familiar to existing kernel/driver contributors.
- ❌ Memory safety on the multi-card scheduler is a manual exercise.
  `std::shared_ptr`, RAII, and modern guidelines reduce the surface but
  do not eliminate UAFs / data races, which is exactly the class of bug
  that would corrupt inference output silently and undetectably.
- ❌ Build story (CMake / Meson + dependency vendoring) is heavier than
  Cargo and harder to reproduce on a fresh Global South dev machine
  with patchy network.
- ❌ Async ergonomics (concurrent ioctl streams across N cards) lag Rust's
  `tokio` by a wide margin; we would end up reinventing executors.

### Pure C (rejected)

- ✅ Zero impedance with kernel headers and GGML.
- ❌ Distributed scheduler in C is a multi-month detour; the project
  cannot afford to rebuild what `tokio` + `Arc` + `Mutex` give us for
  free.
- ❌ Same memory-safety failure mode as C++ without C++'s RAII.

### Zig (rejected)

- ✅ Compelling for systems work; first-class C interop.
- ❌ Pre-1.0; ABI and stdlib churn is incompatible with our "ship the
  imperfect thing now and upgrade" cadence — we cannot afford to chase
  toolchain breaks.
- ❌ No production-grade async runtime; we would build the scheduler's
  concurrency primitives from scratch.

### Go (rejected)

- ✅ Excellent tooling, fast onboarding.
- ❌ GC pauses are incompatible with sub-millisecond inference scheduling
  and DMA descriptor management.
- ❌ FFI overhead (`cgo`) is significant on hot paths; we would still
  need a non-Go shim for GGML calls.

## Consequences

### Positive

- The borrow checker eliminates a class of bug (UAF, data race, dangling
  ioctl handle) that would otherwise consume scarce verification budget.
- Cargo workspace gives reproducible builds with one command:
  `cargo build --workspace`. CI is straightforward.
- `tokio` + `nix` provide production-grade async ioctl, signal handling,
  and PCIe peer enumeration without bespoke infrastructure.
- Rust-for-Linux is mainline since 6.1; the kernel-side trajectory aligns
  with our userspace choice if we ever migrate the driver in-tree later.
- Aligns with the open-source ecosystem we want to participate in:
  `tokenizers`, `candle`, `llama-rs`, `burn`, `vortex` are all Rust;
  contributing back is friction-free.

### Negative

- GGML calls go through `bindgen` FFI. We accept this overhead; the hot
  path is the device, not the host.
- Build pipeline is heterogeneous: Cargo for userspace, kbuild for the
  driver, and `bindgen` (with `libclang`) for the GGML FFI. CI must
  install all three. Documented in `.github/workflows/ci.yml` as the
  workspace lands.
- Contributors fluent only in C++ have a learning ramp. We accept this:
  the cooperative's contribution policy expects ramp-up time, and
  Rust's payoff is durable.

### Neutral

- The driver is and remains C. The Rust↔C boundary is the ioctl ABI,
  which is small, versioned, and tested. ADR-003 will pin the
  versioning policy.

## Build & test posture

- `cargo build --workspace --release` produces all userspace artefacts.
- `cargo test --workspace` runs unit tests; per project policy
  ("always write tests", `feedback_testing.md`), every PR adds tests.
- Integration tests against `MAST/verif/axi4_mem_model` (cocotb) live
  under `tests/integration/` and run via `pytest`. The Rust runtime is
  invoked as a subprocess from the harness, mirroring how downstream
  operators will call it.
- A future PR adds `clippy --workspace --all-targets -D warnings` and
  `rustfmt --check` to CI, gating merges.

## Follow-ups (issues to open after merge)

- **ADR-002 — Driver model**: out-of-tree kernel module first; UIO/VFIO
  evaluation for a later generation.
- **ADR-003 — Public interface contracts (ioctl, Rust public API, GGML
  backend entry points)**: versioning policy and backwards-compatibility
  rules across minor versions.
- **PR #1 — PCIe driver skeleton**: enumerator, ioctl stub, kbuild scaffold.
- **PR #2 — Userspace runtime library**: per this ADR; minimal API surface.
- **PR #3 — GGML int4 matmul kernel skeleton**: AXI4 transactions
  validated against `MAST/verif/axi4_mem_model`.
- **PR #4 — Distributed scheduler stub**: multi-card enumeration; TP/MP
  collective-op signatures (per `project_multicard_parallelism.md`).

## References

- Project memory (auto-loaded): `team_structure_4plus1.md`,
  `project_mission_and_open_fpga_commitment.md`,
  `project_multicard_parallelism.md`, `feedback_testing.md`,
  `feedback_solo_mode_and_pr_workflow.md`.
- [Rust-for-Linux](https://rust-for-linux.com/) — mainline since 6.1.
- [GGML upstream](https://github.com/ggml-org/ggml) — C/C++ inference engine.
- [`bindgen`](https://rust-lang.github.io/rust-bindgen/) — FFI generator.
- [`tokio`](https://tokio.rs/) — async runtime.
- [`nix`](https://crates.io/crates/nix) — `*nix` syscalls including
  `ioctl_*` macros.
