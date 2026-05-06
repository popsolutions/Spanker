<!--
SPDX-License-Identifier: CC-BY-SA-4.0
Copyright (c) 2026 PopSolutions Cooperative
-->

# ADR-002 — Linux driver model

- **Status:** Accepted
- **Date:** 2026-05-05
- **Stream:** stream-3 (Software Stack — Spanker)
- **Authors:** Agent 3 (Software Stack)
- **Supersedes:** —
- **Superseded by:** —
- **Related:** ADR-001 (runtime language); ADR-003 (public interface
  contracts; to be opened)

## Context

The Spanker kernel driver mediates between the Linux PCIe subsystem
and the userspace runtime defined by ADR-001. It owns:

- PCIe device probe / enumeration of one or more Sails (vendor + device
  ID match, BAR mapping, IRQ allocation).
- The ioctl interface used by the Rust userspace runtime to dispatch
  work, manage buffers, and read telemetry.
- DMA ring management for command submission and result retrieval.
- Multi-card peer awareness: when N Sails are present in the same host,
  the driver must expose them as `/dev/spanker0` … `/dev/spankerN-1`
  with stable ordering, so the userspace scheduler (PR #4) can build a
  consistent topology view.

Several kernel-side mechanisms are available for FPGA / accelerator
drivers; they differ in maturity, IOMMU dependency, and how much logic
lives in kernel space vs userspace:

1. **Out-of-tree kernel module** (`kbuild` + DKMS packaging). Owned by
   us, shipped with the project. Standard Linux character device with
   ioctl ABI.
2. **In-tree mainline submission** under `drivers/accel/` or a
   dedicated subsystem. Long upstream review cycle; locks the ABI to
   the kernel's stable-API guarantees.
3. **UIO (Userspace I/O)**: thin in-kernel shim that maps BARs to
   userspace; the actual driver lives in userspace.
4. **VFIO** with vfio-pci: framework for assigning a PCI device to a
   userspace process, requires a working IOMMU group.
5. **Rust-for-Linux kernel module**: write the kmod in Rust against the
   kernel's evolving Rust bindings (mainline since 6.1).

Project constraints that bear on the choice:

- **Pre-silicon hardware.** No `lspci -nn` line for our vendor/device
  exists in any production kernel database. We cannot submit upstream
  for a device that has not yet taped out. Mainline review cycles also
  take 9–12 months — incompatible with the MVP cadence.
- **Global South target hosts.** The driver must work on cheap
  consumer-grade x86_64 and aarch64 boards. We cannot assume IOMMU is
  enabled in firmware, in mainline kernels for those boards, or that
  end users have permissions to flip BIOS settings. This rules VFIO
  out for the MVP.
- **DMA descriptor management is non-trivial.** Allocation, mapping,
  cache flushing, completion, and teardown across multiple cards.
  Doing it all from userspace via UIO requires re-implementing pieces
  the kernel already provides (`dma_alloc_coherent`,
  `dma_map_sg`, `dma_unmap_sg`, IRQ handling). Cost > benefit.
- **Multi-card parallelism is first-class** (per project memory). The
  driver must expose a coherent multi-device topology that the
  scheduler can rely on; that wants kernel-side coordination.
- **Solo development mode** (per project memory). Verification budget
  is finite. We need a path where we can iterate the ABI quickly, ship
  reproducibly to test boards, and not block on an upstream maintainer.
- **License posture.** Code that links Linux kernel headers must be
  GPL-2.0-only (per `LICENSE`). The ADR scope here does not change the
  cooperative's broader Apache-2.0 default for userspace components;
  it only confirms the per-file SPDX header for the driver tree.

## Decision

**Out-of-tree kernel module** (`kbuild`-built, GPL-2.0-only) is the
driver model for the MVP and Generation A silicon. The driver lives
under `src/driver/` and is packaged for end users via **DKMS** so that
each kernel upgrade rebuilds it automatically without manual
intervention.

Specifically:

- Single source tree; one `Makefile` invoking the kernel's kbuild
  system (`obj-m`, `make -C $(KDIR) M=$(PWD) modules`).
- Per-file `SPDX-License-Identifier: GPL-2.0-only` on every `.c` /
  `.h` under `src/driver/`.
- Character device family: `/dev/spankerN` (one per probed Sail), with
  device numbering driven by PCIe enumeration order anchored to a
  per-card serial read from a known BAR offset. The serial-anchored
  ordering keeps device names stable across reboots even when PCIe
  enumeration order changes.
- ioctl ABI is the single, versioned boundary between this driver and
  the Rust runtime. The ABI lives in a header shared between kernel
  and userspace (`src/driver/include/uapi/spanker_ioctl.h`) and is
  governed by ADR-003 (Public interface contracts) — to be opened.
- KUnit for in-kernel unit tests; cocotb-driven integration tests
  against `MAST/verif/axi4_mem_model` exercise the driver's DMA path
  end-to-end (mocked PCIe BAR via a kernel test harness or a
  user-space process talking to a virtual PCI shim — chosen at PR #1
  time, not in this ADR).

UIO and VFIO are **deferred**, not foreclosed. A later ADR will
revisit them once Generation A silicon is in customer hands and the
operational profile is understood — specifically, when multi-tenant
inference workloads start asking for VM-grade isolation and the
install base has matured enough that we can require an IOMMU.

In-tree mainline submission is **deferred to Generation B**, once
silicon has shipped, the ABI has stabilised across at least one minor
revision, and we have a stable upstream story for the GGML backend.

Rust-for-Linux is **monitored, not adopted**, for Generation A. The
PCIe / DMA bindings are still expanding (as of kernel 6.10–6.11); we
revisit when those bindings cover what we need without `unsafe`
escape hatches in our hot paths.

## Considered alternatives

### In-tree mainline driver (deferred)

- ✅ Free packaging; ships with every distro kernel.
- ✅ Permanent stable-API guarantees mean less ongoing maintenance.
- ❌ Upstream review cycle is 9–12 months minimum and requires
  silicon availability for maintainers to test against; our MVP is
  pre-silicon.
- ❌ Locks the ABI to kernel stable-API rules before we know what the
  ABI should look like in production.

### UIO + userspace driver (rejected)

- ✅ Simplest in-kernel surface; almost all logic lives in Rust
  userspace where ADR-001 already lives.
- ❌ DMA descriptor management is much harder from userspace without
  rebuilding what the kernel already provides; we'd add complexity to
  save a small amount of kbuild code.
- ❌ IRQ delivery and coalescing in UIO is coarse compared to a
  bespoke driver's ability to batch completions.
- ❌ Multi-card peer-to-peer DMA setup needs kernel-side coordination
  that UIO does not natively provide.

### VFIO + vfio-pci (rejected for MVP)

- ✅ Strong isolation; VM-grade device assignment.
- ❌ Requires functional IOMMU groups; not a safe assumption on the
  cheap Global South target hardware (per project mission).
- ❌ Operationally heavier (binding/unbinding from `vfio-pci`,
  managing IOMMU groups) than the MVP audience can absorb.
- 🔁 Reconsider in a later ADR when the multi-tenant deployment
  profile demands it.

### Rust-for-Linux kernel module (rejected for Gen A)

- ✅ Aligns with ADR-001's userspace language choice.
- ❌ Kernel-side Rust API for PCIe + DMA is still in flux; we would
  end up with kernel-side `unsafe` blocks wrapping C bindings, which
  buys us nothing.
- ❌ Smaller pool of kernel reviewers fluent in Rust kmods reduces
  upstream optionality if we pivot to in-tree later.
- 🔁 Monitor and re-evaluate at Generation B.

## Consequences

### Positive

- Fast iteration on the ABI without upstream review pressure.
- DKMS packaging is well-trodden on Debian/Ubuntu/Arch/Manjaro
  derivatives — covers our Global South host targets.
- KUnit + cocotb integration gives us a credible verification story
  before silicon arrives.
- The ABI header is the single boundary between C and Rust, which
  ADR-003 will lock with SemVer rules.

### Negative

- Out-of-tree drivers carry maintenance cost across kernel API
  changes (DMA, IRQ, PCI subsystem refactors). We accept this as the
  price of fast iteration and budget review time per kernel LTS bump.
- Distros that don't run DKMS need a manual `make install` recipe —
  documented in `docs/install.md` (to be added with PR #1).
- We must publish driver tarballs per kernel-LTS major version
  alongside any release, increasing release-engineering surface area.

### Neutral

- License remains GPL-2.0-only per file under `src/driver/`,
  consistent with the top-level `LICENSE` carve-out.
- Userspace stays Apache-2.0 / Rust per ADR-001.

## Build & test posture

- `make -C src/driver` builds the kmod against the running kernel's
  headers (or `KDIR=` for cross-builds).
- `make -C src/driver tests` runs KUnit tests in-tree.
- Integration test harness lives under `tests/integration/driver/`
  and is invoked by CI on a kernel image fetched per the PR's
  matrix (current LTS + previous LTS at minimum).
- `dkms` recipe added with PR #1; CI verifies `dkms install` against
  a clean Debian image.

## Follow-ups (issues to open after merge)

- **ADR-003 — Public interface contracts** (ioctl numbers, Rust
  public API, GGML backend entry points). SemVer policy, deprecation
  windows, ABI testing.
- **PR #1 — PCIe driver skeleton.** Vendor/device ID stubs, kbuild
  scaffold, character device registration, ioctl dispatcher with one
  no-op `SPANKER_IOC_PING` entry, KUnit smoke test, DKMS recipe.
- **Issue — install/uninstall docs** under `docs/install.md`,
  covering manual `make install` and DKMS paths.
- **Issue — kernel-LTS support matrix** (initial: 6.1 LTS, 6.6 LTS,
  6.12 LTS).

## References

- Project memory (auto-loaded): `team_structure_4plus1.md`,
  `project_mission_and_open_fpga_commitment.md`,
  `project_multicard_parallelism.md`, `feedback_testing.md`,
  `feedback_solo_mode_and_pr_workflow.md`.
- ADR-001 — Userspace runtime language (Rust).
- [Linux kbuild documentation](https://www.kernel.org/doc/html/latest/kbuild/index.html).
- [DKMS](https://github.com/dell/dkms).
- [UIO subsystem](https://www.kernel.org/doc/html/latest/driver-api/uio-howto.html).
- [VFIO](https://www.kernel.org/doc/html/latest/driver-api/vfio.html).
- [Rust-for-Linux](https://rust-for-linux.com/).
- [KUnit](https://www.kernel.org/doc/html/latest/dev-tools/kunit/index.html).
