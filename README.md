<!--
SPDX-License-Identifier: CC-BY-SA-4.0
Copyright (c) 2026 PopSolutions Cooperative
-->

# Spanker — PopSolutions Sails software stack

> *In a tall ship, the **spanker** sail is the aftmost driving sail —
> the one that holds the ship's direction at the helm. The Spanker is
> the helmsman of the fleet.*

**Spanker** is the software counterpart to the PopSolutions Sails
hardware. It contains:

- **Linux kernel driver** — PCIe enumerator, ioctl interface, DMA
  management for the Sails accelerator boards
- **Userspace runtime** — process for orchestrating workloads, memory
  management, kernel binary loading
- **GGML inference backend** — port of GGML kernel ops to RVV +
  custom matrix extension (Xpop_matmul)
- **Distributed scheduler** — TP/MP/PP across N Sails (multi-card
  parallelism is first-class per project mission)

The first target Sail is `InnerJib7EA` (POPC_16A) — embedded entry
SBC. Spanker integrates with the AXI4 boundary defined in
`popsolutions/MAST` and validated by `popsolutions/InnerJib7EA`.

## Status

Bootstrap (2026-05-05). Skeleton landed; real work begins with the
first PR by Agent 3 (Software Stack) per the 4+1 operating model.

## License

- Software contributions: Apache 2.0 (kernel driver may be GPL-2.0-only
  if linking to Linux kernel headers requires; documented per file)
- Documentation: CC-BY-SA 4.0

See LICENSE for the full text.

## Layout

```
Spanker/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── src/
│   ├── driver/        (Linux kernel module — C)
│   ├── runtime/       (userspace orchestrator — Rust or C++; ADR-001 decides)
│   └── backends/
│       └── ggml/      (GGML kernel ports for Sails)
├── tests/             (pytest harness; integration tests against MAST cocotb sim)
├── docs/
│   ├── adr/           (architectural decisions)
│   └── api/           (versioned interface contracts: ioctl, runtime API, GGML)
└── .github/workflows/ (CI)
```

## Related repos

- [`popsolutions/MAST`](https://github.com/popsolutions/MAST) — RTL trunk
- [`popsolutions/InnerJib7EA`](https://github.com/popsolutions/InnerJib7EA) — first product target
- [`popsolutions/Stays`](https://github.com/popsolutions/Stays) — FPGA hardware

## Contributing

Per `popsolutions/MAST/GOVERNANCE.md`: cooperative-affiliate-only code
contribution. DCO sign-off on every commit.
