/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */
/*
 * Copyright (c) 2026 PopSolutions Cooperative
 *
 * Public ioctl ABI for the Spanker accelerator driver.
 *
 * This header is shared between the in-kernel driver under
 * src/driver/ and the Rust userspace runtime (per ADR-001).
 *
 * Stability: v0 is unstable. The ABI may break between any two
 * commits while the major version is 0. The first stable major
 * version (v1) lands when ADR-003 (Public interface contracts) is
 * accepted and PR #2 ships the runtime that depends on it.
 *
 * Discoverability: SPANKER_IOC_GET_VERSION lets userspace fail
 * cleanly when it sees a kernel driver with an incompatible major.
 */

#ifndef _UAPI_LINUX_SPANKER_IOCTL_H
#define _UAPI_LINUX_SPANKER_IOCTL_H

#include <linux/types.h>
#include <linux/ioctl.h>

/*
 * ioctl magic byte.
 *
 * 0xE3 is used here as a placeholder; before any in-tree submission
 * (Generation B per ADR-002), this must be reconciled with
 * Documentation/userspace-api/ioctl/ioctl-number.rst in mainline.
 */
#define SPANKER_IOC_MAGIC 0xE3

/* ABI version reported by SPANKER_IOC_GET_VERSION. */
#define SPANKER_ABI_VERSION_MAJOR 0
#define SPANKER_ABI_VERSION_MINOR 1
#define SPANKER_ABI_VERSION_PATCH 0

/*
 * struct spanker_version — wire format of SPANKER_IOC_GET_VERSION.
 *
 * Fields are __u16 to keep the struct exactly 8 bytes regardless
 * of architecture or compiler padding choices. Reserved must be 0.
 */
struct spanker_version {
	__u16 major;
	__u16 minor;
	__u16 patch;
	__u16 reserved;
};

/*
 * Smoke-test the ioctl dispatcher.
 *
 * No payload, no return data; success indicates the driver is
 * loaded and the dispatcher reaches this opcode.
 */
#define SPANKER_IOC_PING        _IO(SPANKER_IOC_MAGIC, 0x01)

/*
 * Read driver/ABI version.
 *
 * Userspace passes a writable struct spanker_version; the kernel
 * fills it with SPANKER_ABI_VERSION_{MAJOR,MINOR,PATCH} and a
 * zeroed reserved field.
 */
#define SPANKER_IOC_GET_VERSION _IOR(SPANKER_IOC_MAGIC, 0x02, struct spanker_version)

#endif /* _UAPI_LINUX_SPANKER_IOCTL_H */
