// SPDX-License-Identifier: GPL-2.0-only
/*
 * Copyright (c) 2026 PopSolutions Cooperative
 *
 * Spanker — ioctl dispatcher for /dev/spankerctl.
 *
 * Implements the v0 ABI declared in
 * <uapi/linux/spanker_ioctl.h>. The skeleton handles two opcodes:
 *
 *   SPANKER_IOC_PING        — no-op; returns 0 on success.
 *   SPANKER_IOC_GET_VERSION — copies struct spanker_version
 *                              {0,1,0,0} to userspace.
 *
 * Future PRs add per-device opcodes (buffer alloc/free, work
 * submit, completion poll) keyed off /dev/spanker0..N-1, governed
 * by ADR-003 (Public interface contracts).
 */

#include <linux/module.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/errno.h>
#include <linux/compat.h>

#include "include/uapi/spanker_ioctl.h"

static long spanker_ctl_ioctl(struct file *filp, unsigned int cmd,
			      unsigned long arg)
{
	if (_IOC_TYPE(cmd) != SPANKER_IOC_MAGIC)
		return -ENOTTY;

	switch (cmd) {
	case SPANKER_IOC_PING:
		return 0;

	case SPANKER_IOC_GET_VERSION: {
		struct spanker_version v = {
			.major    = SPANKER_ABI_VERSION_MAJOR,
			.minor    = SPANKER_ABI_VERSION_MINOR,
			.patch    = SPANKER_ABI_VERSION_PATCH,
			.reserved = 0,
		};
		if (copy_to_user((void __user *)arg, &v, sizeof(v)))
			return -EFAULT;
		return 0;
	}

	default:
		return -ENOTTY;
	}
}

const struct file_operations spanker_ctl_fops = {
	.owner          = THIS_MODULE,
	.open           = stream_open,
	.unlocked_ioctl = spanker_ctl_ioctl,
	.compat_ioctl   = compat_ptr_ioctl,
};
