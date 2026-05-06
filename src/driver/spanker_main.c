// SPDX-License-Identifier: GPL-2.0-only
/*
 * Copyright (c) 2026 PopSolutions Cooperative
 *
 * Spanker — main module file.
 *
 * Per ADR-002 (out-of-tree kbuild kernel module), this file owns
 * module init/exit, the singleton control character device
 * /dev/spankerctl, and PCI driver registration.
 *
 * The skeleton creates /dev/spankerctl unconditionally so userspace
 * can verify the driver is loaded and read the v0 ABI even when no
 * Sails are physically attached. Per-Sail device nodes
 * /dev/spanker0..N-1 are added by a follow-up PR once the PCI probe
 * path is wired up against real silicon.
 *
 * The PCI driver registration uses placeholder vendor/device IDs
 * (0xDEAD/0xBEEF). PopSolutions has not yet been assigned a PCI
 * vendor ID; the follow-up issue cited in PR #1's description tracks
 * reconciliation before silicon ships.
 */

#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/version.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/pci.h>

#include "include/uapi/spanker_ioctl.h"

#define SPANKER_DRIVER_NAME  "spanker"
#define SPANKER_CTL_DEV_NAME "spankerctl"

/*
 * Placeholder PCI IDs. Must be replaced before silicon; tracked by
 * the follow-up issue cited in the PR #1 description.
 */
#define SPANKER_PCI_VENDOR_PLACEHOLDER 0xDEAD
#define SPANKER_PCI_DEVICE_PLACEHOLDER 0xBEEF

/* Defined in spanker_ioctl.c. */
extern const struct file_operations spanker_ctl_fops;

static const struct pci_device_id spanker_pci_ids[] = {
	{ PCI_DEVICE(SPANKER_PCI_VENDOR_PLACEHOLDER,
		     SPANKER_PCI_DEVICE_PLACEHOLDER) },
	{ 0 }
};
MODULE_DEVICE_TABLE(pci, spanker_pci_ids);

static int spanker_pci_probe(struct pci_dev *pdev,
			     const struct pci_device_id *id)
{
	/*
	 * Skeleton: log probe and accept. BAR mapping, IRQ allocation,
	 * and DMA-coherent ring setup land in follow-up PRs.
	 */
	pci_info(pdev,
		 "spanker: probed placeholder device (vendor=0x%04x device=0x%04x)\n",
		 id->vendor, id->device);
	return 0;
}

static void spanker_pci_remove(struct pci_dev *pdev)
{
	pci_info(pdev, "spanker: remove placeholder device\n");
}

static struct pci_driver spanker_pci_driver = {
	.name     = SPANKER_DRIVER_NAME,
	.id_table = spanker_pci_ids,
	.probe    = spanker_pci_probe,
	.remove   = spanker_pci_remove,
};

/* Singleton control device. */
static dev_t           spanker_ctl_dev_num;
static struct cdev     spanker_ctl_cdev;
static struct class   *spanker_ctl_class;
static struct device  *spanker_ctl_device;

static int __init spanker_init(void)
{
	int err;

	err = alloc_chrdev_region(&spanker_ctl_dev_num, 0, 1,
				  SPANKER_CTL_DEV_NAME);
	if (err < 0) {
		pr_err("spanker: alloc_chrdev_region failed: %d\n", err);
		return err;
	}

	cdev_init(&spanker_ctl_cdev, &spanker_ctl_fops);
	spanker_ctl_cdev.owner = THIS_MODULE;
	err = cdev_add(&spanker_ctl_cdev, spanker_ctl_dev_num, 1);
	if (err < 0) {
		pr_err("spanker: cdev_add failed: %d\n", err);
		goto unregister_region;
	}

#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 4, 0)
	spanker_ctl_class = class_create(THIS_MODULE, SPANKER_CTL_DEV_NAME);
#else
	spanker_ctl_class = class_create(SPANKER_CTL_DEV_NAME);
#endif
	if (IS_ERR(spanker_ctl_class)) {
		err = PTR_ERR(spanker_ctl_class);
		pr_err("spanker: class_create failed: %d\n", err);
		goto del_cdev;
	}

	spanker_ctl_device = device_create(spanker_ctl_class, NULL,
					   spanker_ctl_dev_num, NULL,
					   SPANKER_CTL_DEV_NAME);
	if (IS_ERR(spanker_ctl_device)) {
		err = PTR_ERR(spanker_ctl_device);
		pr_err("spanker: device_create failed: %d\n", err);
		goto destroy_class;
	}

	err = pci_register_driver(&spanker_pci_driver);
	if (err) {
		pr_err("spanker: pci_register_driver failed: %d\n", err);
		goto destroy_device;
	}

	pr_info("spanker: loaded (ABI v%u.%u.%u, /dev/%s ready)\n",
		SPANKER_ABI_VERSION_MAJOR,
		SPANKER_ABI_VERSION_MINOR,
		SPANKER_ABI_VERSION_PATCH,
		SPANKER_CTL_DEV_NAME);
	return 0;

destroy_device:
	device_destroy(spanker_ctl_class, spanker_ctl_dev_num);
destroy_class:
	class_destroy(spanker_ctl_class);
del_cdev:
	cdev_del(&spanker_ctl_cdev);
unregister_region:
	unregister_chrdev_region(spanker_ctl_dev_num, 1);
	return err;
}

static void __exit spanker_exit(void)
{
	pci_unregister_driver(&spanker_pci_driver);
	device_destroy(spanker_ctl_class, spanker_ctl_dev_num);
	class_destroy(spanker_ctl_class);
	cdev_del(&spanker_ctl_cdev);
	unregister_chrdev_region(spanker_ctl_dev_num, 1);
	pr_info("spanker: unloaded\n");
}

module_init(spanker_init);
module_exit(spanker_exit);

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("PopSolutions Cooperative");
MODULE_DESCRIPTION("Spanker accelerator driver (skeleton; PR #1)");
MODULE_VERSION("0.1.0");
