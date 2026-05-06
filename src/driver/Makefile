# SPDX-License-Identifier: GPL-2.0-only
# Copyright (c) 2026 PopSolutions Cooperative
#
# Top-level Makefile for the Spanker accelerator driver
# (out-of-tree kernel module). Per ADR-002 — Linux driver model.
#
# Usage:
#   make             # build spanker.ko against the running kernel
#   make KDIR=...    # cross-build against a specific kernel tree
#   make clean       # remove build artefacts
#
# Kbuild does the actual work; this Makefile only wraps the
# canonical "make -C $(KDIR) M=$(PWD)" recipe.

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

.PHONY: default modules clean help

default: modules

modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

help:
	@echo "Spanker driver — out-of-tree kbuild module"
	@echo ""
	@echo "Targets:"
	@echo "  modules   build spanker.ko (default)"
	@echo "  clean     remove build artefacts"
	@echo ""
	@echo "Variables:"
	@echo "  KDIR      kernel build tree (default: /lib/modules/\$$(uname -r)/build)"
