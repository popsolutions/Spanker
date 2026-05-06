/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 PopSolutions Cooperative */

/*
 * bindgen entry point for the ggml-spanker crate.
 *
 * Pulls in the spanker driver's userspace ioctl header so that the
 * Rust side can re-derive the wire-stable types and ioctl numbers
 * from the canonical C definitions instead of hand-mirroring them.
 *
 * Today this wraps:
 *   - struct spanker_version          (8-byte version triple)
 *   - SPANKER_IOC_MAGIC               (0xE3 placeholder)
 *   - SPANKER_ABI_VERSION_{MAJOR,MINOR,PATCH}
 *   - SPANKER_IOC_PING / _GET_VERSION numbers
 *
 * SPANKER_IOC_WORK_SUBMIT is *not* in the UAPI header yet — it
 * lands in the kernel-driver PR that wires DDR3-backed work
 * dispatch (cross-stream issue Spanker #9). When that PR ships,
 * this wrapper.h does not need to change: the new ioctl number
 * and its descriptor struct will appear automatically.
 */

#ifndef _SPANKER_GGML_BINDGEN_WRAPPER_H
#define _SPANKER_GGML_BINDGEN_WRAPPER_H

#include "spanker_ioctl.h"

#endif /* _SPANKER_GGML_BINDGEN_WRAPPER_H */
