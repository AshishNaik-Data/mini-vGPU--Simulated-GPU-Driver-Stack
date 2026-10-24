/*
 * aligned_alloc.h - aligned malloc/free written by hand.
 *
 * This is the classic NVIDIA interview question ("write an aligned malloc").
 * The device uses it to allocate VRAM on a page boundary.
 */
#ifndef VGPU_ALIGNED_ALLOC_H
#define VGPU_ALIGNED_ALLOC_H

#include <stddef.h>

/* alignment must be a power of two. Returns NULL on failure or bad arguments. */
void *vgpu_aligned_malloc(size_t size, size_t alignment);

/* Accepts NULL, like free(). Only pass pointers from vgpu_aligned_malloc(). */
void vgpu_aligned_free(void *ptr);

#endif /* VGPU_ALIGNED_ALLOC_H */
