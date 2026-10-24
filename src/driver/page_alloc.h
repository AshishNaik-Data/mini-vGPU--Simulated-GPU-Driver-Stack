/*
 * page_alloc.h - physical page allocator for device memory (VRAM).
 *
 * One bit per 4 KiB page: 1 = used, 0 = free. Allocations are physically
 * contiguous runs of pages, optionally aligned to a power-of-two page count.
 *
 * Not thread-safe by itself: the device holds its memory lock around calls.
 */
#ifndef VGPU_PAGE_ALLOC_H
#define VGPU_PAGE_ALLOC_H

#include <stdint.h>

struct vgpu_page_alloc {
    uint64_t *bits;
    uint32_t  npages;
    uint32_t  used;
};

int  vgpu_pa_init(struct vgpu_page_alloc *pa, uint32_t npages);
void vgpu_pa_destroy(struct vgpu_page_alloc *pa);

/* Returns the first page frame number (>= 0), or -ENOMEM / -EINVAL. */
int64_t vgpu_pa_alloc(struct vgpu_page_alloc *pa, uint32_t count, uint32_t align_pages);
void    vgpu_pa_free(struct vgpu_page_alloc *pa, uint32_t pfn, uint32_t count);

int      vgpu_pa_is_used(const struct vgpu_page_alloc *pa, uint32_t pfn);
uint32_t vgpu_pa_count_used(const struct vgpu_page_alloc *pa);       /* popcount over the bitmap */
uint32_t vgpu_pa_largest_free_run(const struct vgpu_page_alloc *pa); /* fragmentation metric     */

#endif /* VGPU_PAGE_ALLOC_H */
