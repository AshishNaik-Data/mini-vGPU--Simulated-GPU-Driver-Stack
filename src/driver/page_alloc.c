#include "page_alloc.h"

#include <errno.h>
#include <stdlib.h>

#define WORD_BITS 64u

static inline uint32_t word_index(uint32_t pfn) { return pfn / WORD_BITS; }
static inline uint64_t bit_mask(uint32_t pfn) { return 1ull << (pfn % WORD_BITS); }

int vgpu_pa_init(struct vgpu_page_alloc *pa, uint32_t npages)
{
    if (npages == 0)
        return -EINVAL;
    uint32_t nwords = (npages + WORD_BITS - 1) / WORD_BITS;
    pa->bits = calloc(nwords, sizeof(uint64_t));
    if (!pa->bits)
        return -ENOMEM;
    pa->npages = npages;
    pa->used   = 0;
    return 0;
}

void vgpu_pa_destroy(struct vgpu_page_alloc *pa)
{
    free(pa->bits);
    pa->bits   = NULL;
    pa->npages = 0;
    pa->used   = 0;
}

int vgpu_pa_is_used(const struct vgpu_page_alloc *pa, uint32_t pfn)
{
    return (pa->bits[word_index(pfn)] & bit_mask(pfn)) != 0;
}

static void set_range(struct vgpu_page_alloc *pa, uint32_t pfn, uint32_t count, int used)
{
    for (uint32_t i = pfn; i < pfn + count; i++) {
        if (used)
            pa->bits[word_index(i)] |= bit_mask(i);
        else
            pa->bits[word_index(i)] &= ~bit_mask(i);
    }
}

/*
 * First fit. When a used page is found inside the candidate window we jump
 * past it instead of advancing by one, so each page is examined O(1) times
 * per window start.
 *
 * TODO(exercise): skip whole words with `bits[w] == ~0ull` and use
 * __builtin_ctzll() to find the next free bit, then compare with best fit.
 */
int64_t vgpu_pa_alloc(struct vgpu_page_alloc *pa, uint32_t count, uint32_t align_pages)
{
    if (count == 0 || count > pa->npages)
        return -EINVAL;
    if (align_pages == 0)
        align_pages = 1;
    if ((align_pages & (align_pages - 1)) != 0)
        return -EINVAL;

    uint64_t start = 0;
    while (1) {
        start = (start + align_pages - 1) & ~(uint64_t)(align_pages - 1);
        if (start + count > pa->npages)
            return -ENOMEM;

        uint32_t i = 0;
        while (i < count && !vgpu_pa_is_used(pa, (uint32_t)start + i))
            i++;

        if (i == count) {
            set_range(pa, (uint32_t)start, count, 1);
            pa->used += count;
            return (int64_t)start;
        }
        start += i + 1; /* resume after the used page we hit */
    }
}

void vgpu_pa_free(struct vgpu_page_alloc *pa, uint32_t pfn, uint32_t count)
{
    set_range(pa, pfn, count, 0);
    pa->used -= count;
}

uint32_t vgpu_pa_count_used(const struct vgpu_page_alloc *pa)
{
    uint32_t nwords = (pa->npages + WORD_BITS - 1) / WORD_BITS;
    uint32_t total  = 0;
    for (uint32_t w = 0; w < nwords; w++)
        total += (uint32_t)__builtin_popcountll(pa->bits[w]);
    return total;
}

uint32_t vgpu_pa_largest_free_run(const struct vgpu_page_alloc *pa)
{
    uint32_t best = 0, run = 0;
    for (uint32_t i = 0; i < pa->npages; i++) {
        if (vgpu_pa_is_used(pa, i)) {
            run = 0;
        } else if (++run > best) {
            best = run;
        }
    }
    return best;
}
