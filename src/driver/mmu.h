/*
 * mmu.h - per-context GPU virtual memory: a page table plus a small TLB.
 *
 * Each context (think: process) gets its own GPU virtual address space, so
 * one context can never read another's buffers. The "hardware" translates
 * every GPU VA through this MMU before touching memory.
 *
 *   GPU VA:  [ virtual page number (vpn) | 12-bit page offset ]
 *   PTE:     [ pfn (24 bits)             | flags (8 bits)     ]
 *
 * A single-level page table. Not thread-safe by itself: the device holds its
 * lock around every call.
 *
 * Large pages: 16 consecutive PTEs whose virtual and physical page numbers are
 * both 16-aligned can be marked VGPU_PTE_LARGE. The TLB then caches the whole
 * 64 KiB group in one entry, so 16 entries cover 1 MiB instead of 64 KiB.
 */
#ifndef VGPU_MMU_H
#define VGPU_MMU_H

#include <stdint.h>

#define VGPU_PTE_VALID     0x1u
#define VGPU_PTE_WRITE     0x2u
#define VGPU_PTE_LARGE     0x4u /* part of an aligned 64 KiB large page */
#define VGPU_PTE_PFN_SHIFT 8u

#define VGPU_TLB_ENTRIES   16u
#define VGPU_LARGE_PAGES   16u  /* 4 KiB pages per large page (64 KiB) */

typedef uint32_t vgpu_pte;

struct vgpu_tlb_entry {
    uint64_t vpn;   /* for a large entry: the first vpn of the group  */
    vgpu_pte pte;   /* for a large entry: the PTE of the first page   */
    int      valid;
    int      large;
    struct vgpu_tlb_entry *prev, *next; /* LRU list: head = most recently used */
};

struct vgpu_tlb {
    struct vgpu_tlb_entry  entries[VGPU_TLB_ENTRIES];
    struct vgpu_tlb_entry *head, *tail;
    uint64_t hits, misses;
};

struct vgpu_mmu {
    vgpu_pte       *ptes;    /* indexed by vpn */
    uint32_t        nvpages; /* size of the VA space in pages */
    struct vgpu_tlb tlb;
    uint64_t        faults;
};

int  vgpu_mmu_init(struct vgpu_mmu *mmu, uint32_t nvpages);
void vgpu_mmu_destroy(struct vgpu_mmu *mmu);

/* With VGPU_PTE_LARGE in flags, vpn, pfn and count must all be multiples of 16. */
int  vgpu_mmu_map(struct vgpu_mmu *mmu, uint64_t vpn, uint32_t pfn, uint32_t count, uint32_t flags);
void vgpu_mmu_unmap(struct vgpu_mmu *mmu, uint64_t vpn, uint32_t count); /* also flushes the TLB */

/* Translate one address. Returns 0, or -EFAULT on a page fault. */
int vgpu_mmu_translate(struct vgpu_mmu *mmu, uint64_t va, int write, uint64_t *pa);

/*
 * Translate [va, va + size). Every page must be mapped and the physical pages
 * must be contiguous, so the engine can access the range through one pointer.
 * TODO(exercise): support scatter-gather (non-contiguous) mappings instead.
 */
int vgpu_mmu_translate_range(struct vgpu_mmu *mmu, uint64_t va, uint64_t size, int write, uint64_t *pa);

#endif /* VGPU_MMU_H */
