#include "mmu.h"

#include <errno.h>
#include <stdlib.h>

#include "vgpu/uapi.h"

/* ---- TLB: fully associative, LRU replacement ------------------------------
 *
 * A real TLB compares all entries in parallel (it is a CAM). With 16 entries a
 * linear scan is our software equivalent. The doubly linked list keeps entries
 * in recency order so "touch" and "evict least recently used" are both O(1).
 * This is the same structure as the LRU cache interview question.
 */

static void list_remove(struct vgpu_tlb *tlb, struct vgpu_tlb_entry *e)
{
    if (e->prev) e->prev->next = e->next; else tlb->head = e->next;
    if (e->next) e->next->prev = e->prev; else tlb->tail = e->prev;
    e->prev = e->next = NULL;
}

static void list_push_front(struct vgpu_tlb *tlb, struct vgpu_tlb_entry *e)
{
    e->prev = NULL;
    e->next = tlb->head;
    if (tlb->head) tlb->head->prev = e; else tlb->tail = e;
    tlb->head = e;
}

static void list_push_back(struct vgpu_tlb *tlb, struct vgpu_tlb_entry *e)
{
    e->next = NULL;
    e->prev = tlb->tail;
    if (tlb->tail) tlb->tail->next = e; else tlb->head = e;
    tlb->tail = e;
}

static void tlb_init(struct vgpu_tlb *tlb)
{
    tlb->head = tlb->tail = NULL;
    tlb->hits = tlb->misses = 0;
    for (uint32_t i = 0; i < VGPU_TLB_ENTRIES; i++) {
        tlb->entries[i].valid = 0;
        tlb->entries[i].large = 0;
        tlb->entries[i].prev = tlb->entries[i].next = NULL;
        list_push_back(tlb, &tlb->entries[i]);
    }
}

static uint64_t large_base(uint64_t vpn) { return vpn & ~(uint64_t)(VGPU_LARGE_PAGES - 1); }

/* Does this entry translate vpn? A large entry covers its whole 16-page group. */
static int covers(const struct vgpu_tlb_entry *e, uint64_t vpn)
{
    return e->valid && (e->large ? e->vpn == large_base(vpn) : e->vpn == vpn);
}

static struct vgpu_tlb_entry *tlb_lookup(struct vgpu_tlb *tlb, uint64_t vpn)
{
    for (struct vgpu_tlb_entry *e = tlb->head; e; e = e->next) {
        if (covers(e, vpn)) {
            list_remove(tlb, e);
            list_push_front(tlb, e);
            return e;
        }
    }
    return NULL;
}

static void tlb_insert(struct vgpu_tlb *tlb, uint64_t vpn, vgpu_pte pte)
{
    /* Invalid entries are kept at the tail, so the tail is always the victim. */
    struct vgpu_tlb_entry *victim = tlb->tail;
    list_remove(tlb, victim);
    victim->large = (pte & VGPU_PTE_LARGE) != 0;
    if (victim->large) {
        /* Store the group's first page, so any page in it can be computed. */
        uint64_t offset = vpn - large_base(vpn);
        vpn = large_base(vpn);
        pte -= (vgpu_pte)(offset << VGPU_PTE_PFN_SHIFT);
    }
    victim->vpn   = vpn;
    victim->pte   = pte;
    victim->valid = 1;
    list_push_front(tlb, victim);
}

/* At most one entry covers a vpn: entries are only added after a lookup missed. */
static void tlb_invalidate(struct vgpu_tlb *tlb, uint64_t vpn)
{
    for (struct vgpu_tlb_entry *e = tlb->head; e; e = e->next) {
        if (covers(e, vpn)) {
            e->valid = 0;
            list_remove(tlb, e);
            list_push_back(tlb, e);
            return;
        }
    }
}

/* ---- page table ----------------------------------------------------------- */

int vgpu_mmu_init(struct vgpu_mmu *mmu, uint32_t nvpages)
{
    mmu->ptes = calloc(nvpages, sizeof(vgpu_pte));
    if (!mmu->ptes)
        return -ENOMEM;
    mmu->nvpages = nvpages;
    mmu->faults  = 0;
    tlb_init(&mmu->tlb);
    return 0;
}

void vgpu_mmu_destroy(struct vgpu_mmu *mmu)
{
    free(mmu->ptes);
    mmu->ptes    = NULL;
    mmu->nvpages = 0;
}

int vgpu_mmu_map(struct vgpu_mmu *mmu, uint64_t vpn, uint32_t pfn, uint32_t count, uint32_t flags)
{
    if (vpn == 0 || vpn + count > mmu->nvpages) /* page 0 stays unmapped: NULL faults */
        return -EINVAL;
    if (pfn > (UINT32_MAX >> VGPU_PTE_PFN_SHIFT) - count)
        return -EINVAL;
    if ((flags & VGPU_PTE_LARGE) &&
        (vpn % VGPU_LARGE_PAGES || pfn % VGPU_LARGE_PAGES || count % VGPU_LARGE_PAGES))
        return -EINVAL; /* a large page must be a whole, aligned group */

    for (uint32_t i = 0; i < count; i++)
        if (mmu->ptes[vpn + i] & VGPU_PTE_VALID)
            return -EEXIST;

    for (uint32_t i = 0; i < count; i++)
        mmu->ptes[vpn + i] = ((pfn + i) << VGPU_PTE_PFN_SHIFT) | flags | VGPU_PTE_VALID;
    return 0;
}

void vgpu_mmu_unmap(struct vgpu_mmu *mmu, uint64_t vpn, uint32_t count)
{
    for (uint32_t i = 0; i < count; i++) {
        mmu->ptes[vpn + i] = 0;
        /* A stale TLB entry would let the engine keep using freed memory. */
        tlb_invalidate(&mmu->tlb, vpn + i);
    }
}

static int lookup_pte(struct vgpu_mmu *mmu, uint64_t vpn, vgpu_pte *pte)
{
    struct vgpu_tlb_entry *e = tlb_lookup(&mmu->tlb, vpn);
    if (e) {
        mmu->tlb.hits++;
        *pte = e->pte;
        if (e->large) /* advance from the group's first page to this one */
            *pte += (vgpu_pte)((vpn - e->vpn) << VGPU_PTE_PFN_SHIFT);
        return 0;
    }
    mmu->tlb.misses++;

    if (vpn >= mmu->nvpages || !(mmu->ptes[vpn] & VGPU_PTE_VALID)) {
        mmu->faults++;
        return -EFAULT;
    }
    *pte = mmu->ptes[vpn];
    tlb_insert(&mmu->tlb, vpn, *pte);
    return 0;
}

int vgpu_mmu_translate(struct vgpu_mmu *mmu, uint64_t va, int write, uint64_t *pa)
{
    vgpu_pte pte;
    int err = lookup_pte(mmu, va >> VGPU_PAGE_SHIFT, &pte);
    if (err)
        return err;
    if (write && !(pte & VGPU_PTE_WRITE)) {
        mmu->faults++;
        return -EFAULT;
    }
    uint64_t pfn = pte >> VGPU_PTE_PFN_SHIFT;
    *pa = (pfn << VGPU_PAGE_SHIFT) | (va & VGPU_PAGE_MASK);
    return 0;
}

int vgpu_mmu_translate_range(struct vgpu_mmu *mmu, uint64_t va, uint64_t size, int write, uint64_t *pa)
{
    if (size == 0 || va + size < va) /* empty or wraps around */
        return -EINVAL;

    uint64_t first_pa;
    int err = vgpu_mmu_translate(mmu, va, write, &first_pa);
    if (err)
        return err;

    uint64_t first_vpn = va >> VGPU_PAGE_SHIFT;
    uint64_t last_vpn  = (va + size - 1) >> VGPU_PAGE_SHIFT;
    for (uint64_t vpn = first_vpn + 1; vpn <= last_vpn; vpn++) {
        uint64_t page_pa;
        err = vgpu_mmu_translate(mmu, vpn << VGPU_PAGE_SHIFT, write, &page_pa);
        if (err)
            return err;
        uint64_t expected = (first_pa & ~(uint64_t)VGPU_PAGE_MASK) + ((vpn - first_vpn) << VGPU_PAGE_SHIFT);
        if (page_pa != expected)
            return -EFAULT; /* not physically contiguous */
    }
    *pa = first_pa;
    return 0;
}
