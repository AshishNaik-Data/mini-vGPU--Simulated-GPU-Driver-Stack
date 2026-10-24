/* test_driver.c - tests for the vGPU driver. Run with `make test`. */
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

#include "aligned_alloc.h"
#include "mmu.h"
#include "ops.h"
#include "page_alloc.h"
#include "test_framework.h"
#include "vgpu/vgpu.h"

/* ---- building blocks ---------------------------------------------------------------- */

static void test_uapi_layout(void)
{
    CHECK_EQ(sizeof(struct vgpu_cmd_hdr), 16);
    CHECK_EQ(sizeof(struct vgpu_cmd), 40);
    CHECK_EQ(offsetof(struct vgpu_cmd, u.image.width), 32);
}

static void test_aligned_malloc(void)
{
    for (size_t align = 1; align <= 8192; align <<= 1) {
        uint8_t *p = vgpu_aligned_malloc(1000, align);
        CHECK(p != NULL);
        CHECK_EQ((uintptr_t)p % align, 0);
        memset(p, 0xAB, 1000); /* must be writable end to end (ASan checks this) */
        vgpu_aligned_free(p);
    }
    CHECK(vgpu_aligned_malloc(100, 3) == NULL);       /* not a power of two */
    CHECK(vgpu_aligned_malloc(SIZE_MAX, 16) == NULL); /* size would overflow */
    vgpu_aligned_free(NULL);                          /* like free(NULL) */
}

static void test_page_alloc(void)
{
    struct vgpu_page_alloc pa;
    CHECK_EQ(vgpu_pa_init(&pa, 100), 0);
    CHECK_EQ(vgpu_pa_alloc(&pa, 10, 1), 0);
    CHECK_EQ(vgpu_pa_alloc(&pa, 5, 8), 16);       /* next multiple of 8 after page 9 */
    CHECK_EQ(vgpu_pa_count_used(&pa), 15);        /* popcount agrees with the counter */
    CHECK_EQ(vgpu_pa_alloc(&pa, 90, 1), -ENOMEM);
    CHECK_EQ(vgpu_pa_alloc(&pa, 4, 3), -EINVAL);  /* alignment must be a power of two */
    CHECK_EQ(vgpu_pa_largest_free_run(&pa), 79);  /* pages 21..99 */

    vgpu_pa_free(&pa, 0, 10);
    CHECK_EQ(vgpu_pa_alloc(&pa, 6, 1), 0);        /* first fit reuses the hole at 0..15 */
    CHECK_EQ(vgpu_pa_alloc(&pa, 12, 1), 21);      /* the 10 pages left in it are too few */
    vgpu_pa_destroy(&pa);
}

static void test_mmu_translate(void)
{
    struct vgpu_mmu mmu;
    uint64_t pa;
    CHECK_EQ(vgpu_mmu_init(&mmu, 64), 0);
    CHECK_EQ(vgpu_mmu_map(&mmu, 0, 0, 1, VGPU_PTE_WRITE), -EINVAL); /* page 0 is reserved */
    CHECK_EQ(vgpu_mmu_map(&mmu, 1, 10, 4, VGPU_PTE_WRITE), 0);
    CHECK_EQ(vgpu_mmu_map(&mmu, 3, 20, 1, VGPU_PTE_WRITE), -EEXIST);
    CHECK_EQ(vgpu_mmu_map(&mmu, 8, 30, 1, 0), 0);                   /* read-only page */

    CHECK_EQ(vgpu_mmu_translate(&mmu, 2 * VGPU_PAGE_SIZE + 123, 0, &pa), 0);
    CHECK_EQ(pa, 11 * VGPU_PAGE_SIZE + 123);                         /* vpn 2 -> pfn 11 */
    CHECK_EQ(vgpu_mmu_translate(&mmu, 0, 0, &pa), -EFAULT);         /* NULL faults */
    CHECK_EQ(vgpu_mmu_translate(&mmu, 8 * VGPU_PAGE_SIZE, 1, &pa), -EFAULT); /* write to RO */
    CHECK_EQ(mmu.faults, 2);

    vgpu_mmu_unmap(&mmu, 1, 4);
    CHECK_EQ(vgpu_mmu_translate(&mmu, 2 * VGPU_PAGE_SIZE, 0, &pa), -EFAULT); /* TLB flushed */
    vgpu_mmu_destroy(&mmu);
}

static uint64_t touch(struct vgpu_mmu *mmu, uint64_t vpn)
{
    uint64_t pa, before = mmu->tlb.misses;
    vgpu_mmu_translate(mmu, vpn << VGPU_PAGE_SHIFT, 0, &pa);
    return mmu->tlb.misses - before; /* 1 = miss, 0 = hit */
}

static void test_tlb_lru(void)
{
    struct vgpu_mmu mmu;
    CHECK_EQ(vgpu_mmu_init(&mmu, 64), 0);
    CHECK_EQ(vgpu_mmu_map(&mmu, 1, 0, 32, VGPU_PTE_WRITE), 0);
    for (uint64_t v = 1; v <= VGPU_TLB_ENTRIES; v++)
        CHECK_EQ(touch(&mmu, v), 1);                /* cold misses fill the TLB */
    CHECK_EQ(touch(&mmu, 1), 0);                    /* hit: page 1 becomes most recent */
    CHECK_EQ(touch(&mmu, VGPU_TLB_ENTRIES + 1), 1); /* evicts the LRU entry, page 2 */
    CHECK_EQ(touch(&mmu, 1), 0);                    /* page 1 survived */
    CHECK_EQ(touch(&mmu, 2), 1);                    /* page 2 was evicted */
    vgpu_mmu_destroy(&mmu);
}

static void test_ops(void)
{
    uint8_t rgba[8] = {255, 255, 255, 255, 255, 0, 0, 255}, gray[2];
    vgpu_op_grayscale(gray, rgba, 2, 1);
    CHECK_EQ(gray[0], 255);
    CHECK_EQ(gray[1], 76); /* (77 * 255) >> 8 */

    uint8_t flat[9], out[9];
    memset(flat, 100, sizeof(flat));
    vgpu_op_blur3(out, flat, 3, 3);
    CHECK_EQ(out[4], 100); /* blurring a flat image changes nothing */
    vgpu_op_sobel(out, flat, 3, 3);
    CHECK_EQ(out[4], 0);   /* and it has no edges */

    uint8_t step[9] = {0, 0, 200, 0, 0, 200, 0, 0, 200};
    vgpu_op_sobel(out, step, 3, 3);
    CHECK_EQ(out[4], 255); /* a strong vertical edge saturates */
}

static void test_large_pages_in_mmu(void)
{
    struct vgpu_mmu mmu;
    uint64_t pa;
    CHECK_EQ(vgpu_mmu_init(&mmu, 128), 0);
    CHECK_EQ(vgpu_mmu_map(&mmu, 17, 32, 16, VGPU_PTE_WRITE | VGPU_PTE_LARGE), -EINVAL); /* vpn not aligned */
    CHECK_EQ(vgpu_mmu_map(&mmu, 16, 33, 16, VGPU_PTE_WRITE | VGPU_PTE_LARGE), -EINVAL); /* pfn not aligned */
    CHECK_EQ(vgpu_mmu_map(&mmu, 16, 32, 32, VGPU_PTE_WRITE | VGPU_PTE_LARGE), 0);       /* two large pages */

    for (uint64_t v = 16; v < 48; v++)
        touch(&mmu, v);
    CHECK_EQ(mmu.tlb.misses, 2); /* one miss per 64 KiB page, not per 4 KiB page */
    CHECK_EQ(mmu.tlb.hits, 30);

    CHECK_EQ(vgpu_mmu_translate(&mmu, 21 * VGPU_PAGE_SIZE + 7, 0, &pa), 0);
    CHECK_EQ(pa, 37 * VGPU_PAGE_SIZE + 7); /* vpn 21 = group start 16 + 5 -> pfn 32 + 5 */

    vgpu_mmu_unmap(&mmu, 16, 16);          /* invalidates the cached large entry */
    CHECK_EQ(vgpu_mmu_translate(&mmu, 20 * VGPU_PAGE_SIZE, 0, &pa), -EFAULT);
    CHECK_EQ(vgpu_mmu_translate(&mmu, 40 * VGPU_PAGE_SIZE, 0, &pa), 0); /* second group still mapped */
    vgpu_mmu_destroy(&mmu);
}

/* ---- the device ----------------------------------------------------------------------- */

static vgpu_device *make_device(enum vgpu_sched_policy policy, int paused, uint32_t ring)
{
    struct vgpu_config cfg;
    vgpu_config_default(&cfg);
    cfg.vram_bytes   = 16u << 20;
    cfg.policy       = policy;
    cfg.start_paused = paused;
    cfg.ring_entries = ring;
    vgpu_device *dev = NULL;
    CHECK_EQ(vgpu_device_create(&cfg, &dev), 0);
    return dev;
}

static void test_fill_then_copy(void)
{
    vgpu_device *dev = make_device(VGPU_SCHED_FCFS, 0, 64);
    uint32_t ctx;
    uint64_t a, b, fence;
    const uint64_t size = 3 * VGPU_PAGE_SIZE + 100; /* spans four pages */
    CHECK_EQ(vgpu_ctx_create(dev, 0, &ctx), 0);
    CHECK_EQ(vgpu_bo_create(dev, ctx, size, &a), 0);
    CHECK_EQ(vgpu_bo_create(dev, ctx, size, &b), 0);

    struct vgpu_cmd cmd = vgpu_cmd_fill(a, size, 0xDEADBEEF);
    CHECK_EQ(vgpu_submit(dev, ctx, &cmd, NULL, NULL, &fence), 0);
    CHECK_EQ(fence, 1);
    cmd = vgpu_cmd_copy(b, a, size);
    CHECK_EQ(vgpu_submit(dev, ctx, &cmd, NULL, NULL, &fence), 0);
    CHECK_EQ(fence, 2);
    CHECK_EQ(vgpu_fence_wait(dev, ctx, fence, 1000), 0);

    uint32_t *pb = vgpu_bo_map(dev, ctx, b);
    CHECK_EQ(pb[0], 0xDEADBEEF);
    CHECK_EQ(pb[size / 4 - 1], 0xDEADBEEF); /* copy ran after fill: in-order queue */
    CHECK_EQ(vgpu_fence_wait(dev, ctx, 99, 0), -EINVAL); /* never submitted */
    CHECK_EQ(vgpu_ctx_destroy(dev, ctx), 0);

    struct vgpu_stats s;
    vgpu_get_stats(dev, &s);
    CHECK_EQ(s.vram_used_bytes, 0); /* destroying the context freed its buffers */
    vgpu_device_destroy(dev);
}

static void test_guard_page_fault_is_sticky(void)
{
    vgpu_device *dev = make_device(VGPU_SCHED_FCFS, 0, 64);
    uint32_t ctx;
    uint64_t buf, f1, f2;
    CHECK_EQ(vgpu_ctx_create(dev, 0, &ctx), 0);
    CHECK_EQ(vgpu_bo_create(dev, ctx, 4096, &buf), 0);

    struct vgpu_cmd cmd = vgpu_cmd_fill(buf, 4096 + 4, 1); /* 4 bytes into the guard page */
    CHECK_EQ(vgpu_submit(dev, ctx, &cmd, NULL, NULL, &f1), 0);
    CHECK_EQ(vgpu_fence_wait(dev, ctx, f1, 1000), -EIO);

    cmd = vgpu_cmd_fill(buf, 4096, 1); /* valid, but the context is already broken */
    CHECK_EQ(vgpu_submit(dev, ctx, &cmd, NULL, NULL, &f2), 0);
    CHECK_EQ(vgpu_fence_wait(dev, ctx, f2, 1000), -EIO);
    vgpu_device_destroy(dev);
}

static void test_context_isolation(void)
{
    vgpu_device *dev = make_device(VGPU_SCHED_FCFS, 0, 64);
    uint32_t c1, c2;
    uint64_t buf, fence;
    CHECK_EQ(vgpu_ctx_create(dev, 0, &c1), 0);
    CHECK_EQ(vgpu_ctx_create(dev, 0, &c2), 0);
    CHECK_EQ(vgpu_bo_create(dev, c1, 4096, &buf), 0);

    /* c2 uses c1's address: it means nothing in c2's address space. */
    struct vgpu_cmd cmd = vgpu_cmd_fill(buf, 4096, 7);
    CHECK_EQ(vgpu_submit(dev, c2, &cmd, NULL, NULL, &fence), 0);
    CHECK_EQ(vgpu_fence_wait(dev, c2, fence, 1000), -EIO);
    CHECK_EQ(((uint32_t *)vgpu_bo_map(dev, c1, buf))[0], 0); /* c1's data untouched */
    vgpu_device_destroy(dev);
}

static void test_ring_full_timeout_and_busy(void)
{
    vgpu_device *dev = make_device(VGPU_SCHED_FCFS, 1 /* paused */, 4);
    uint32_t ctx;
    uint64_t buf, fence;
    CHECK_EQ(vgpu_ctx_create(dev, 0, &ctx), 0);
    CHECK_EQ(vgpu_bo_create(dev, ctx, 4096, &buf), 0);

    struct vgpu_cmd cmd = vgpu_cmd_fill(buf, 4096, 1);
    for (int i = 0; i < 4; i++)
        CHECK_EQ(vgpu_submit(dev, ctx, &cmd, NULL, NULL, &fence), 0);
    CHECK_EQ(vgpu_submit(dev, ctx, &cmd, NULL, NULL, NULL), -EAGAIN); /* ring full */
    CHECK_EQ(vgpu_fence_wait(dev, ctx, fence, 0), -ETIMEDOUT);        /* poll */
    CHECK_EQ(vgpu_fence_wait(dev, ctx, fence, 30), -ETIMEDOUT);       /* timed wait */
    CHECK_EQ(vgpu_bo_free(dev, ctx, buf), -EBUSY);                    /* jobs still pending */

    vgpu_device_start(dev);
    CHECK_EQ(vgpu_fence_wait(dev, ctx, fence, 1000), 0);
    CHECK_EQ(vgpu_bo_free(dev, ctx, buf), 0);
    CHECK_EQ(vgpu_bo_free(dev, ctx, buf), -ENOENT); /* double free is caught */
    vgpu_device_destroy(dev);
}

/* One 1 MiB job, with and without large pages: the TLB hit rate tells them apart. */
static double hit_rate_for_fill(int large_pages)
{
    struct vgpu_config cfg;
    vgpu_config_default(&cfg);
    cfg.vram_bytes  = 16u << 20;
    cfg.large_pages = large_pages;
    vgpu_device *dev;
    uint32_t ctx;
    uint64_t buf, fence;
    struct vgpu_stats s;
    CHECK_EQ(vgpu_device_create(&cfg, &dev), 0);
    CHECK_EQ(vgpu_ctx_create(dev, 0, &ctx), 0);
    CHECK_EQ(vgpu_bo_create(dev, ctx, 1u << 20, &buf), 0);
    CHECK_EQ(buf % (large_pages ? 65536 : 4096), 0); /* large buffers start 64 KiB-aligned */
    struct vgpu_cmd cmd = vgpu_cmd_fill(buf, 1u << 20, 5);
    CHECK_EQ(vgpu_submit(dev, ctx, &cmd, NULL, NULL, &fence), 0);
    CHECK_EQ(vgpu_fence_wait(dev, ctx, fence, 1000), 0);
    CHECK_EQ(((uint32_t *)vgpu_bo_map(dev, ctx, buf))[(1u << 18) - 1], 5);
    vgpu_get_stats(dev, &s);
    vgpu_device_destroy(dev);
    return (double)s.tlb_hits / (double)(s.tlb_hits + s.tlb_misses);
}

static void test_large_pages_raise_tlb_hit_rate(void)
{
    CHECK(hit_rate_for_fill(0) < 0.01); /* 256 small pages, each seen once: all misses */
    CHECK(hit_rate_for_fill(1) > 0.9);  /* 16 large pages: 15 of every 16 lookups hit */
}

/* ---- scheduling ------------------------------------------------------------------------- */

struct order_log {
    uint32_t ctx[16];
    int      len;
};

static void record_order(uint32_t ctx_id, uint64_t fence, void *user)
{
    (void)fence;
    struct order_log *log = user;
    log->ctx[log->len++] = ctx_id; /* callbacks run one at a time on the engine thread */
}

/*
 * Three contexts, all jobs queued before the engine starts:
 *   ctx 0: priority 1, big jobs   ctx 1: priority 9, small jobs   ctx 2: priority 5, medium
 * Submission order: 0, 1, 2, 0, 1, 2.
 */
static int order_is(enum vgpu_sched_policy policy, const uint32_t expected[6])
{
    static const int      prio[3]  = {1, 9, 5};
    static const uint64_t bytes[3] = {64u << 10, 4u << 10, 16u << 10};
    struct order_log log = {.len = 0};
    vgpu_device *dev = make_device(policy, 1, 64);
    uint32_t ctx[3];
    uint64_t buf[3];
    for (int i = 0; i < 3; i++) {
        vgpu_ctx_create(dev, prio[i], &ctx[i]);
        vgpu_bo_create(dev, ctx[i], bytes[i], &buf[i]);
    }
    for (int round = 0; round < 2; round++)
        for (int i = 0; i < 3; i++) {
            struct vgpu_cmd cmd = vgpu_cmd_fill(buf[i], bytes[i], 0);
            vgpu_submit(dev, ctx[i], &cmd, record_order, &log, NULL);
        }
    vgpu_device_start(dev);
    for (int i = 0; i < 3; i++)
        vgpu_ctx_finish(dev, ctx[i], 1000);
    vgpu_device_destroy(dev);

    if (log.len != 6)
        return 0;
    for (int i = 0; i < 6; i++)
        if (log.ctx[i] != expected[i])
            return 0;
    return 1;
}

static void test_sched_policies(void)
{
    CHECK(order_is(VGPU_SCHED_FCFS, (const uint32_t[]){0, 1, 2, 0, 1, 2}));
    CHECK(order_is(VGPU_SCHED_RR, (const uint32_t[]){0, 1, 2, 0, 1, 2}));
    CHECK(order_is(VGPU_SCHED_PRIORITY, (const uint32_t[]){1, 1, 2, 2, 0, 0}));
    CHECK(order_is(VGPU_SCHED_SJF, (const uint32_t[]){1, 1, 2, 2, 0, 0}));
}

/*
 * Starvation: ctx 1 (priority 3) has 8 jobs, ctx 0 (priority 0) has 1, queued after them.
 * PRIORITY runs ctx 0 last. AGING adds 1 per pick ctx 0 waits: after 3 picks it ties
 * (3 vs 3, the older job wins), after 4 it is ahead, so it runs 5th.
 */
static int starvation_order_is(enum vgpu_sched_policy policy, const uint32_t expected[9])
{
    struct order_log log = {.len = 0};
    vgpu_device *dev = make_device(policy, 1, 64);
    uint32_t low, high;
    uint64_t low_buf, high_buf;
    vgpu_ctx_create(dev, 0, &low);  /* ctx 0 */
    vgpu_ctx_create(dev, 3, &high); /* ctx 1 */
    vgpu_bo_create(dev, low, 4096, &low_buf);
    vgpu_bo_create(dev, high, 4096, &high_buf);
    for (int i = 0; i < 8; i++) {
        struct vgpu_cmd cmd = vgpu_cmd_fill(high_buf, 4096, 0);
        vgpu_submit(dev, high, &cmd, record_order, &log, NULL);
    }
    struct vgpu_cmd cmd = vgpu_cmd_fill(low_buf, 4096, 0);
    vgpu_submit(dev, low, &cmd, record_order, &log, NULL);
    vgpu_device_start(dev);
    vgpu_ctx_finish(dev, low, 1000);
    vgpu_ctx_finish(dev, high, 1000);
    vgpu_device_destroy(dev);

    if (log.len != 9)
        return 0;
    for (int i = 0; i < 9; i++)
        if (log.ctx[i] != expected[i])
            return 0;
    return 1;
}

static void test_aging_prevents_starvation(void)
{
    CHECK(starvation_order_is(VGPU_SCHED_PRIORITY, (const uint32_t[]){1, 1, 1, 1, 1, 1, 1, 1, 0}));
    CHECK(starvation_order_is(VGPU_SCHED_AGING, (const uint32_t[]){1, 1, 1, 1, 0, 1, 1, 1, 1}));
}

/* A callback may submit more work: proves callbacks run without the device lock held. */
struct chain {
    vgpu_device    *dev;
    uint32_t        ctx;
    uint64_t        buf;
    int             done;
    pthread_mutex_t m;
    pthread_cond_t  cv;
};

static void resubmit(uint32_t ctx_id, uint64_t fence, void *user)
{
    (void)ctx_id;
    (void)fence;
    struct chain *c = user;
    pthread_mutex_lock(&c->m);
    int done = ++c->done;
    pthread_cond_signal(&c->cv);
    pthread_mutex_unlock(&c->m);
    if (done < 10) {
        struct vgpu_cmd cmd = vgpu_cmd_fill(c->buf, 4096, (uint32_t)done);
        vgpu_submit(c->dev, c->ctx, &cmd, resubmit, c, NULL); /* would deadlock if the lock were held */
    }
}

static void test_callback_can_submit(void)
{
    vgpu_device *dev = make_device(VGPU_SCHED_FCFS, 0, 64);
    struct chain c = {.dev = dev, .done = 0};
    pthread_mutex_init(&c.m, NULL);
    pthread_cond_init(&c.cv, NULL);
    CHECK_EQ(vgpu_ctx_create(dev, 0, &c.ctx), 0);
    CHECK_EQ(vgpu_bo_create(dev, c.ctx, 4096, &c.buf), 0);

    struct vgpu_cmd cmd = vgpu_cmd_fill(c.buf, 4096, 0);
    CHECK_EQ(vgpu_submit(dev, c.ctx, &cmd, resubmit, &c, NULL), 0);
    pthread_mutex_lock(&c.m);
    while (c.done < 10)
        pthread_cond_wait(&c.cv, &c.m);
    pthread_mutex_unlock(&c.m);
    CHECK_EQ(c.done, 10);

    vgpu_device_destroy(dev);
    pthread_mutex_destroy(&c.m);
    pthread_cond_destroy(&c.cv);
}

/* ---- concurrency ------------------------------------------------------------------------- */

struct stress_arg {
    vgpu_device *dev;
    int          ok;
};

static void *stress_thread(void *p)
{
    struct stress_arg *arg = p;
    uint32_t ctx;
    uint64_t a, b;
    const uint64_t size = 16 * 1024;
    if (vgpu_ctx_create(arg->dev, 0, &ctx) || vgpu_bo_create(arg->dev, ctx, size, &a) ||
        vgpu_bo_create(arg->dev, ctx, size, &b))
        return NULL;
    uint32_t *pb = vgpu_bo_map(arg->dev, ctx, b);

    for (uint32_t iter = 1; iter <= 200; iter++) {
        struct vgpu_cmd cmds[2] = {vgpu_cmd_fill(a, size, iter), vgpu_cmd_copy(b, a, size)};
        uint64_t fence = 0;
        for (int i = 0; i < 2; i++) {
            int err;
            while ((err = vgpu_submit(arg->dev, ctx, &cmds[i], NULL, NULL, &fence)) == -EAGAIN)
                ; /* ring full: other threads are submitting too */
            if (err)
                return NULL;
        }
        if (vgpu_fence_wait(arg->dev, ctx, fence, 5000) != 0 || pb[size / 4 - 1] != iter)
            return NULL;
    }
    arg->ok = vgpu_ctx_destroy(arg->dev, ctx) == 0;
    return NULL;
}

static void test_multithreaded_stress(void)
{
    vgpu_device *dev = make_device(VGPU_SCHED_RR, 0, 16);
    pthread_t threads[6];
    struct stress_arg args[6];
    for (int i = 0; i < 6; i++) {
        args[i] = (struct stress_arg){.dev = dev, .ok = 0};
        pthread_create(&threads[i], NULL, stress_thread, &args[i]);
    }
    for (int i = 0; i < 6; i++) {
        pthread_join(threads[i], NULL);
        CHECK(args[i].ok);
    }
    struct vgpu_stats s;
    vgpu_get_stats(dev, &s);
    CHECK_EQ(s.jobs_completed, 6 * 200 * 2);
    CHECK_EQ(s.vram_used_bytes, 0); /* no leaked buffers */
    vgpu_device_destroy(dev);
}

static void test_bad_arguments(void)
{
    struct vgpu_config cfg;
    vgpu_device *dev;
    vgpu_config_default(&cfg);
    cfg.ring_entries = 100; /* not a power of two */
    CHECK_EQ(vgpu_device_create(&cfg, &dev), -EINVAL);

    dev = make_device(VGPU_SCHED_FCFS, 0, 64);
    uint32_t ctx;
    uint64_t va;
    CHECK_EQ(vgpu_ctx_create(dev, 0, &ctx), 0);
    CHECK_EQ(vgpu_bo_create(dev, ctx, 0, &va), -EINVAL);
    CHECK_EQ(vgpu_bo_create(dev, ctx, 1ull << 40, &va), -EINVAL);
    CHECK_EQ(vgpu_bo_create(dev, VGPU_MAX_CTX - 1, 4096, &va), -EINVAL); /* no such context */

    struct vgpu_cmd cmd = vgpu_cmd_fill(0, 4, 0);
    cmd.hdr.op = 200;
    CHECK_EQ(vgpu_submit(dev, ctx, &cmd, NULL, NULL, NULL), -EINVAL);

    for (int i = 1; i < VGPU_MAX_CTX; i++)
        CHECK_EQ(vgpu_ctx_create(dev, 0, &ctx), 0);
    CHECK_EQ(vgpu_ctx_create(dev, 0, &ctx), -EBUSY); /* every slot taken */
    vgpu_device_destroy(dev);
}

int main(void)
{
    printf("vGPU driver tests\n");
    RUN(test_uapi_layout);
    RUN(test_aligned_malloc);
    RUN(test_page_alloc);
    RUN(test_mmu_translate);
    RUN(test_tlb_lru);
    RUN(test_large_pages_in_mmu);
    RUN(test_ops);
    RUN(test_fill_then_copy);
    RUN(test_guard_page_fault_is_sticky);
    RUN(test_context_isolation);
    RUN(test_ring_full_timeout_and_busy);
    RUN(test_large_pages_raise_tlb_hit_rate);
    RUN(test_sched_policies);
    RUN(test_aging_prevents_starvation);
    RUN(test_callback_can_submit);
    RUN(test_multithreaded_stress);
    RUN(test_bad_arguments);
    return TEST_SUMMARY();
}
