/*
 * device.c - the vGPU device: contexts, buffers, command ring, engine thread.
 *
 * Life of one job:
 *
 *   vgpu_submit()     user thread: copy the command into the ring, advance
 *        |            ring_tail, ring the doorbell
 *        v
 *   engine_thread()   "the hardware": move commands from the ring into the
 *        |            scheduler, pick the next job, translate its GPU addresses
 *        |            through the context's MMU, run it with the lock dropped
 *        v
 *   retire            update the fence and wake waiters (the "interrupt"),
 *                     then call the completion callback
 *
 * Locking: one mutex protects all device state. Kernels run with it released,
 * so user threads can keep submitting while the engine works. One big lock is
 * a deliberate first step: easy to prove correct; split it only after
 * measuring contention.
 */
#include "vgpu/vgpu.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "aligned_alloc.h"
#include "mmu.h"
#include "ops.h"
#include "page_alloc.h"
#include "sched.h"

#define VGPU_VA_PAGES (1u << 16) /* 256 MiB of GPU virtual address space per context */

/* Driver bookkeeping for one ring slot (not part of the hardware command). */
struct ring_slot {
    vgpu_completion_fn on_done;
    void              *user;
    uint64_t           submit_ns;
};

struct bo {
    uint64_t   va;
    uint64_t   size;
    uint32_t   pfn;
    uint32_t   npages;     /* pages mapped                                */
    uint32_t   phys_pages; /* pages allocated: npages plus any colour pad */
    struct bo *next;
};

struct ctx {
    int             in_use;
    int             priority;
    struct vgpu_mmu mmu;
    uint64_t        next_va;
    struct bo      *bos;
    uint64_t        submitted_seq;
    uint64_t        completed_seq;
    uint64_t        first_fault_fence; /* 0 = no fault so far */
};

struct vgpu_device {
    struct vgpu_config     cfg;
    uint8_t               *vram;
    uint64_t               vram_bytes;
    struct vgpu_page_alloc pages;
    struct ctx             ctx[VGPU_MAX_CTX];

    /* Command ring. head and tail are free-running counters: the slot is
     * counter & mask, and tail - head is the fill level even after wrap-around. */
    struct vgpu_cmd  *ring;
    struct ring_slot *slots;
    uint32_t          ring_mask;
    uint32_t          ring_head; /* advanced by the engine (consumer) */
    uint32_t          ring_tail; /* advanced by the driver (producer) */

    struct vgpu_sched sched;
    uint64_t          jobs_completed;

    pthread_mutex_t lock;
    pthread_cond_t  doorbell_cv; /* the engine sleeps here when idle        */
    pthread_cond_t  fence_cv;    /* waiters sleep here: the "interrupt" line */
    pthread_t       engine;
    int             paused;
    int             shutdown;
};

/* ---- helpers ------------------------------------------------------------------ */

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* pthread_cond_timedwait takes an absolute CLOCK_REALTIME deadline. */
static struct timespec deadline_after_ms(uint32_t ms)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t nsec = (uint64_t)tv.tv_usec * 1000ull + (uint64_t)(ms % 1000) * 1000000ull;
    struct timespec ts;
    ts.tv_sec  = tv.tv_sec + (time_t)(ms / 1000) + (time_t)(nsec / 1000000000ull);
    ts.tv_nsec = (long)(nsec % 1000000000ull);
    return ts;
}

static struct ctx *get_ctx(vgpu_device *dev, uint32_t ctx_id)
{
    if (ctx_id >= VGPU_MAX_CTX || !dev->ctx[ctx_id].in_use)
        return NULL;
    return &dev->ctx[ctx_id];
}

const char *vgpu_policy_name(enum vgpu_sched_policy p)
{
    switch (p) {
    case VGPU_SCHED_FCFS:     return "fcfs";
    case VGPU_SCHED_RR:       return "rr";
    case VGPU_SCHED_PRIORITY: return "priority";
    case VGPU_SCHED_SJF:      return "sjf";
    case VGPU_SCHED_AGING:    return "aging";
    }
    return "unknown";
}

const char *vgpu_op_name(uint16_t op)
{
    static const char *names[VGPU_OP_COUNT] = {"nop", "fill", "copy", "grayscale", "blur3", "sobel"};
    return op < VGPU_OP_COUNT ? names[op] : "invalid";
}

/* ---- command builders ------------------------------------------------------------ */

static struct vgpu_cmd cmd_with_op(uint16_t op)
{
    struct vgpu_cmd c;
    memset(&c, 0, sizeof(c));
    c.hdr.op = op;
    return c;
}

struct vgpu_cmd vgpu_cmd_fill(uint64_t dst_va, uint64_t size, uint32_t value)
{
    struct vgpu_cmd c = cmd_with_op(VGPU_OP_FILL);
    c.u.fill.dst_va = dst_va;
    c.u.fill.size   = size;
    c.u.fill.value  = value;
    return c;
}

struct vgpu_cmd vgpu_cmd_copy(uint64_t dst_va, uint64_t src_va, uint64_t size)
{
    struct vgpu_cmd c = cmd_with_op(VGPU_OP_COPY);
    c.u.copy.dst_va = dst_va;
    c.u.copy.src_va = src_va;
    c.u.copy.size   = size;
    return c;
}

struct vgpu_cmd vgpu_cmd_image(uint16_t op, uint64_t dst_va, uint64_t src_va,
                               uint32_t width, uint32_t height)
{
    struct vgpu_cmd c = cmd_with_op(op);
    c.u.image.dst_va = dst_va;
    c.u.image.src_va = src_va;
    c.u.image.width  = width;
    c.u.image.height = height;
    return c;
}

/* ---- the engine ("the hardware") ----------------------------------------------------- */

/* Host pointers for one job, looked up through the MMU while holding the lock. */
struct resolved {
    uint16_t    op;
    uint32_t    value;
    void       *dst;
    const void *src;
    uint64_t    n;
    uint32_t    width, height;
};

static int map_range(vgpu_device *dev, struct ctx *c, uint64_t va, uint64_t size, int write, void **out)
{
    uint64_t pa;
    int err = vgpu_mmu_translate_range(&c->mmu, va, size, write, &pa);
    if (err)
        return err;
    if (pa + size > dev->vram_bytes)
        return -EFAULT;
    *out = dev->vram + pa;
    return 0;
}

static int resolve_job(vgpu_device *dev, const struct vgpu_job *job, struct resolved *r)
{
    struct ctx *c = &dev->ctx[job->ctx_id];
    const struct vgpu_cmd *cmd = &job->cmd;
    void *src = NULL;
    int err;

    memset(r, 0, sizeof(*r));
    r->op = cmd->hdr.op;

    switch (cmd->hdr.op) {
    case VGPU_OP_NOP:
        return 0;
    case VGPU_OP_FILL:
        if (cmd->u.fill.size % 4 != 0)
            return -EINVAL;
        r->n     = cmd->u.fill.size / 4;
        r->value = cmd->u.fill.value;
        return map_range(dev, c, cmd->u.fill.dst_va, cmd->u.fill.size, 1, &r->dst);
    case VGPU_OP_COPY:
        r->n = cmd->u.copy.size;
        err = map_range(dev, c, cmd->u.copy.src_va, r->n, 0, &src);
        if (!err)
            err = map_range(dev, c, cmd->u.copy.dst_va, r->n, 1, &r->dst);
        r->src = src;
        return err;
    case VGPU_OP_GRAYSCALE:
    case VGPU_OP_BLUR3:
    case VGPU_OP_SOBEL: {
        uint64_t pixels = (uint64_t)cmd->u.image.width * cmd->u.image.height;
        if (pixels == 0)
            return -EINVAL;
        uint64_t src_bytes = cmd->hdr.op == VGPU_OP_GRAYSCALE ? pixels * 4 : pixels;
        r->width  = cmd->u.image.width;
        r->height = cmd->u.image.height;
        err = map_range(dev, c, cmd->u.image.src_va, src_bytes, 0, &src);
        if (!err)
            err = map_range(dev, c, cmd->u.image.dst_va, pixels, 1, &r->dst);
        r->src = src;
        return err;
    }
    default:
        return -EINVAL;
    }
}

static void run_resolved(const struct resolved *r)
{
    switch (r->op) {
    case VGPU_OP_FILL:      vgpu_op_fill(r->dst, (size_t)r->n, r->value); break;
    case VGPU_OP_COPY:      memmove(r->dst, r->src, (size_t)r->n); break; /* the "DMA" engine */
    case VGPU_OP_GRAYSCALE: vgpu_op_grayscale(r->dst, r->src, r->width, r->height); break;
    case VGPU_OP_BLUR3:     vgpu_op_blur3(r->dst, r->src, r->width, r->height); break;
    case VGPU_OP_SOBEL:     vgpu_op_sobel(r->dst, r->src, r->width, r->height); break;
    default: break;
    }
}

static uint64_t job_size(const struct vgpu_cmd *cmd)
{
    switch (cmd->hdr.op) {
    case VGPU_OP_FILL: return cmd->u.fill.size;
    case VGPU_OP_COPY: return cmd->u.copy.size;
    case VGPU_OP_GRAYSCALE:
    case VGPU_OP_BLUR3:
    case VGPU_OP_SOBEL: return (uint64_t)cmd->u.image.width * cmd->u.image.height;
    default:            return 0;
    }
}

/* Move every command the driver produced from the ring into the scheduler. */
static void fetch_from_ring(vgpu_device *dev)
{
    while (dev->ring_head != dev->ring_tail) {
        uint32_t slot = dev->ring_head & dev->ring_mask;
        struct vgpu_job *job = calloc(1, sizeof(*job));
        if (!job)
            break; /* leave the command in the ring and retry on the next pass */
        job->cmd       = dev->ring[slot];
        job->ctx_id    = job->cmd.hdr.ctx_id;
        job->cost      = vgpu_op_estimate_cost(job->cmd.hdr.op, job_size(&job->cmd));
        job->on_done   = dev->slots[slot].on_done;
        job->user      = dev->slots[slot].user;
        job->submit_ns = dev->slots[slot].submit_ns;
        vgpu_sched_enqueue(&dev->sched, job);
        dev->ring_head++;
    }
}

static void *engine_thread(void *arg)
{
    vgpu_device *dev = arg;

    pthread_mutex_lock(&dev->lock);
    while (!dev->shutdown) {
        if (dev->paused) {
            pthread_cond_wait(&dev->doorbell_cv, &dev->lock);
            continue;
        }
        fetch_from_ring(dev);
        struct vgpu_job *job = vgpu_sched_pick(&dev->sched);
        if (!job) {
            pthread_cond_wait(&dev->doorbell_cv, &dev->lock); /* idle until the next doorbell */
            continue;
        }

        struct resolved r;
        int err = resolve_job(dev, job, &r);

        pthread_mutex_unlock(&dev->lock); /* run the kernel without blocking submitters */
        job->start_ns = now_ns();
        if (!err)
            run_resolved(&r);
        job->end_ns = now_ns();
        pthread_mutex_lock(&dev->lock);

        /* Retire the job: real hardware would raise an interrupt here. */
        struct ctx *c = &dev->ctx[job->ctx_id];
        c->completed_seq = job->cmd.hdr.fence_seq;
        if (err && c->first_fault_fence == 0)
            c->first_fault_fence = job->cmd.hdr.fence_seq;
        dev->jobs_completed++;
        vgpu_sched_record(&dev->sched, job);
        pthread_cond_broadcast(&dev->fence_cv);

        /* Call back with the lock dropped: a callback that calls vgpu_submit()
         * would otherwise deadlock on dev->lock. */
        if (job->on_done) {
            pthread_mutex_unlock(&dev->lock);
            job->on_done(job->ctx_id, job->cmd.hdr.fence_seq, job->user);
            pthread_mutex_lock(&dev->lock);
        }
        free(job);
    }
    pthread_mutex_unlock(&dev->lock);
    return NULL;
}

/* ---- device lifetime ---------------------------------------------------------------------- */

void vgpu_config_default(struct vgpu_config *cfg)
{
    cfg->vram_bytes   = 64ull << 20; /* 64 MiB */
    cfg->ring_entries = 256;
    cfg->policy       = VGPU_SCHED_FCFS;
    cfg->start_paused = 0;
    cfg->large_pages  = 1;
}

static void free_device_memory(vgpu_device *dev)
{
    vgpu_pa_destroy(&dev->pages);
    vgpu_aligned_free(dev->vram);
    free(dev->ring);
    free(dev->slots);
    free(dev);
}

int vgpu_device_create(const struct vgpu_config *cfg_in, vgpu_device **out)
{
    struct vgpu_config cfg;
    if (cfg_in)
        cfg = *cfg_in;
    else
        vgpu_config_default(&cfg);

    if (cfg.ring_entries == 0 || (cfg.ring_entries & (cfg.ring_entries - 1)) != 0)
        return -EINVAL;
    if (cfg.vram_bytes < VGPU_PAGE_SIZE || cfg.vram_bytes > (1ull << 36))
        return -EINVAL;
    if ((unsigned)cfg.policy >= VGPU_SCHED_POLICY_COUNT)
        return -EINVAL;

    vgpu_device *dev = calloc(1, sizeof(*dev));
    if (!dev)
        return -ENOMEM;
    dev->cfg        = cfg;
    dev->vram_bytes = (cfg.vram_bytes + VGPU_PAGE_MASK) & ~(uint64_t)VGPU_PAGE_MASK;
    dev->vram       = vgpu_aligned_malloc((size_t)dev->vram_bytes, VGPU_PAGE_SIZE);
    dev->ring       = calloc(cfg.ring_entries, sizeof(*dev->ring));
    dev->slots      = calloc(cfg.ring_entries, sizeof(*dev->slots));
    dev->ring_mask  = cfg.ring_entries - 1;
    if (!dev->vram || !dev->ring || !dev->slots ||
        vgpu_pa_init(&dev->pages, (uint32_t)(dev->vram_bytes >> VGPU_PAGE_SHIFT)) != 0) {
        free_device_memory(dev);
        return -ENOMEM;
    }

    vgpu_sched_init(&dev->sched, cfg.policy);
    pthread_mutex_init(&dev->lock, NULL);
    pthread_cond_init(&dev->doorbell_cv, NULL);
    pthread_cond_init(&dev->fence_cv, NULL);
    dev->paused = cfg.start_paused;

    if (pthread_create(&dev->engine, NULL, engine_thread, dev) != 0) {
        pthread_mutex_destroy(&dev->lock);
        pthread_cond_destroy(&dev->doorbell_cv);
        pthread_cond_destroy(&dev->fence_cv);
        free_device_memory(dev);
        return -EAGAIN;
    }
    *out = dev;
    return 0;
}

static void free_ctx_resources(vgpu_device *dev, struct ctx *c)
{
    struct bo *b = c->bos;
    while (b) {
        struct bo *next = b->next;
        vgpu_pa_free(&dev->pages, b->pfn, b->phys_pages);
        free(b);
        b = next;
    }
    c->bos = NULL;
    vgpu_mmu_destroy(&c->mmu);
    c->in_use = 0;
}

void vgpu_device_destroy(vgpu_device *dev)
{
    if (!dev)
        return;
    pthread_mutex_lock(&dev->lock);
    dev->shutdown = 1;
    pthread_cond_broadcast(&dev->doorbell_cv);
    pthread_mutex_unlock(&dev->lock);
    pthread_join(dev->engine, NULL);

    /* Drop work that never ran: first out of the ring, then out of the queues. */
    fetch_from_ring(dev);
    struct vgpu_job *job;
    while ((job = vgpu_sched_pick(&dev->sched)) != NULL)
        free(job);
    for (uint32_t i = 0; i < VGPU_MAX_CTX; i++)
        if (dev->ctx[i].in_use)
            free_ctx_resources(dev, &dev->ctx[i]);

    pthread_mutex_destroy(&dev->lock);
    pthread_cond_destroy(&dev->doorbell_cv);
    pthread_cond_destroy(&dev->fence_cv);
    free_device_memory(dev);
}

void vgpu_device_start(vgpu_device *dev)
{
    pthread_mutex_lock(&dev->lock);
    dev->paused = 0;
    pthread_cond_broadcast(&dev->doorbell_cv);
    pthread_mutex_unlock(&dev->lock);
}

/* ---- contexts --------------------------------------------------------------------------- */

int vgpu_ctx_create(vgpu_device *dev, int priority, uint32_t *ctx_id)
{
    pthread_mutex_lock(&dev->lock);
    int err = -EBUSY; /* every slot is taken */
    for (uint32_t i = 0; i < VGPU_MAX_CTX; i++) {
        struct ctx *c = &dev->ctx[i];
        if (c->in_use)
            continue;
        memset(c, 0, sizeof(*c));
        err = vgpu_mmu_init(&c->mmu, VGPU_VA_PAGES);
        if (err)
            break;
        c->in_use   = 1;
        c->priority = priority;
        c->next_va  = VGPU_VA_BASE;
        dev->sched.q[i].priority = priority;
        *ctx_id = i;
        break;
    }
    pthread_mutex_unlock(&dev->lock);
    return err;
}

int vgpu_ctx_destroy(vgpu_device *dev, uint32_t ctx_id)
{
    int err = vgpu_ctx_finish(dev, ctx_id, 5000);
    if (err && err != -EIO) /* a faulted context can still be torn down */
        return err;

    pthread_mutex_lock(&dev->lock);
    struct ctx *c = get_ctx(dev, ctx_id);
    if (!c)
        err = -EINVAL;
    else if (c->completed_seq != c->submitted_seq) /* someone submitted after finish */
        err = -EBUSY;
    else {
        free_ctx_resources(dev, c);
        err = 0;
    }
    pthread_mutex_unlock(&dev->lock);
    return err;
}

/* ---- buffers -------------------------------------------------------------------------------- */

int vgpu_bo_create(vgpu_device *dev, uint32_t ctx_id, uint64_t size, uint64_t *gpu_va)
{
    if (size == 0 || size > dev->vram_bytes)
        return -EINVAL;
    uint32_t npages = (uint32_t)((size + VGPU_PAGE_MASK) >> VGPU_PAGE_SHIFT);

    struct bo *b = calloc(1, sizeof(*b));
    if (!b)
        return -ENOMEM;

    /* Big buffers get large pages: align both the virtual and the physical start
     * to 64 KiB, and map every whole 64 KiB group as one large page. */
    int      large = dev->cfg.large_pages && npages >= VGPU_LARGE_PAGES;
    uint32_t align = large ? VGPU_LARGE_PAGES : 1;
    uint32_t big   = large ? npages / VGPU_LARGE_PAGES * VGPU_LARGE_PAGES : 0;
    /*
     * Colour pad: one extra unmapped large page after each large buffer. Without
     * it, buffers whose size rounds up to a power of two (a 1080p plane rounds up
     * to exactly 2 MiB) sit exactly a power of two apart in memory, and the CPU's
     * caches treat them as conflicting: Sobel then runs ~3x slower (finding A in
     * docs/DESIGN_DECISIONS.md, measured with `make layout`).
     */
    uint32_t phys  = large ? npages + VGPU_LARGE_PAGES : npages;

    pthread_mutex_lock(&dev->lock);
    struct ctx *c = get_ctx(dev, ctx_id);
    int err = c ? 0 : -EINVAL;
    uint64_t vpn = 0;
    if (c) {
        vpn = c->next_va >> VGPU_PAGE_SHIFT;
        vpn = (vpn + align - 1) / align * align;
    }
    if (!err && vpn + npages + 1 > VGPU_VA_PAGES)
        err = -ENOMEM; /* out of virtual address space */

    int64_t pfn = -1;
    if (!err) {
        pfn = vgpu_pa_alloc(&dev->pages, phys, align);
        if (pfn < 0)
            err = (int)pfn;
    }
    if (!err && big)
        err = vgpu_mmu_map(&c->mmu, vpn, (uint32_t)pfn, big, VGPU_PTE_WRITE | VGPU_PTE_LARGE);
    if (!err && npages > big) {
        err = vgpu_mmu_map(&c->mmu, vpn + big, (uint32_t)pfn + big, npages - big, VGPU_PTE_WRITE);
        if (err && big)
            vgpu_mmu_unmap(&c->mmu, vpn, big);
    }
    if (err && pfn >= 0)
        vgpu_pa_free(&dev->pages, (uint32_t)pfn, phys);
    if (err) {
        pthread_mutex_unlock(&dev->lock);
        free(b);
        return err;
    }

    b->va     = vpn << VGPU_PAGE_SHIFT;
    b->size   = size;
    b->pfn    = (uint32_t)pfn;
    b->npages     = npages;
    b->phys_pages = phys;
    b->next   = c->bos;
    c->bos    = b;
    /* Leave one unmapped guard page after every buffer, so overruns fault. */
    c->next_va = (vpn + npages + 1) << VGPU_PAGE_SHIFT;
    pthread_mutex_unlock(&dev->lock);

    /* Never hand out another context's old data. */
    memset(dev->vram + ((uint64_t)b->pfn << VGPU_PAGE_SHIFT), 0, (size_t)npages << VGPU_PAGE_SHIFT);
    *gpu_va = b->va;
    return 0;
}

int vgpu_bo_free(vgpu_device *dev, uint32_t ctx_id, uint64_t gpu_va)
{
    pthread_mutex_lock(&dev->lock);
    struct ctx *c = get_ctx(dev, ctx_id);
    int err = c ? 0 : -EINVAL;
    /* TODO(exercise): track which buffers each job uses instead of this coarse check. */
    if (!err && c->completed_seq != c->submitted_seq)
        err = -EBUSY; /* the engine may still be using the buffer */

    struct bo **link = c ? &c->bos : NULL;
    while (!err && *link && (*link)->va != gpu_va)
        link = &(*link)->next;
    if (!err && !*link)
        err = -ENOENT;

    if (!err) {
        struct bo *b = *link;
        *link = b->next;
        vgpu_mmu_unmap(&c->mmu, b->va >> VGPU_PAGE_SHIFT, b->npages);
        vgpu_pa_free(&dev->pages, b->pfn, b->phys_pages);
        free(b);
    }
    pthread_mutex_unlock(&dev->lock);
    return err;
}

void *vgpu_bo_map(vgpu_device *dev, uint32_t ctx_id, uint64_t gpu_va)
{
    void *p = NULL;
    pthread_mutex_lock(&dev->lock);
    struct ctx *c = get_ctx(dev, ctx_id);
    for (struct bo *b = c ? c->bos : NULL; b; b = b->next)
        if (gpu_va >= b->va && gpu_va < b->va + b->size) {
            p = dev->vram + ((uint64_t)b->pfn << VGPU_PAGE_SHIFT) + (gpu_va - b->va);
            break;
        }
    pthread_mutex_unlock(&dev->lock);
    return p;
}

/* ---- submission and fences ----------------------------------------------------------------------- */

int vgpu_submit(vgpu_device *dev, uint32_t ctx_id, const struct vgpu_cmd *cmd,
                vgpu_completion_fn on_done, void *user, uint64_t *fence)
{
    if (!cmd || cmd->hdr.op >= VGPU_OP_COUNT || cmd->hdr.flags != 0)
        return -EINVAL;

    pthread_mutex_lock(&dev->lock);
    struct ctx *c = get_ctx(dev, ctx_id);
    if (!c) {
        pthread_mutex_unlock(&dev->lock);
        return -EINVAL;
    }
    if (dev->ring_tail - dev->ring_head == dev->cfg.ring_entries) {
        pthread_mutex_unlock(&dev->lock);
        return -EAGAIN; /* ring full: the caller decides whether to retry */
    }

    uint32_t slot = dev->ring_tail & dev->ring_mask;
    uint64_t seq  = ++c->submitted_seq;
    dev->ring[slot]               = *cmd;
    dev->ring[slot].hdr.ctx_id    = ctx_id;
    dev->ring[slot].hdr.fence_seq = seq;
    dev->slots[slot].on_done      = on_done;
    dev->slots[slot].user         = user;
    dev->slots[slot].submit_ns    = now_ns();
    dev->ring_tail++;
    pthread_cond_signal(&dev->doorbell_cv); /* ring the doorbell */
    pthread_mutex_unlock(&dev->lock);

    if (fence)
        *fence = seq;
    return 0;
}

int vgpu_fence_wait(vgpu_device *dev, uint32_t ctx_id, uint64_t fence, uint32_t timeout_ms)
{
    pthread_mutex_lock(&dev->lock);
    struct ctx *c = get_ctx(dev, ctx_id);
    int err = 0;
    if (!c || fence > c->submitted_seq) {
        err = -EINVAL;
    } else {
        struct timespec deadline = deadline_after_ms(timeout_ms);
        while (c->completed_seq < fence) { /* a loop, because wakeups can be spurious */
            if (timeout_ms == 0) {
                err = -ETIMEDOUT;
                break;
            }
            if (timeout_ms == VGPU_WAIT_FOREVER) {
                pthread_cond_wait(&dev->fence_cv, &dev->lock);
            } else if (pthread_cond_timedwait(&dev->fence_cv, &dev->lock, &deadline) == ETIMEDOUT) {
                err = c->completed_seq >= fence ? 0 : -ETIMEDOUT;
                break;
            }
        }
        if (!err && c->first_fault_fence && c->first_fault_fence <= fence)
            err = -EIO;
    }
    pthread_mutex_unlock(&dev->lock);
    return err;
}

int vgpu_ctx_finish(vgpu_device *dev, uint32_t ctx_id, uint32_t timeout_ms)
{
    pthread_mutex_lock(&dev->lock);
    struct ctx *c = get_ctx(dev, ctx_id);
    uint64_t last = c ? c->submitted_seq : 0;
    pthread_mutex_unlock(&dev->lock);
    if (!c)
        return -EINVAL;
    return last ? vgpu_fence_wait(dev, ctx_id, last, timeout_ms) : 0;
}

/* ---- introspection ---------------------------------------------------------------------------------- */

void vgpu_get_stats(vgpu_device *dev, struct vgpu_stats *s)
{
    memset(s, 0, sizeof(*s));
    pthread_mutex_lock(&dev->lock);
    s->vram_total_bytes = dev->vram_bytes;
    s->vram_used_bytes  = (uint64_t)dev->pages.used << VGPU_PAGE_SHIFT;
    s->jobs_completed   = dev->jobs_completed;
    for (uint32_t i = 0; i < VGPU_MAX_CTX; i++) {
        if (!dev->ctx[i].in_use)
            continue;
        s->tlb_hits    += dev->ctx[i].mmu.tlb.hits;
        s->tlb_misses  += dev->ctx[i].mmu.tlb.misses;
        s->page_faults += dev->ctx[i].mmu.faults;
    }
    pthread_mutex_unlock(&dev->lock);
}

int vgpu_get_gantt(vgpu_device *dev, struct vgpu_gantt_rec *out, int max)
{
    pthread_mutex_lock(&dev->lock);
    int n = vgpu_sched_copy_gantt(&dev->sched, out, max);
    pthread_mutex_unlock(&dev->lock);
    return n;
}
