/*
 * vgpu.h - public C API of the vGPU driver.
 *
 * The "hardware" is a thread inside the library. All functions return 0 on
 * success or a negative errno value (-ENOMEM, -EINVAL, ...), the same
 * convention the Linux kernel uses.
 */
#ifndef VGPU_VGPU_H
#define VGPU_VGPU_H

#include <stdint.h>

#include "vgpu/uapi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vgpu_device vgpu_device;

#define VGPU_MAX_CTX      8
#define VGPU_WAIT_FOREVER 0xFFFFFFFFu /* timeout_ms value: never time out; 0 means poll */

enum vgpu_sched_policy {
    VGPU_SCHED_FCFS     = 0, /* oldest job first                                */
    VGPU_SCHED_RR       = 1, /* round-robin between contexts, one job per turn  */
    VGPU_SCHED_PRIORITY = 2, /* highest context priority first                  */
    VGPU_SCHED_SJF      = 3, /* shortest estimated job first (non-preemptive)   */
    VGPU_SCHED_AGING    = 4, /* priority + 1 for every pick a context waits     */
};
#define VGPU_SCHED_POLICY_COUNT 5

struct vgpu_config {
    uint64_t vram_bytes;           /* device memory, rounded up to pages        */
    uint32_t ring_entries;         /* command ring size, must be a power of 2  */
    enum vgpu_sched_policy policy;
    int start_paused;              /* engine waits for vgpu_device_start()     */
    int large_pages;               /* map buffers >= 64 KiB with 64 KiB pages  */
};

/* Called on the engine thread when a job finishes: a simulated interrupt handler. */
typedef void (*vgpu_completion_fn)(uint32_t ctx_id, uint64_t fence, void *user);

struct vgpu_stats {
    uint64_t vram_total_bytes;
    uint64_t vram_used_bytes;
    uint64_t jobs_completed;
    uint64_t tlb_hits;
    uint64_t tlb_misses;
    uint64_t page_faults;
};

/* One finished job, for Gantt charts. */
struct vgpu_gantt_rec {
    uint32_t ctx_id;
    uint16_t op;
    uint64_t fence;
    uint64_t submit_ns, start_ns, end_ns;
};

/* ---- device ---------------------------------------------------------------- */
void vgpu_config_default(struct vgpu_config *cfg);
int  vgpu_device_create(const struct vgpu_config *cfg, vgpu_device **out);
void vgpu_device_destroy(vgpu_device *dev);
void vgpu_device_start(vgpu_device *dev); /* releases a start_paused engine */

/* ---- contexts (one per process / stream) ----------------------------------- */
int vgpu_ctx_create(vgpu_device *dev, int priority, uint32_t *ctx_id);
int vgpu_ctx_destroy(vgpu_device *dev, uint32_t ctx_id); /* waits, then frees its buffers */

/* ---- buffers ----------------------------------------------------------------- */
int   vgpu_bo_create(vgpu_device *dev, uint32_t ctx_id, uint64_t size, uint64_t *gpu_va);
int   vgpu_bo_free(vgpu_device *dev, uint32_t ctx_id, uint64_t gpu_va);
void *vgpu_bo_map(vgpu_device *dev, uint32_t ctx_id, uint64_t gpu_va); /* CPU view, like mmap */

/* ---- work ---------------------------------------------------------------------
 * vgpu_submit returns -EAGAIN when the ring is full. A job that touches a bad
 * GPU address still signals its fence, but every later wait on that context
 * returns -EIO: the error is "sticky", like a CUDA context error.
 */
int vgpu_submit(vgpu_device *dev, uint32_t ctx_id, const struct vgpu_cmd *cmd,
                vgpu_completion_fn on_done, void *user, uint64_t *fence);
int vgpu_fence_wait(vgpu_device *dev, uint32_t ctx_id, uint64_t fence, uint32_t timeout_ms);
int vgpu_ctx_finish(vgpu_device *dev, uint32_t ctx_id, uint32_t timeout_ms); /* wait for all */

/* ---- introspection --------------------------------------------------------------- */
void vgpu_get_stats(vgpu_device *dev, struct vgpu_stats *out);
int  vgpu_get_gantt(vgpu_device *dev, struct vgpu_gantt_rec *out, int max); /* returns count */
const char *vgpu_policy_name(enum vgpu_sched_policy p);
const char *vgpu_op_name(uint16_t op);

/* ---- command builders --------------------------------------------------------------- */
struct vgpu_cmd vgpu_cmd_fill(uint64_t dst_va, uint64_t size, uint32_t value);
struct vgpu_cmd vgpu_cmd_copy(uint64_t dst_va, uint64_t src_va, uint64_t size);
struct vgpu_cmd vgpu_cmd_image(uint16_t op, uint64_t dst_va, uint64_t src_va,
                               uint32_t width, uint32_t height);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_VGPU_H */
