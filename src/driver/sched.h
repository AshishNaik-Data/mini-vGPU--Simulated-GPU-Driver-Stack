/*
 * sched.h - picks which context's job the engine runs next.
 *
 * Every context has its own FIFO queue, so jobs of one context always run in
 * submission order (like a CUDA stream). The policy only decides *between*
 * contexts by looking at the job at the head of each queue.
 *
 * Scheduling is non-preemptive: once a job starts it runs to completion.
 * Not thread-safe by itself: the device holds its lock around every call.
 */
#ifndef VGPU_SCHED_H
#define VGPU_SCHED_H

#include <stdint.h>

#include "vgpu/vgpu.h"

struct vgpu_job {
    struct vgpu_cmd    cmd;
    uint32_t           ctx_id;
    uint64_t           cost;      /* estimated work, for SJF          */
    uint64_t           arrival;   /* global arrival order, for FCFS   */
    uint64_t           submit_ns;
    uint64_t           start_ns;
    uint64_t           end_ns;
    vgpu_completion_fn on_done;
    void              *user;
    struct vgpu_job   *next;
};

struct vgpu_job_queue {
    struct vgpu_job *head, *tail;
    uint32_t         count;
    int              priority; /* higher runs first under VGPU_SCHED_PRIORITY */
    uint32_t         age;      /* picks this queue has waited through (AGING) */
};

#define VGPU_GANTT_CAPACITY 1024u

struct vgpu_sched {
    enum vgpu_sched_policy policy;
    struct vgpu_job_queue  q[VGPU_MAX_CTX];
    uint32_t               rr_cursor;
    uint32_t               total;
    uint64_t               next_arrival;

    struct vgpu_gantt_rec  log[VGPU_GANTT_CAPACITY]; /* circular, keeps the latest records */
    uint32_t               log_next;
    uint32_t               log_len;
};

void vgpu_sched_init(struct vgpu_sched *s, enum vgpu_sched_policy policy);
void vgpu_sched_enqueue(struct vgpu_sched *s, struct vgpu_job *job);
struct vgpu_job *vgpu_sched_pick(struct vgpu_sched *s); /* NULL if nothing is queued */
void vgpu_sched_record(struct vgpu_sched *s, const struct vgpu_job *job);
int  vgpu_sched_copy_gantt(const struct vgpu_sched *s, struct vgpu_gantt_rec *out, int max);

#endif /* VGPU_SCHED_H */
