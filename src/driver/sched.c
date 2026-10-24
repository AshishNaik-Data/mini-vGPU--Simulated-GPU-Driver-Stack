#include "sched.h"

#include <string.h>

void vgpu_sched_init(struct vgpu_sched *s, enum vgpu_sched_policy policy)
{
    memset(s, 0, sizeof(*s));
    s->policy    = policy;
    s->rr_cursor = VGPU_MAX_CTX - 1; /* so the first RR pick starts at context 0 */
}

void vgpu_sched_enqueue(struct vgpu_sched *s, struct vgpu_job *job)
{
    struct vgpu_job_queue *q = &s->q[job->ctx_id];
    job->arrival = s->next_arrival++;
    job->next    = NULL;
    if (q->tail)
        q->tail->next = job;
    else
        q->head = job;
    q->tail = job;
    q->count++;
    s->total++;
}

/* Aging: every pick a waiting context is passed over raises its priority by 1. */
static int64_t effective_priority(const struct vgpu_job_queue *q)
{
    return (int64_t)q->priority + q->age;
}

/* Returns 1 if the head of queue a should run before the head of queue b. */
static int runs_before(const struct vgpu_sched *s, uint32_t a, uint32_t b)
{
    const struct vgpu_job *ja = s->q[a].head, *jb = s->q[b].head;
    switch (s->policy) {
    case VGPU_SCHED_PRIORITY:
        if (s->q[a].priority != s->q[b].priority)
            return s->q[a].priority > s->q[b].priority;
        break;
    case VGPU_SCHED_AGING: {
        int64_t pa = effective_priority(&s->q[a]), pb = effective_priority(&s->q[b]);
        if (pa != pb)
            return pa > pb;
        break;
    }
    case VGPU_SCHED_SJF:
        if (ja->cost != jb->cost)
            return ja->cost < jb->cost;
        break;
    default:
        break;
    }
    return ja->arrival < jb->arrival; /* FCFS, and the tie-break for the others */
}

/*
 * O(number of contexts) per pick, which is fine for 8 contexts.
 * TODO(exercise): add a preemptive SJF or MLFQ variant (needs jobs that can be split).
 */
struct vgpu_job *vgpu_sched_pick(struct vgpu_sched *s)
{
    if (s->total == 0)
        return NULL;

    int chosen = -1;
    if (s->policy == VGPU_SCHED_RR) {
        for (uint32_t step = 1; step <= VGPU_MAX_CTX; step++) {
            uint32_t c = (s->rr_cursor + step) % VGPU_MAX_CTX;
            if (s->q[c].head) {
                chosen = (int)c;
                break;
            }
        }
        s->rr_cursor = (uint32_t)chosen;
    } else {
        for (uint32_t c = 0; c < VGPU_MAX_CTX; c++) {
            if (!s->q[c].head)
                continue;
            if (chosen < 0 || runs_before(s, c, (uint32_t)chosen))
                chosen = (int)c;
        }
    }

    /* Everyone who had work and was passed over gets older; the winner starts again. */
    for (uint32_t c = 0; c < VGPU_MAX_CTX; c++)
        if (s->q[c].head && (int)c != chosen)
            s->q[c].age++;
    s->q[chosen].age = 0;

    struct vgpu_job_queue *q = &s->q[chosen];
    struct vgpu_job *job = q->head;
    q->head = job->next;
    if (!q->head)
        q->tail = NULL;
    q->count--;
    s->total--;
    job->next = NULL;
    return job;
}

void vgpu_sched_record(struct vgpu_sched *s, const struct vgpu_job *job)
{
    struct vgpu_gantt_rec *r = &s->log[s->log_next];
    r->ctx_id    = job->ctx_id;
    r->op        = job->cmd.hdr.op;
    r->fence     = job->cmd.hdr.fence_seq;
    r->submit_ns = job->submit_ns;
    r->start_ns  = job->start_ns;
    r->end_ns    = job->end_ns;
    s->log_next  = (s->log_next + 1) % VGPU_GANTT_CAPACITY;
    if (s->log_len < VGPU_GANTT_CAPACITY)
        s->log_len++;
}

int vgpu_sched_copy_gantt(const struct vgpu_sched *s, struct vgpu_gantt_rec *out, int max)
{
    uint32_t first = (s->log_next + VGPU_GANTT_CAPACITY - s->log_len) % VGPU_GANTT_CAPACITY;
    int n = 0;
    for (uint32_t i = 0; i < s->log_len && n < max; i++)
        out[n++] = s->log[(first + i) % VGPU_GANTT_CAPACITY]; /* oldest first */
    return n;
}
