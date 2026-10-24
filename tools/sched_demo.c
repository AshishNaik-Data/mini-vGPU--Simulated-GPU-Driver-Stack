/*
 * sched_demo.c - run the same workload under every scheduling policy and
 * compare job order and waiting time, like a textbook Gantt-chart exercise.
 *
 *   ./build/sched_demo            all four policies
 *   ./build/sched_demo sjf        just one
 *
 * The device starts paused and every job is queued first, so all policies see
 * the same queue (as if all processes arrive at t = 0).
 */
#include <stdio.h>
#include <string.h>

#include "vgpu/vgpu.h"

struct workload {
    int      priority;
    uint64_t bytes; /* each job fills this much memory: bigger = longer burst */
};

/* ctx 0: long, low priority   ctx 1: short, high priority
 * ctx 2: medium               ctx 3: short-ish             */
static const struct workload WORK[] = {{1, 8u << 20}, {5, 1u << 20}, {3, 4u << 20}, {2, 2u << 20}};
#define NCTX         4
#define JOBS_PER_CTX 4
#define NJOBS        (NCTX * JOBS_PER_CTX)

/* Queue the workload on a paused device, release it, wait. Returns the device. */
static vgpu_device *run_workload(enum vgpu_sched_policy policy)
{
    struct vgpu_config cfg;
    vgpu_config_default(&cfg);
    cfg.vram_bytes   = 64ull << 20;
    cfg.policy       = policy;
    cfg.start_paused = 1;

    vgpu_device *dev;
    if (vgpu_device_create(&cfg, &dev) != 0)
        return NULL;

    uint32_t ctx[NCTX];
    for (int i = 0; i < NCTX; i++) {
        uint64_t va;
        if (vgpu_ctx_create(dev, WORK[i].priority, &ctx[i]) != 0 ||
            vgpu_bo_create(dev, ctx[i], WORK[i].bytes, &va) != 0) {
            vgpu_device_destroy(dev);
            return NULL;
        }
        for (int j = 0; j < JOBS_PER_CTX; j++) {
            struct vgpu_cmd cmd = vgpu_cmd_fill(va, WORK[i].bytes, (uint32_t)j);
            vgpu_submit(dev, ctx[i], &cmd, NULL, NULL, NULL);
        }
    }
    vgpu_device_start(dev);
    for (int i = 0; i < NCTX; i++)
        vgpu_ctx_finish(dev, ctx[i], VGPU_WAIT_FOREVER);
    return dev;
}

static int report(enum vgpu_sched_policy policy)
{
    /* Warm-up run, discarded: the first run pays for cold caches and CPU clock ramp-up. */
    vgpu_device *dev = run_workload(policy);
    if (!dev)
        return 1;
    vgpu_device_destroy(dev);

    dev = run_workload(policy);
    if (!dev)
        return 1;
    struct vgpu_gantt_rec rec[NJOBS];
    int n = vgpu_get_gantt(dev, rec, NJOBS);
    vgpu_device_destroy(dev);

    double wait_ms[NCTX] = {0}, total = 0;
    char order[NJOBS + 1] = {0};
    for (int i = 0; i < n; i++) {
        double w = (double)(rec[i].start_ns - rec[i].submit_ns) / 1e6;
        wait_ms[rec[i].ctx_id] += w / JOBS_PER_CTX;
        total += w / n;
        order[i] = (char)('0' + rec[i].ctx_id);
    }
    printf("%-9s order %s   avg wait: ", vgpu_policy_name(policy), order);
    for (int c = 0; c < NCTX; c++)
        printf("ctx%d %.2f  ", c, wait_ms[c]);
    printf("| all %.2f ms\n", total);
    return 0;
}

int main(int argc, char **argv)
{
    printf("4 contexts x 4 jobs. Priorities: ctx0=1 ctx1=5 ctx2=3 ctx3=2. "
           "Job size: ctx0 > ctx2 > ctx3 > ctx1\n\n");
    for (int p = 0; p < VGPU_SCHED_POLICY_COUNT; p++) {
        if (argc > 1 && strcmp(argv[1], vgpu_policy_name((enum vgpu_sched_policy)p)) != 0)
            continue;
        if (report((enum vgpu_sched_policy)p) != 0) {
            fprintf(stderr, "device setup failed\n");
            return 1;
        }
    }
    return 0;
}
