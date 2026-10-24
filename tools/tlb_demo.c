/*
 * tlb_demo.c - TLB hit rate of the 1080p edge pipeline, with 4 KiB pages only
 * versus 64 KiB large pages.
 *
 *   ./build/tlb_demo
 */
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "vgpu/vgpu.h"

#define W      1920u
#define H      1080u
#define FRAMES 10

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int run(int large_pages, int quiet)
{
    struct vgpu_config cfg;
    vgpu_config_default(&cfg);
    cfg.large_pages = large_pages;

    vgpu_device *dev;
    uint32_t ctx;
    uint64_t rgba, gray, blur, edges, fence = 0;
    if (vgpu_device_create(&cfg, &dev) || vgpu_ctx_create(dev, 0, &ctx) ||
        vgpu_bo_create(dev, ctx, (uint64_t)W * H * 4, &rgba) || vgpu_bo_create(dev, ctx, W * H, &gray) ||
        vgpu_bo_create(dev, ctx, W * H, &blur) || vgpu_bo_create(dev, ctx, W * H, &edges))
        return 1;
    memset(vgpu_bo_map(dev, ctx, rgba), 128, (size_t)W * H * 4);

    double t0 = now_ms();
    for (int f = 0; f < FRAMES; f++) {
        struct vgpu_cmd cmds[] = {
            vgpu_cmd_image(VGPU_OP_GRAYSCALE, gray, rgba, W, H),
            vgpu_cmd_image(VGPU_OP_BLUR3, blur, gray, W, H),
            vgpu_cmd_image(VGPU_OP_SOBEL, edges, blur, W, H),
        };
        for (int i = 0; i < 3; i++)
            vgpu_submit(dev, ctx, &cmds[i], NULL, NULL, &fence);
    }
    vgpu_fence_wait(dev, ctx, fence, VGPU_WAIT_FOREVER);
    double ms = (now_ms() - t0) / FRAMES;

    struct vgpu_stats s;
    vgpu_get_stats(dev, &s);
    uint64_t lookups = s.tlb_hits + s.tlb_misses;
    if (!quiet)
        printf("%-16s %10llu %10llu %9.1f%% %10.2f\n", large_pages ? "64 KiB pages" : "4 KiB pages",
           (unsigned long long)lookups, (unsigned long long)s.tlb_misses,
           100.0 * (double)s.tlb_hits / (double)lookups, ms);
    vgpu_device_destroy(dev);
    return 0;
}

int main(void)
{
    printf("1080p edge pipeline, %d frames, 16-entry LRU TLB\n\n", FRAMES);
    printf("%-16s %10s %10s %10s %10s\n", "mapping", "lookups", "misses", "hit rate", "ms/frame");
    if (run(0, 1)) /* warm-up, discarded: lets the CPU clock ramp up */
        return 1;
    return run(0, 0) || run(1, 0);
}
