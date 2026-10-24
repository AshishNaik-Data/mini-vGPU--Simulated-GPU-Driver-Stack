/*
 * layout_sweep.c - how much does it matter where two buffers sit in memory?
 *
 *   ./build/layout_sweep
 *
 * Runs the Sobel and blur kernels with the output buffer placed at different
 * distances after the input buffer, inside one big allocation. Same code, same
 * data: only the distance changes. On Apple Silicon some distances (exactly
 * 2 MiB, or an odd multiple of 16 KiB) are 1.5-3x slower. See finding A in
 * docs/DESIGN_DECISIONS.md.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "aligned_alloc.h"
#include "ops.h"

#define W 1920u
#define H 1080u

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

typedef void (*kernel_fn)(uint8_t *dst, const uint8_t *src, uint32_t w, uint32_t h);

static double best_of(kernel_fn k, uint8_t *dst, const uint8_t *src, int runs)
{
    double best = 1e9;
    for (int i = 0; i < runs; i++) {
        double t = now_ms();
        k(dst, src, W, H);
        double d = now_ms() - t;
        if (d < best)
            best = d;
    }
    return best;
}

int main(void)
{
    const size_t n = (size_t)W * H;          /* one 8-bit 1080p plane: 2,073,600 bytes */
    const size_t base = 2u << 20;            /* output starts 2 MiB after the input... */
    const size_t extra[] = {0, 4096, 8192, 16384, 32768, 49152, 65536, 131072}; /* ...plus this */
    size_t total = base + 131072 + n;

    uint8_t *blk = vgpu_aligned_malloc(total, 2u << 20);
    if (!blk)
        return 1;
    memset(blk, 0, total);
    for (size_t i = 0; i < n; i++)
        blk[i] = (uint8_t)(i * 37);

    double warm = now_ms(); /* let the CPU clock ramp up first */
    while (now_ms() - warm < 500)
        vgpu_op_sobel(blk + base + 4096, blk, W, H);

    printf("%-26s %10s %10s\n", "output - input distance", "sobel ms", "blur ms");
    for (size_t i = 0; i < sizeof(extra) / sizeof(extra[0]); i++) {
        uint8_t *dst = blk + base + extra[i];
        printf("2 MiB + %-18zu %10.2f %10.2f\n", extra[i],
               best_of(vgpu_op_sobel, dst, blk, 10), best_of(vgpu_op_blur3, dst, blk, 10));
    }
    vgpu_aligned_free(blk);
    return 0;
}
