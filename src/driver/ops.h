/*
 * ops.h - the compute kernels the simulated engine runs.
 *
 * They work on plain host pointers, so the CPU backend calls the very same
 * functions. All image math is integer-only, which means the CPU, vGPU and
 * CUDA versions must produce bit-identical output.
 */
#ifndef VGPU_OPS_H
#define VGPU_OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void vgpu_op_fill(uint32_t *dst, size_t count, uint32_t value);

/* gray = (77 R + 150 G + 29 B) >> 8, from RGBA8 (4 bytes per pixel). */
void vgpu_op_grayscale(uint8_t *dst, const uint8_t *rgba, uint32_t width, uint32_t height);

/* 3x3 box blur, integer sum / 9, clamp-to-edge borders. */
void vgpu_op_blur3(uint8_t *dst, const uint8_t *src, uint32_t width, uint32_t height);

/* Sobel magnitude min(255, |gx| + |gy|), clamp-to-edge borders. */
void vgpu_op_sobel(uint8_t *dst, const uint8_t *src, uint32_t width, uint32_t height);

/* Only rows [y0, y1) of dst, still reading neighbours from the whole image:
 * lets several threads split one image into bands. */
void vgpu_op_blur3_rows(uint8_t *dst, const uint8_t *src, uint32_t width, uint32_t height,
                        uint32_t y0, uint32_t y1);
void vgpu_op_sobel_rows(uint8_t *dst, const uint8_t *src, uint32_t width, uint32_t height,
                        uint32_t y0, uint32_t y1);

/* Cost estimate in "work units" used by the SJF scheduler (bytes touched). */
uint64_t vgpu_op_estimate_cost(uint16_t op, uint64_t size_or_pixels);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_OPS_H */
