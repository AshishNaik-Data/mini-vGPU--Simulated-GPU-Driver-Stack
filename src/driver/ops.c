#include "ops.h"

#include "vgpu/uapi.h"

void vgpu_op_fill(uint32_t *dst, size_t count, uint32_t value)
{
    for (size_t i = 0; i < count; i++)
        dst[i] = value;
}

void vgpu_op_grayscale(uint8_t *dst, const uint8_t *rgba, uint32_t width, uint32_t height)
{
    size_t n = (size_t)width * height;
    for (size_t i = 0; i < n; i++) {
        const uint8_t *p = rgba + 4 * i;
        dst[i] = (uint8_t)((77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8);
    }
}

static inline uint32_t clampi(int64_t v, uint32_t hi)
{
    return v < 0 ? 0u : (v > (int64_t)hi ? hi : (uint32_t)v);
}

/* Pixel at (x, y) with out-of-range coordinates clamped to the nearest edge. */
static inline int px(const uint8_t *img, uint32_t w, uint32_t h, int64_t x, int64_t y)
{
    return img[(size_t)clampi(y, h - 1) * w + clampi(x, w - 1)];
}

void vgpu_op_blur3_rows(uint8_t *dst, const uint8_t *src, uint32_t width, uint32_t height,
                        uint32_t y0, uint32_t y1)
{
    for (uint32_t y = y0; y < y1; y++) {
        for (uint32_t x = 0; x < width; x++) {
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    sum += px(src, width, height, (int64_t)x + dx, (int64_t)y + dy);
            dst[(size_t)y * width + x] = (uint8_t)(sum / 9);
        }
    }
}

void vgpu_op_blur3(uint8_t *dst, const uint8_t *src, uint32_t width, uint32_t height)
{
    vgpu_op_blur3_rows(dst, src, width, height, 0, height);
}

void vgpu_op_sobel_rows(uint8_t *dst, const uint8_t *src, uint32_t width, uint32_t height,
                        uint32_t y0, uint32_t y1)
{
    for (uint32_t y = y0; y < y1; y++) {
        for (uint32_t x = 0; x < width; x++) {
            int64_t X = x, Y = y;
            int gx = -px(src, width, height, X - 1, Y - 1) + px(src, width, height, X + 1, Y - 1)
                     - 2 * px(src, width, height, X - 1, Y) + 2 * px(src, width, height, X + 1, Y)
                     - px(src, width, height, X - 1, Y + 1) + px(src, width, height, X + 1, Y + 1);
            int gy = -px(src, width, height, X - 1, Y - 1) - 2 * px(src, width, height, X, Y - 1)
                     - px(src, width, height, X + 1, Y - 1) + px(src, width, height, X - 1, Y + 1)
                     + 2 * px(src, width, height, X, Y + 1) + px(src, width, height, X + 1, Y + 1);
            int mag = (gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy);
            dst[(size_t)y * width + x] = (uint8_t)(mag > 255 ? 255 : mag);
        }
    }
}

void vgpu_op_sobel(uint8_t *dst, const uint8_t *src, uint32_t width, uint32_t height)
{
    vgpu_op_sobel_rows(dst, src, width, height, 0, height);
}

uint64_t vgpu_op_estimate_cost(uint16_t op, uint64_t n)
{
    switch (op) {
    case VGPU_OP_FILL:      return n;         /* n = bytes written           */
    case VGPU_OP_COPY:      return 2 * n;     /* read + write                */
    case VGPU_OP_GRAYSCALE: return 5 * n;     /* n pixels: 4 in + 1 out      */
    case VGPU_OP_BLUR3:     return 10 * n;    /* 9 taps + 1 write            */
    case VGPU_OP_SOBEL:     return 13 * n;    /* 12 taps + 1 write           */
    default:                return 1;
    }
}
