/*
 * uapi.h - the binary command format the driver puts in the command ring.
 *
 * Only fixed-width types and an explicit, padding-free layout, so the same
 * header could later be shared with a kernel module. Sizes are checked at
 * compile time: an accidental layout change breaks the build, not the device.
 */
#ifndef VGPU_UAPI_H
#define VGPU_UAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define VGPU_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
extern "C" {
#else
#define VGPU_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

#define VGPU_PAGE_SHIFT 12u
#define VGPU_PAGE_SIZE  (1u << VGPU_PAGE_SHIFT) /* 4 KiB, same as the CPU */
#define VGPU_PAGE_MASK  (VGPU_PAGE_SIZE - 1u)

/* GPU virtual address 0 is never mapped, so a NULL VA faults like a NULL pointer. */
#define VGPU_VA_BASE    ((uint64_t)VGPU_PAGE_SIZE)

enum vgpu_op {
    VGPU_OP_NOP       = 0,
    VGPU_OP_FILL      = 1, /* fill dst with a 32-bit pattern               */
    VGPU_OP_COPY      = 2, /* "DMA" copy src -> dst                        */
    VGPU_OP_GRAYSCALE = 3, /* RGBA8 image -> 8-bit gray                    */
    VGPU_OP_BLUR3     = 4, /* 3x3 box blur on 8-bit gray, clamp-to-edge     */
    VGPU_OP_SOBEL     = 5, /* Sobel edge magnitude (|gx|+|gy|, saturated)  */
    VGPU_OP_COUNT
};

/* 16 bytes: every command starts with this header. */
struct vgpu_cmd_hdr {
    uint16_t op;        /* enum vgpu_op                                   */
    uint16_t flags;     /* reserved, must be 0                            */
    uint32_t ctx_id;    /* filled in by the driver on submit              */
    uint64_t fence_seq; /* filled in by the driver on submit              */
};

/*
 * One command = header + a union of per-op payloads. The union makes every
 * command the same size (40 bytes), so the ring is a plain array.
 */
struct vgpu_cmd {
    struct vgpu_cmd_hdr hdr;
    union {
        struct {
            uint64_t dst_va;
            uint64_t size;   /* bytes, multiple of 4 */
            uint32_t value;
            uint32_t _pad;
        } fill;
        struct {
            uint64_t dst_va;
            uint64_t src_va;
            uint64_t size;   /* bytes */
        } copy;
        struct {
            uint64_t dst_va;
            uint64_t src_va;
            uint32_t width;
            uint32_t height;
        } image;             /* GRAYSCALE, BLUR3, SOBEL */
    } u;
};

VGPU_STATIC_ASSERT(sizeof(struct vgpu_cmd_hdr) == 16, "vgpu_cmd_hdr layout changed");
VGPU_STATIC_ASSERT(sizeof(struct vgpu_cmd) == 40, "vgpu_cmd layout changed");
VGPU_STATIC_ASSERT(offsetof(struct vgpu_cmd, u) == 16, "payload must follow header");

#ifdef __cplusplus
}
#endif

#endif /* VGPU_UAPI_H */
