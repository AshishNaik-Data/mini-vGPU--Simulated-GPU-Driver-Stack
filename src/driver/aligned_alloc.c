#include "aligned_alloc.h"

#include <stdint.h>
#include <stdlib.h>

/*
 * Layout of one allocation:
 *
 *   raw                      aligned (returned)
 *   |  slack  | saved raw ptr | user bytes ...           |
 *              ^ aligned - sizeof(void *)
 *
 * We over-allocate by (alignment - 1) bytes so an aligned address always fits,
 * plus one pointer so free() can find the original malloc() block.
 */
void *vgpu_aligned_malloc(size_t size, size_t alignment)
{
    if (size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0)
        return NULL;
    if (alignment < sizeof(void *))
        alignment = sizeof(void *);

    size_t extra = alignment - 1 + sizeof(void *);
    if (size > SIZE_MAX - extra) /* size + extra would overflow */
        return NULL;

    void *raw = malloc(size + extra);
    if (!raw)
        return NULL;

    uintptr_t start   = (uintptr_t)raw + sizeof(void *);
    uintptr_t aligned = (start + alignment - 1) & ~(uintptr_t)(alignment - 1);

    ((void **)aligned)[-1] = raw;
    return (void *)aligned;
}

void vgpu_aligned_free(void *ptr)
{
    if (!ptr)
        return;
    free(((void **)ptr)[-1]);
}
