# Code guide: how mini vGPU works, why it was built this way, and what it is for

This guide walks through every part of the code in the order a job flows through it. It then explains the design choices, how the results were evaluated, and where the ideas are used in real systems. Read it with the code open beside it.

**Contents**
1. [The big picture](#1-the-big-picture)
2. [File-by-file walkthrough](#2-file-by-file-walkthrough)
3. [Why it was built this way](#3-why-it-was-built-this-way)
4. [What the project achieved](#4-what-the-project-achieved)
5. [How the outcomes are evaluated](#5-how-the-outcomes-are-evaluated)
6. [Future use and real-world applications](#6-future-use-and-real-world-applications)

---

## 1. The big picture

### What the project is

A real GPU driver has to solve a handful of problems:
- **Memory:** where do buffers live, and how is one program kept from touching another program's memory?
- **Submission:** how does work get from the application to the hardware?
- **Scheduling:** when several programs want the GPU, who goes next?
- **Completion:** how does the application find out the work is finished?
- **Speed:** where does the time actually go?

mini vGPU solves each of these in a small form. The "GPU" is a **thread** working on a block of ordinary memory, so the whole stack runs on a laptop, and every part can be read, tested and measured.

### The layers

```
 ┌─────────────────── applications ────────────────────────────────────┐
 │ edge_bench.cpp   sched_demo.c   tlb_demo.c   layout_sweep.c   tests │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │  vgpu.h  (the driver's public C API)
 ┌──────────────────────────────▼──────────────────────────────────────┐
 │ DRIVER   device.c   contexts, buffers, ring, fences, engine thread  │
 │          mmu.c      page tables + TLB (address translation)         │
 │          page_alloc.c  which pages of "VRAM" are free               │
 │          sched.c    which job runs next                             │
 │          ops.c      the actual computations (the "shader cores")    │
 │          aligned_alloc.c  page-aligned memory for VRAM              │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────────────┐
 │ "HARDWARE"  one engine thread + a 64 MiB block of memory ("VRAM")   │
 └─────────────────────────────────────────────────────────────────────┘

 cuda/pipeline.cu: the same computation on a real NVIDIA GPU, for comparison
```

### The life of one job

This is the most important thing to be able to explain. Using the blur step as the example:

```
 application                     driver                          "GPU" (engine thread)
 ───────────                     ──────                          ─────────────────────
 vgpu_cmd_image(BLUR3, ...)
 vgpu_submit() ──────────────▶ 1. copy the 40-byte command
                                  into the ring buffer slot
                               2. ring_tail++, assign fence #N
                               3. signal doorbell_cv ──────────▶ 4. wake up, move commands
                                                                    from ring into per-context
                                                                    queues (fetch_from_ring)
                                                                 5. scheduler picks a job
                                                                 6. translate its GPU addresses
                                                                    via the context's page
                                                                    table + TLB (resolve_job)
                                                                 7. unlock, run the kernel
                                                                    (vgpu_op_blur3), relock
                                                                 8. completed_seq = N,
 vgpu_fence_wait(N) ◀─────────────────────────────────────────────  broadcast fence_cv
 returns 0                                                          ("interrupt")
                                                                 9. call the job's callback
```

---

## 2. File-by-file walkthrough

### 2.1 `include/vgpu/uapi.h`: the command format

**Purpose:** defines exactly what one command looks like in memory. "uapi" means "user-space API": in a real driver, this header is shared between the application and the kernel.

**Important parts:**

```c
struct vgpu_cmd_hdr {        /* 16 bytes */
    uint16_t op;             /* which operation: FILL, COPY, GRAYSCALE, BLUR3, SOBEL */
    uint16_t flags;          /* must be 0 */
    uint32_t ctx_id;         /* filled in by the driver */
    uint64_t fence_seq;      /* filled in by the driver */
};

struct vgpu_cmd {
    struct vgpu_cmd_hdr hdr;
    union {                  /* only ONE of these is used, depending on op */
        struct { uint64_t dst_va, size; uint32_t value, _pad; } fill;
        struct { uint64_t dst_va, src_va, size; }             copy;
        struct { uint64_t dst_va, src_va; uint32_t width, height; } image;
    } u;
};
_Static_assert(sizeof(struct vgpu_cmd) == 40, "vgpu_cmd layout changed");
```

- **Why a `union`:** every operation needs different parameters, but the ring buffer is a plain array, so every command must be the same size. A union overlays the alternatives in the same 24 bytes, so every command is 16 + 24 = 40 bytes.
- **Why fixed-width types (`uint64_t`, never `int` or `long`):** the layout must be identical for every compiler and for a kernel module. `long` is 4 bytes on some systems and 8 on others.
- **Why `_pad`:** it makes the padding visible. Otherwise the compiler would insert hidden padding to align the next field.
- **Why `_Static_assert`:** if someone adds a field and changes the size, the **build fails** instead of the driver silently misreading commands. This is a common interview topic (struct padding and `sizeof`).
- `VGPU_VA_BASE = 4096`: the first GPU address handed out is 4096, never 0. So a NULL GPU address always faults, just like a NULL pointer in C.

### 2.2 `include/vgpu/vgpu.h`: the public API

**Purpose:** everything an application can call. The functions group into:

| Group | Functions | Meaning |
|---|---|---|
| Device | `vgpu_device_create/destroy/start` | power the "GPU" on and off |
| Contexts | `vgpu_ctx_create/destroy` | one per application or stream: its own address space and job queue |
| Buffers | `vgpu_bo_create/free/map` | allocate GPU memory; `map` gives the CPU a pointer to it (like `mmap`) |
| Work | `vgpu_submit`, `vgpu_fence_wait`, `vgpu_ctx_finish` | send a command; wait for it to complete |
| Inspection | `vgpu_get_stats`, `vgpu_get_gantt` | TLB hits and misses, memory used, the timeline of finished jobs |

**Conventions:**
- Every function returns **0 or a negative errno** (`-EINVAL`, `-ENOMEM`, `-EAGAIN`, `-EIO`…), exactly like the Linux kernel. So the API could later become a real kernel interface unchanged.
- `vgpu_device` is an **opaque type**: applications only ever hold a pointer to it and can't reach inside. This is encapsulation in C.
- `vgpu_completion_fn` is a **function pointer type**. An application can pass a function that the engine calls when a job finishes.

### 2.3 `src/driver/aligned_alloc.c`: aligned malloc

**Purpose:** device memory ("VRAM") must start on a 4 KiB page boundary, but `malloc` only guarantees 16 bytes. This file is the classic interview answer to "write an aligned malloc".

```c
void *vgpu_aligned_malloc(size_t size, size_t alignment)
{
    /* alignment must be a power of two: exactly one bit set */
    if (size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) return NULL;
    size_t extra = alignment - 1 + sizeof(void *);
    if (size > SIZE_MAX - extra) return NULL;                  /* overflow check */

    void *raw = malloc(size + extra);                          /* over-allocate      */
    uintptr_t start   = (uintptr_t)raw + sizeof(void *);       /* room for a pointer */
    uintptr_t aligned = (start + alignment - 1) & ~(uintptr_t)(alignment - 1); /* round up */
    ((void **)aligned)[-1] = raw;                              /* remember original  */
    return (void *)aligned;
}
void vgpu_aligned_free(void *ptr) { if (ptr) free(((void **)ptr)[-1]); }
```

How it works:
1. Allocate `alignment − 1` extra bytes, so an aligned address is guaranteed to exist inside the block. Add space for one pointer.
2. Round up with the bit trick `(x + a − 1) & ~(a − 1)`, which works because `a` is a power of two.
3. Store the original `malloc` pointer in the 8 bytes just before the returned address, because `free` must receive the original pointer.

### 2.4 `src/driver/page_alloc.c`: which VRAM pages are free

**Purpose:** VRAM is 64 MiB = 16,384 pages of 4 KiB. This allocator tracks which pages are used with **one bit per page** (a bitmap: 16,384 bits = 2 KiB of bookkeeping).

- `vgpu_pa_alloc(count, align)`: **first fit**. It scans for the first run of `count` free pages that starts on a multiple of `align`. When it hits a used page inside a candidate run, it skips past that page instead of moving forward by one, so each page is examined about once.
- `vgpu_pa_free(pfn, count)`: clears the bits.
- `vgpu_pa_count_used()`: counts the 1 bits with `__builtin_popcountll`, one CPU instruction per 64 pages. The tests use it to check that the "used pages" counter never drifts from the real bitmap.
- `vgpu_pa_largest_free_run()`: measures **fragmentation**, meaning free memory that's split into pieces too small to use.

Bit operations used: `bits[pfn / 64] |= 1ull << (pfn % 64)` sets a bit, `&= ~mask` clears it, and `& mask` tests it.

### 2.5 `src/driver/mmu.c`: page tables and the TLB

**Purpose:** each context gets its **own virtual address space**. A GPU address is translated to a physical VRAM address before any access. This is what keeps contexts apart, and it's the same mechanism a CPU uses for processes.

**Address format:**
```
 GPU virtual address:  [ virtual page number (vpn) | 12-bit offset within the page ]
 Page table entry:     [ physical frame number (pfn), 24 bits | flags, 8 bits ]
 flags: VALID (0x1), WRITE (0x2), LARGE (0x4)
```

**The page table** is a single-level array with one 32-bit entry per virtual page: 65,536 entries = 256 MiB of address space per context. Entry 0 is never mapped, so address 0 faults.

**The TLB** (translation lookaside buffer) caches recent translations, so the page table isn't read on every access. It has 16 entries, **fully associative** (any entry can hold any page), with **LRU replacement** (evict the least recently used entry):

```c
static struct vgpu_tlb_entry *tlb_lookup(struct vgpu_tlb *tlb, uint64_t vpn)
{
    for (e = tlb->head; e; e = e->next)          /* most recently used first */
        if (covers(e, vpn)) {
            list_remove(tlb, e);                 /* O(1): doubly linked list */
            list_push_front(tlb, e);             /* now the most recently used */
            return e;
        }
    return NULL;                                 /* miss */
}
```

- The entries form a **doubly linked list in recency order**. A hit moves the entry to the front, and the victim on a miss is always the tail. Both are O(1). **This is exactly the LRU-cache interview question.**
- Invalid entries are kept at the tail, so they get reused before any valid entry is evicted.

**Translation** (`lookup_pte`): TLB hit → done. Otherwise walk the page table. If the entry isn't valid, it's a **page fault** (`-EFAULT`, and `faults++`). Writing to a page without the WRITE flag also faults.

**Unmapping** clears the entries and **invalidates matching TLB entries**. Without that, the engine could keep using a stale translation to memory that has already been freed and handed to someone else. Real operating systems call this a TLB shootdown.

**Large pages** (64 KiB = 16 small pages): if 16 consecutive entries are aligned in both virtual and physical memory, they're marked `LARGE`. The TLB then stores **one entry for the whole group**:

```c
static int covers(const struct vgpu_tlb_entry *e, uint64_t vpn)
{   /* a large entry matches any vpn in its 16-page group */
    return e->valid && (e->large ? e->vpn == (vpn & ~15) : e->vpn == vpn);
}
```

On a hit for page *k* of a group, the physical frame is the group's first frame + *k*. So 16 TLB entries now cover 1 MiB instead of 64 KiB, and that took the hit rate from 0% to 92.4%.

`vgpu_mmu_translate_range` translates a whole buffer and checks that its physical pages are contiguous, so the engine can use a single pointer for it.

### 2.6 `src/driver/sched.c`: which job runs next

**Purpose:** several contexts can have work waiting at once, and the scheduler decides who goes next.

**Structure:** every context has its own **FIFO queue**, a singly linked list with head and tail pointers. Jobs *within* a context always run in order, because later jobs depend on earlier ones (blur must run after grayscale). The policy only chooses **between** contexts, by comparing the job at the head of each queue.

```c
static int runs_before(const struct vgpu_sched *s, uint32_t a, uint32_t b)
{
    switch (s->policy) {
    case VGPU_SCHED_PRIORITY: /* higher context priority wins */
    case VGPU_SCHED_SJF:      /* smaller estimated cost wins */
    case VGPU_SCHED_AGING:    /* higher (priority + age) wins */
    }
    return ja->arrival < jb->arrival;   /* FCFS, and the tie-break for all */
}
```

| Policy | Rule | Strength | Weakness |
|---|---|---|---|
| FCFS | oldest job first | simple, fair by arrival | short jobs wait behind long ones |
| Round-robin | next context in turn, one job each | every context makes progress | ignores urgency |
| Priority | highest priority first | important work is fast | low priority can **starve** |
| SJF | smallest estimated cost first | **lowest average wait** | long jobs wait; needs a cost estimate |
| Aging | priority + 1 for each pick a context waits | priority **without** starvation | slightly higher average wait |

**Aging** in code: after every pick, each context that had work but wasn't chosen gets `age++`, and the chosen one resets to `age = 0`. A waiting context's effective priority keeps rising until it wins.

**Cost estimate for SJF** (`vgpu_op_estimate_cost` in `ops.c`): roughly the number of bytes the operation touches. Sobel reads 12 neighbours and writes 1, so its cost is 13 × pixels.

`vgpu_sched_record` logs every finished job (context, submit, start and end times) into a **circular buffer** of 1,024 records. The scheduler demo prints its Gantt-chart-style results from this log.

Scheduling is **non-preemptive**: once a job starts, it runs to completion. GPUs historically worked this way too.

### 2.7 `src/driver/ops.c`: the computations

**Purpose:** what the "shader cores" actually do. Three image operations form an edge-detection pipeline:

1. **Grayscale:** `gray = (77·R + 150·G + 29·B) >> 8`. The weights reflect how sensitive the eye is to each colour, and they add up to 256, so `>> 8` divides by 256.
2. **3×3 box blur:** each output pixel is the average of the 9 pixels around it, `sum / 9`. It removes noise before edge detection.
3. **Sobel:** measures how fast brightness changes horizontally (`gx`) and vertically (`gy`). Edge strength = `|gx| + |gy|`, capped at 255.

**Edge handling (clamp-to-edge):** a pixel just outside the image takes the value of the nearest edge pixel:
```c
static inline int px(const uint8_t *img, uint32_t w, uint32_t h, int64_t x, int64_t y)
{   return img[(size_t)clampi(y, h - 1) * w + clampi(x, w - 1)];   }
```

**Why integer-only math:** floating-point results can differ slightly between a CPU and a GPU. With integers, the CPU, vGPU and CUDA versions must produce **byte-identical** output, so correctness can be checked exactly.

`vgpu_op_blur3_rows` and `vgpu_op_sobel_rows` compute only rows `[y0, y1)`, but still read neighbours from the whole image. This lets the threaded CPU backend give each thread its own band of rows.

### 2.8 `src/driver/device.c`: the heart of the driver

This is the largest file. It ties all the others together.

#### Data structures

```c
struct bo  { va, size, pfn, npages, phys_pages, next };  /* one buffer (linked list per context) */
struct ctx { in_use, priority, mmu, next_va, bos,
             submitted_seq, completed_seq, first_fault_fence };
struct vgpu_device {
    vram, pages (allocator), ctx[8],
    ring[], slots[], ring_head, ring_tail,        /* the command ring */
    sched, lock, doorbell_cv, fence_cv, engine thread, paused, shutdown
};
```

#### The command ring (producer–consumer)

The ring is a fixed array of commands with two counters. `ring_tail` is advanced by the **producer** (`vgpu_submit`) and `ring_head` by the **consumer** (the engine).
- The slot index is `counter & mask`, where `mask = size − 1` and the size is a power of two. This is a fast modulo.
- The counters are **free-running 32-bit unsigned integers** that never reset. `tail − head` gives the number of commands waiting, and stays correct even after the counters overflow, because unsigned subtraction wraps around too.
- Ring full (`tail − head == size`) → `vgpu_submit` returns `-EAGAIN`, and the caller decides whether to retry.

```c
int vgpu_submit(...)
{
    lock;
    if (ring_tail - ring_head == ring_entries) return -EAGAIN;    /* full */
    slot = ring_tail & ring_mask;
    seq  = ++ctx->submitted_seq;                                 /* this job's fence */
    ring[slot] = *cmd;  ring[slot].hdr.ctx_id = ctx_id;  ring[slot].hdr.fence_seq = seq;
    slots[slot] = {on_done, user, now};                          /* driver bookkeeping */
    ring_tail++;
    pthread_cond_signal(&doorbell_cv);                           /* ring the doorbell */
    unlock;
}
```

#### The engine thread: the "GPU"

```c
static void *engine_thread(void *arg)
{
    lock;
    while (!shutdown) {
        if (paused) { wait(doorbell_cv); continue; }
        fetch_from_ring(dev);                    /* ring → per-context queues     */
        job = vgpu_sched_pick(&sched);           /* the policy decides            */
        if (!job) { wait(doorbell_cv); continue; } /* idle: sleep until a doorbell */

        err = resolve_job(dev, job, &r);         /* translate addresses (MMU)     */
        unlock;
        run_resolved(&r);                        /* compute WITHOUT the lock      */
        lock;

        ctx->completed_seq = job->fence_seq;     /* retire                        */
        if (err) remember first_fault_fence;     /* sticky error                  */
        vgpu_sched_record(...);                  /* timeline log                  */
        pthread_cond_broadcast(&fence_cv);       /* the "interrupt"               */
        if (job->on_done) { unlock; on_done(...); lock; }   /* callback, lock dropped */
        free(job);
    }
}
```

Three points interviewers like:
1. **The kernel runs with the lock released**, so applications can keep submitting while the GPU is busy.
2. **The callback runs with the lock released.** If a callback calls `vgpu_submit`, which needs the same lock, holding it would **deadlock**. `test_callback_can_submit` proves this works.
3. `pthread_cond_wait` **atomically releases the lock and sleeps**, then re-acquires the lock when woken. That's why a condition variable is always used together with a mutex.

#### Address translation for a job (`resolve_job` / `map_range`)

For every buffer a job touches, the engine calls `vgpu_mmu_translate_range` on that context's MMU, with `write = 1` for destinations. Any failure (unmapped page, guard page, a read-only page being written, or another context's address) makes the job a **fault**. The job still "completes", so nobody waits forever, but the context is marked broken.

#### Buffers (`vgpu_bo_create`)

1. Round the size up to whole pages.
2. If large pages are enabled and the buffer is at least 64 KiB, align both the virtual and the physical start to 64 KiB.
3. Allocate physical pages, **plus one extra 64 KiB "colour pad"** for large buffers (explained in §4).
4. Map the whole 64 KiB groups as `LARGE` and any leftover pages as normal 4 KiB pages.
5. Leave one **unmapped guard page** after the buffer in virtual memory, so overruns fault.
6. **Zero the memory**, so no context can read data left behind by another (an information leak).

`vgpu_bo_free` refuses with `-EBUSY` while the context has unfinished jobs, so the engine can't be using memory that has been freed.

#### Fences: waiting for completion

Each context counts `submitted_seq` and `completed_seq`. Fence *N* has signalled when `completed_seq ≥ N`. This works because jobs in a context complete in order.

```c
while (c->completed_seq < fence) {                     /* a LOOP, not an if:    */
    pthread_cond_timedwait(&fence_cv, &lock, &deadline); /* wakeups can be spurious */
}
if (c->first_fault_fence && c->first_fault_fence <= fence) err = -EIO;   /* sticky error */
```

A timeout of 0 means "just check" (polling), and `VGPU_WAIT_FOREVER` means block.

**Sticky errors:** once a job in a context faults, every later wait on that context returns `-EIO`, because its memory may now be inconsistent. CUDA behaves the same way. Other contexts are unaffected.

### 2.9 `cpp/edge_bench.cpp`: the C++ layer and the benchmark

**Purpose:** shows modern C++ design on top of the C driver, and compares four ways of running the pipeline.

| C++ idea | Where | Why |
|---|---|---|
| **Singleton** | `Device::instance()` | one device per process. The function-local `static` is created once, thread-safely (guaranteed since C++11) |
| **RAII** | `Context`, `Buffer` | the constructor acquires the resource and the destructor releases it, so nothing leaks, even if an exception is thrown |
| **Move-only types** | copy constructors `= delete`, move constructors defined | copying a `Buffer` would mean two objects freeing the same memory (a double free) |
| **Abstract class + virtual functions** | `Backend` with `virtual void edge_pipeline(...) = 0` | the benchmark calls every backend the same way: runtime polymorphism through the vtable |
| **Virtual destructor** | `virtual ~Backend() = default` | deleting a derived object through a `Backend*` must run the derived destructor |
| **Factory** | `make_backend("cpu-threads")` | callers choose an implementation by name |
| **Rule of three** | `ArenaCpuBackend` deletes copying | it owns a raw allocation |

**The four backends** run the same three kernels:
- `cpu`: buffers are separate `std::vector`s; one thread.
- `cpu-arena`: all buffers in one aligned block laid out like VRAM; one thread. This exists to **test a hypothesis** (§4, finding A).
- `cpu-threads`: the arena layout, with rows split across all cores. Each stage is fully joined before the next starts, because blur needs the *whole* grayscale image to be finished. **Join acts as a barrier.** Each thread writes only its own rows, so no locks are needed.
- `vgpu`: through the whole driver, with buffers, three submitted commands and one fence wait.

`VgpuBackend` declares its `Context` **before** its `Buffer`s. C++ destroys members in reverse order, so the buffers are freed before the context that owns them.

`main()` generates a deterministic test image (the same random generator every run), warms up, times 10–20 runs per backend, reports the **median**, and **fails if any backend's output differs from the CPU's by even one byte**.

### 2.10 `tools/`: the experiments

| Tool | Experiment | Controlled variable |
|---|---|---|
| `sched_demo.c` | same 16 jobs under all 5 policies | only the policy |
| `tlb_demo.c` | 1080p pipeline with 4 KiB vs 64 KiB pages | only the page size |
| `layout_sweep.c` | Sobel/blur with the output buffer at different distances from the input | only the address |

`sched_demo` starts the device **paused**, queues every job, then releases it. Every policy therefore sees exactly the same queue, like the textbook exercise where all processes arrive at t = 0.

### 2.11 `tests/test_driver.c`: proving it works

17 tests and 189 checks, with a tiny harness (`CHECK`, `CHECK_EQ`, `RUN`) and no external libraries. Every feature has a test, including the failure cases:

| Test | What it proves |
|---|---|
| `test_uapi_layout` | the command is exactly 40 bytes, with fields at the expected offsets |
| `test_aligned_malloc` | every alignment from 1 to 8192 works; bad input returns NULL; no overflow |
| `test_page_alloc` | alignment, out-of-memory, reusing freed holes, fragmentation |
| `test_mmu_translate` | correct translation, NULL fault, read-only fault, TLB flushed on unmap |
| `test_tlb_lru` | exact LRU eviction order |
| `test_large_pages_in_mmu` | alignment rules, 2 misses for 32 pages, correct addresses |
| `test_ops` | known-answer checks (pure red → 76; flat image → no edges) |
| `test_fill_then_copy` | jobs in a context run in order; buffers are freed with the context |
| `test_guard_page_fault_is_sticky` | 4 bytes past a buffer faults, and the error persists |
| `test_context_isolation` | one context cannot touch another's memory |
| `test_ring_full_timeout_and_busy` | `-EAGAIN`, poll and timed waits, `-EBUSY`, double free caught |
| `test_large_pages_raise_tlb_hit_rate` | under 1% → over 90% hits on a real job |
| `test_sched_policies` | the exact job order for each policy |
| `test_aging_prevents_starvation` | priority runs ctx 0 last; aging runs it 5th |
| `test_callback_can_submit` | no deadlock when a callback submits work |
| `test_multithreaded_stress` | 6 threads × 400 jobs: correct data, no leaked memory |
| `test_bad_arguments` | invalid sizes, contexts and opcodes are rejected; context limit enforced |

### 2.12 `cuda/pipeline.cu`: the real GPU

**Kernels:**
- `k_gray`: one thread per pixel. It reads a `uchar4` (4 bytes at once), so the 32 threads of a warp read 128 consecutive bytes in one transaction. This is **memory coalescing**.
- `k_stencil_naive`: one thread per pixel; each of the 9 neighbours is read from global memory.
- `k_stencil_tiled`: a 16×16 block of threads first loads an 18×18 tile (including the 1-pixel border, or "halo") into fast **shared memory**, then computes from there. `__syncthreads()` makes every thread wait until the tile is fully loaded. No thread may `return` before that barrier, so the bounds check comes **after** it.

**Measurements in `main`:**
1. single-threaded CPU reference
2. naive and tiled kernels alone, timed with `cudaEvent`s, which use the GPU's own clock
3. one frame including upload and download
4. a batch of 16 frames from **pageable** memory, 1 stream
5. the same from **pinned** memory (`cudaMallocHost`), 1 stream
6. pinned memory with **4 streams**, so one frame's upload overlaps another frame's computation

```c
for (int i = 0; i < frames; i++) {
    int s = i % kStreams;                       /* round-robin over 4 streams     */
    cudaMemcpyAsync(sf[s].rgba, in_i, ..., HostToDevice, streams[s]);  /* upload   */
    launch(sf[s], w, h, true, streams[s]);                             /* compute  */
    cudaMemcpyAsync(out_i, sf[s].edges, ..., DeviceToHost, streams[s]); /* download */
}
cudaDeviceSynchronize();
```

Each stream has its own device buffers, so frames in flight never overwrite each other. A CUDA stream is the same idea as a vGPU context: in order inside, independent between streams.

### 2.13 `Makefile`

`make` builds the driver into a static library, `libvgpu.a`, and links each program against it. `make asan` and `make tsan` rebuild everything in separate directories with sanitizers enabled. Every experiment is a single `make` target, which keeps results reproducible.

---

## 3. Why it was built this way

| Decision | Alternative | Why this one |
|---|---|---|
| **Simulate the GPU in user space** | write a real Linux kernel driver for real hardware | no special hardware needed; a bug is a test failure, not a frozen machine; sanitizers and debuggers work; the same concepts carry over directly to a kernel module |
| **Driver in C** | C++ or Rust | real GPU drivers and the Linux kernel are written in C, and the job description asks for it; C makes memory layout explicit (structs, unions, pointers) |
| **C++ layer on top** | C only | shows RAII, polymorphism and safe resource handling: what NVIDIA's user-space libraries look like |
| **Integer-only kernels** | floating point | byte-for-byte comparison between CPU, vGPU and CUDA gives a strong correctness check |
| **Command ring + doorbell** | function calls straight into the engine | this is how real GPUs receive work: decoupled producer and consumer, with batching |
| **Page tables + TLB** | raw pointers | isolation between contexts, catching NULL and overrun bugs, and measurable translation behaviour |
| **Per-context in-order queues** | one global queue | preserves dependencies inside a context while still letting the policy choose between contexts, like CUDA streams |
| **One big lock** | fine-grained locks | simple to prove correct with ThreadSanitizer; measurement showed the driver path costs only ~1%, so finer locking wasn't needed |
| **Negative errno returns** | exceptions, or custom codes | the kernel convention, so the API could become an `ioctl` interface unchanged |
| **Experiments that change one variable** | a single big benchmark | when a number looks surprising, the cause can be isolated (this is how findings A and C were solved) |
| **CUDA on Colab** | buying a GPU | free, a real data-centre GPU, reproducible by anyone |

---

## 4. What the project achieved

| Area | Outcome | Why it happens |
|---|---|---|
| **Correctness** | 17 tests, 189 checks pass, including under ASan, UBSan and TSan; every backend and every CUDA variant byte-identical | tests for every feature and failure path; integer math |
| **Scheduling** | SJF gives the lowest average wait (0.33 ms); aging cuts the lowest-priority context's wait by 17–23% and provably prevents starvation | short jobs first minimises total waiting; aging raises the priority of anyone kept waiting |
| **TLB** | hit rate 0% → **92.4%** with 64 KiB pages | each job streams ~2,000 pages through 16 entries, which is LRU's worst case; large pages make each entry cover 16× more memory |
| **Finding A** | the vGPU *looked* 1.7× faster than the CPU; with the same layout, the CPU matches it (4.72 vs 4.79 ms) | buffer placement alone makes identical code 2–3× slower; the driver overhead is only ~1% |
| **Finding C** | large pages first made the pipeline 3.5× *slower* (15.0 ms); a 64 KiB colour pad fixed it (4.2 ms) | 64 KiB alignment put every buffer exactly 2 MiB apart, which the CPU's caches handle badly; padding breaks the pattern (page colouring) |
| **Multi-core** | 1.44 ms per frame on 10 threads | rows are independent within a stage |
| **Real GPU** | kernels 72× faster than one CPU thread | thousands of threads, one per pixel |
| **Finding D** | shared-memory tiling was *not* faster | the T4's L1 cache already serves the neighbour reads; the kernels are limited by one-byte accesses, not memory bandwidth |
| **Finding E** | copies are 89% of a frame's time; pinned memory + 4 streams → 3.0× faster (0.856 ms) | pinned memory avoids an extra staging copy; streams overlap copies with computation; PCIe upload is now the limit |

The main lesson is that the biggest effects came from **memory**: where buffers sit, how translation is cached, and how data crosses PCIe. The arithmetic itself mattered much less.

---

## 5. How the outcomes are evaluated

### 5.1 Correctness: does it do the right thing?

1. **Unit tests** check each part on its own, with known answers: exact addresses, exact eviction order, exact job order per policy.
2. **Failure-path tests**: invalid input, out of memory, a full ring, timeouts, faults, double free, crossing between contexts. A driver is judged by how it handles bad input as much as good input.
3. **Stress test:** 6 threads at once, then a check that data is correct and VRAM usage returns to 0 (no leaks).
4. **Sanitizers:**
   - AddressSanitizer catches out-of-bounds access, use-after-free and leaks.
   - UBSan catches undefined behaviour such as signed overflow.
   - ThreadSanitizer catches data races, and was run 10 times to catch rare interleavings.
5. **Golden output:** every backend and every CUDA variant must match the CPU reference **byte for byte**, including at awkward sizes (97×61, 3×1) that catch edge-handling bugs.

### 5.2 Performance: is it fast, and do we know why?

| Rule | How it is applied |
|---|---|
| **Warm up first** | every tool runs once and discards the result, so cold caches and CPU clock ramp-up don't distort the numbers |
| **Repeat, and use a robust statistic** | median of 10–20 runs, or best of N for microbenchmarks |
| **Change one variable at a time** | `tlb_demo` changes only the page size; `layout_sweep` only the address; `cpu-arena` vs `cpu` only the layout |
| **Use the right clock** | `CLOCK_MONOTONIC` on the CPU (never goes backwards); `cudaEvent` on the GPU (measures GPU time, not launch time) |
| **Report variation honestly** | scheduling times vary up to 2× between runs, so the report treats the *ordering* as the result |
| **Explain before claiming** | the "vGPU is faster" result was investigated until it was explained, not reported as a speedup |

### 5.3 Metrics

| Metric | Measures |
|---|---|
| Average waiting time per context | scheduling quality and fairness |
| Job order | that each policy is implemented correctly |
| TLB hit rate (`hits / lookups`) | effectiveness of address-translation caching |
| ms per frame, megapixels per second | throughput |
| Speedup vs one CPU thread | benefit of parallel hardware |
| Effective bandwidth (bytes ÷ time) vs the hardware peak | whether a kernel is limited by memory or by computation |
| Share of time in copies | the cost of moving data to and from the GPU |

---

## 6. Future use and real-world applications

### 6.1 Where these exact ideas run in real systems

| Real system | Which part of this project it corresponds to |
|---|---|
| **NVIDIA GPU drivers** (Linux and Windows) | command ring, doorbell, fences, per-process GPU virtual memory, channels and contexts |
| **Running several jobs on one GPU** (time-slicing, MPS, MIG in data centres) | the scheduler and its policies, context isolation, starvation and fairness |
| **Cloud GPUs shared between customers** | isolation (`test_context_isolation`), zeroing memory before reuse, sticky faults that don't affect other tenants |
| **GPU virtualisation** (vGPU in virtual machines) | the whole design: a software layer presenting a "GPU" and translating guest addresses, the same idea as this simulator |
| **Operating systems** | page tables, TLBs, huge pages, page colouring, LRU, producer–consumer rings: all used in every OS kernel |
| **CUDA applications** | streams, pinned memory, overlapping copies with computation |

### 6.2 Real-time applications of the image pipeline and its optimisations

Edge detection (grayscale → blur → Sobel) is a standard first stage in computer vision. Lessons such as "overlap copies with computation" and "data movement dominates" apply directly to:

- **Self-driving and driver-assistance systems** (for example NVIDIA DRIVE / Tegra): lane and object detection from cameras at 30–60 frames per second. Each frame must be finished within ~16–33 ms, so streams, pinned memory and predictable scheduling matter.
- **Robotics and drones** (NVIDIA Jetson): obstacle detection and visual navigation on low-power devices, where wasted copies cost battery life.
- **Factory inspection:** finding cracks and defects on a production line from high-speed cameras.
- **Medical imaging:** edge enhancement in X-ray, CT and ultrasound images for faster, clearer diagnosis.
- **Video analytics and surveillance:** many camera streams on one GPU, which needs the scheduling and isolation ideas from the driver.
- **Camera image processing (ISP):** demosaicing, noise reduction and sharpening in phones and cameras. These are stencil operations like blur and Sobel, and the job description lists ISP.
- **AI inference servers:** the same "copies are 89% of the time" problem appears when sending images to a neural network, and is solved the same way (pinned memory, streams, batching).

### 6.3 How the project itself could be used and extended

1. **As a teaching tool:** every OS and GPU concept is visible, testable and measurable in a few thousand lines.
2. **As a policy test bed:** try new scheduling ideas (for example deadline scheduling for real-time video) safely before touching a real driver.
3. **As a fake GPU for testing:** applications written against `vgpu.h` can be tested in CI without a GPU.
4. **Turn it into a Linux kernel module:** `/dev/vgpu0`, with `ioctl` calls reusing `uapi.h` and `mmap` for buffers. The design is already shaped for this (fixed-width command format, errno returns, per-context state).
5. **Drive a virtual PCI device** (QEMU's `edu` device) with real memory-mapped registers, DMA and interrupts.
6. **Real-time scheduling:** add deadlines ("this frame must finish within 33 ms") and measure missed deadlines, as an automotive system would require.
7. **A CUDA backend inside `edge_bench`:** the same `Backend` interface running on a real GPU, so all five backends can be compared in one program.

---

*To check yourself: close this guide, open `device.c`, and explain the engine loop line by line. Then explain why the colour pad exists. If you can do both, you understand the core of the project.*
