# mini vGPU: design and evaluation of a small GPU driver stack

*Project report. All numbers were measured on an Apple M4 laptop (4 performance + 6 efficiency cores) and on an NVIDIA T4 in Google Colab.*

## Abstract

mini vGPU is a simulated GPU and its driver, written in about 3,200 lines of C, C++ and CUDA. The driver gives each context its own virtual address space with page tables and an LRU TLB, and accepts work through a command ring with a doorbell. It schedules jobs with five policies and reports completion through fences. A C++ layer adds RAII resource management and polymorphic backends. The same image pipeline also runs on a real NVIDIA GPU in CUDA.

The main results:
1. Aging removes starvation at little cost to average waiting time.
2. 64 KiB large pages raise the TLB hit rate on images from 0% to 92.4%.
3. Memory placement alone can make identical code 2–3.5× slower. This explained an apparent "driver speedup", and fixing it needed a page-colouring pad.
4. On the T4, data movement, not computation, dominates: copies are 89% of a frame's time. Pinned memory and four overlapping streams make a frame 3.0× faster, until PCIe upload bandwidth becomes the limit.

## 1. Motivation

GPU system software sits between applications and hardware. It manages device memory and address spaces, queues work, decides what runs next and reports completion. These are the operating-system ideas of virtual memory, scheduling, synchronisation and interrupts, applied to an accelerator. The project builds each of them in a form small enough to understand completely, then measures them.

## 2. Design

```
   applications: edge_bench (C++), sched_demo, tlb_demo, tests
 ────────────────────────────────────────────────────────── vgpu.h (C API)
   driver   contexts · buffer objects · page tables + LRU TLB
            command ring + doorbell · scheduler (5 policies) · fences
 ──────────────────────────────────────────────────────────
   "hardware": engine thread · VRAM arena · copy ("DMA") engine
```

**Life of a job.**
1. `vgpu_submit()` writes a 40-byte command into the ring and rings the doorbell.
2. The engine thread moves commands into per-context queues.
3. The scheduler picks the next job.
4. The engine translates its addresses through the context's MMU and runs it with the device lock released.
5. On completion it signals the job's fence, wakes waiters (the "interrupt") and calls the job's completion callback.

**Memory.**
- A bitmap allocator hands out physically contiguous runs of 4 KiB pages.
- Each context has a single-level page table (65,536 entries, 256 MiB of virtual address space) and a 16-entry fully associative TLB with LRU replacement.
- Address 0 is never mapped, so NULL faults.
- An unmapped guard page follows every buffer, so overruns fault.
- Buffers are zeroed when allocated, so one context can't read another's old data.
- Buffers of 64 KiB or more are mapped with 64 KiB large pages, aligned in both virtual and physical memory.

**Command format.** A 16-byte header plus a union of payloads gives a fixed 40-byte command, checked by `_Static_assert`. The ring indices are free-running 32-bit counters, so `tail - head` is the fill level even after wrap-around.

**Scheduling.** Each context has an in-order queue, like a CUDA stream, and the scheduler chooses only between the heads of these queues. Policies: FCFS, round-robin, priority, shortest-job-first (by estimated bytes touched), and priority with aging (+1 effective priority for every pick a waiting context is passed over). Scheduling is non-preemptive.

**Concurrency.** A single device mutex protects all state. Condition variables implement the doorbell and fence waits. Callbacks run with the lock released, so a callback can submit more work without deadlocking.

**Errors.** Functions return negative errno values. A faulting job still signals its fence, but marks its context as failed ("sticky"), so later waits on that context return `-EIO`.

**C++ layer** (`cpp/edge_bench.cpp`): a Meyers singleton for the device, move-only RAII `Context` and `Buffer` types, an abstract `Backend` with four implementations, and a factory.

## 3. Implementation

| Area | Lines | Contents |
|---|---|---|
| `src/driver` | 1,500 | device, MMU/TLB, page allocator, aligned malloc, scheduler, kernels |
| `include` | 183 | command format and public API |
| `cpp` | 349 | C++ layer, four backends, benchmark |
| `tools` | 244 | scheduler, TLB and memory-layout demos |
| `tests` | 558 | 17 tests, 189 checks |
| `cuda` | 364 | CUDA pipeline: naive, tiled, pinned memory, streams |

## 4. Testing

17 tests cover:
- the command layout and the aligned malloc
- allocator placement, alignment and fragmentation
- address translation, read-only and NULL faults, TLB invalidation and LRU eviction order
- large-page translation and hit rates
- the image kernels
- in-order execution, guard-page faults and sticky errors, context isolation
- a full ring, timeouts, and freeing a busy buffer
- the job order produced by every scheduling policy, including a starvation scenario
- callbacks that submit new work
- six threads submitting at once with leak checking

Every test passes as normal, under AddressSanitizer + UBSan, and under ThreadSanitizer (10 repeated runs, no failures). The benchmark checks that all four backends produce byte-identical output, including at odd sizes (97×61, 3×1). The CUDA program checks every GPU result against the CPU.

## 5. Results

### 5.1 Scheduling (4 contexts × 4 jobs, all queued at t = 0)

| Policy | Job order (context ids) | Avg wait, all | Avg wait, ctx 0 (lowest priority, longest jobs) |
|---|---|---|---|
| FCFS | 0000111122223333 | 0.66 ms | 0.30 ms |
| Round-robin | 0123012301230123 | 0.55 ms | 0.47 ms |
| Priority | 1111222233330000 | 0.39 ms | 0.82 ms |
| SJF | 1111333322220000 | **0.33 ms** | 0.75 ms |
| Priority + aging | 1112130232023030 | 0.42 ms | 0.68 ms |

The orders are deterministic and match the textbook. SJF minimises the average wait, as theory predicts for non-preemptive scheduling when all jobs arrive together. Aging lets ctx 0 run earlier, cutting its wait by 17% in this run and 23% in another, at a small cost to the average. With a continuous stream of high-priority work it guarantees progress (`test_aging_prevents_starvation`). Absolute times vary up to 2× between runs, and FCFS and RR swap places, so the ordering is the result, not the milliseconds.

### 5.2 TLB and large pages (1080p pipeline, 10 frames)

| Mapping | Lookups | Misses | Hit rate | ms / frame |
|---|---|---|---|---|
| 4 KiB pages | 45,600 | 45,600 | 0.0% | 4.31 |
| 64 KiB large pages | 45,600 | 3,450 | **92.4%** | 4.23 |
| 64 KiB, *without* the colour pad | 45,600 | 3,450 | 92.4% | 15.0 |

### 5.3 CPU backends (1080p edge pipeline, median of 20)

| Backend | ms / frame | vs `cpu` |
|---|---|---|
| `cpu` (separate `std::vector` buffers) | 7.96 | 1.0× |
| `cpu-arena` (one padded arena) | 4.72 | 1.7× |
| `vgpu` (full driver path) | 4.79 | 1.7× |
| `cpu-threads` (arena, 10 threads) | 1.44 | 5.5× |

### 5.4 Real GPU (NVIDIA T4, 1080p, batches of 16 frames)

| Variant | ms / frame | vs 1 CPU thread |
|---|---|---|
| CPU, 1 thread (Colab VM) | 20.20 | 1× |
| GPU naive, kernels only | 0.279 | 72× |
| GPU shared-memory tiled, kernels only | 0.294 | 69× |
| GPU tiled, one frame with copies (pageable) | 2.56 | 7.9× |
| Batch, pageable memory, 1 stream | 2.76 | 7.3× |
| Batch, pinned memory, 1 stream | 1.19 | 16.9× |
| Batch, pinned memory, 4 streams | **0.856** | **23.6×** |

Every variant produced output byte-identical to the CPU. An earlier run gave the same picture (naive 0.318, tiled 0.326, with copies 2.58 ms). GPU timings vary by about 10% between Colab sessions.

## 6. Findings

**A. The "driver speedup" was memory placement.** At first the vGPU path looked 1.7× faster than the CPU path, although both run the same kernels. `make layout` keeps the code and data fixed and moves only the output buffer. An output buffer an odd multiple of 16 KiB after the input made Sobel and blur 2.3–2.9× slower (5.6 vs 2.45 ms). In another run, buffers exactly 2 MiB apart were 3.3× slower, so the absolute address matters too. A CPU backend using the same arena layout as the vGPU (`cpu-arena`) runs as fast as the vGPU (4.72 vs 4.79 ms). So the gap came entirely from layout, and the driver's own overhead is about 1%. The exact hardware mechanism (cache-set or bank conflicts, or a load/store aliasing predictor) is not confirmed. Confirming it would need hardware counters.

**B. LRU flooding, and the fix.** With 4 KiB pages, every job streams about 2,000 pages through a 16-entry TLB, the classic case where LRU evicts each entry just before reuse, giving 0% hits. 64 KiB pages cover 1 MiB with 16 entries and give 92.4% hits.

**C. Large pages caused a new slowdown, and page colouring fixed it.** Aligning buffers to 64 KiB rounded each 1080p plane to exactly 2 MiB. All planes then sat exactly a power of two apart, and the pipeline ran 3.5× slower (15.0 ms), even though the TLB was hitting. One unused 64 KiB "colour pad" after each large buffer breaks the pattern and restores 4.2 ms. This is the same idea as page colouring in operating systems. The cost is 64 KiB of memory per large buffer.

**D. Shared-memory tiling did not help on the T4.** Tiled 0.294 ms vs naive 0.279 ms, and tiled was also slower in an earlier run (0.326 vs 0.318). The GPU's L1 cache already serves the repeated neighbour reads, and the kernels reach only about 59 GB/s of the T4's ~320 GB/s (minimum traffic 8 bytes per pixel). They are limited by one-byte accesses and instruction overhead, not DRAM bandwidth.

**E. Data movement dominates, and pinned memory plus streams recover most of it.** One frame takes 2.56 ms with copies but only 0.29 ms of compute: **89% is PCIe transfer** (8.3 MB up, 2.1 MB down). Three steps, all in §5.4:
- **Pinned memory: 2.3× faster** (2.76 → 1.19 ms per frame). Copy throughput rises from about 4.2 GB/s to about 11.5 GB/s. From pageable memory, the driver first has to copy each frame into a page-locked staging buffer the DMA engine can read. Pinned memory skips that copy.
- **Four streams: a further 1.4×** (1.19 → 0.856 ms), because one frame's kernels and download run while the next frame uploads.
- **The new limit is upload bandwidth.** At about 11.5 GB/s, uploading one 8.3 MB frame alone takes about 0.72 ms. 0.856 ms is within about 0.14 ms of that, so the kernels and the download are almost fully hidden. More streams cannot help much. Only less data (for example 8-bit input instead of RGBA) or a faster bus can.

These bandwidth figures are estimates, calculated as (time − kernel time) ÷ bytes moved. They are not measured per transfer.

## 7. Lessons

1. **Measure before claiming.** Two apparent results, the vGPU "speedup" and the large-page slowdown, were both memory-layout effects.
2. **Fixes interact.** Large pages fixed the TLB and broke cache behaviour. Page colouring fixed that.
3. **Textbook optimisations depend on the hardware.** Shared-memory tiling, a standard CUDA optimisation, gained nothing on a GPU with an L1 cache.
4. **For accelerators, moving data often matters more than computing on it.**
5. **Simple and correct first.** One big lock and in-order queues were easy to verify with sanitizers, and profiling showed the lock is not the bottleneck: the driver path costs about 1%.

## 8. Limitations and future work

- The "GPU" runs on one CPU core, so the vGPU measures driver mechanics, not GPU speed.
- Scheduling is non-preemptive. Preemptive SJF or MLFQ would need jobs that can be split.
- Buffers must be physically contiguous. Scatter-gather mappings would remove fragmentation.
- `vgpu_bo_free` returns `-EBUSY` while any job in the context is pending. Per-job buffer references would be finer-grained.
- **Kernel module:** move the driver into Linux as `/dev/vgpu0`, with `ioctl` calls reusing `uapi.h`, `copy_from_user` and `mmap`.
- **CUDA:** a fused kernel, 4 pixels per thread, and a stream-count sweep.

## Appendix: reproduce everything

```bash
make test && make asan && make tsan   # correctness
make sched                            # table 5.1
make tlb                              # table 5.2
make bench                            # table 5.3
make layout                           # finding A
```

For table 5.4, open `cuda/vgpu_cuda_pipeline.ipynb` in Colab with a T4 runtime and run all cells.
