# Design decisions and findings

In a project deep-dive, be ready to say what you chose, what you rejected, and why. The findings use the same letters as `REPORT.md`.

## Decisions

1. **One big device lock.** One mutex protects all device state, and kernels run with it released. Easy to prove correct, with no lock ordering to get wrong. Measurement backs it up: the whole driver path costs about 1% (finding A), so the lock is not a bottleneck.

2. **Callbacks run with the lock released.** A callback that calls `vgpu_submit()` would otherwise deadlock on the same mutex. `test_callback_can_submit` proves it works.

3. **In-order queue per context; the scheduler only chooses between contexts.** Jobs in one context depend on each other (grayscale before blur), so they must not be reordered. This is the same model as CUDA streams. Consequence: SJF and priority compare the job at the *head* of each queue.

4. **Aging adds 1 priority per pick a context waits, reset when it runs.** Simple and deterministic, so it can be tested exactly (`test_aging_prevents_starvation`). Rejected: time-based aging, which would make test results depend on timing.

5. **Buffers are physically contiguous.** The MMU still provides isolation, guard pages and NULL faults, but each buffer is one run of pages, so kernels can use a single pointer. The cost is fragmentation. Scatter-gather is an exercise.

6. **Large pages for buffers of 64 KiB or more.** Both virtual and physical start are 64 KiB-aligned, every whole 64 KiB group is marked `VGPU_PTE_LARGE`, and the tail uses 4 KiB pages. One TLB entry then covers 16 pages. The cost is alignment padding in both address spaces.

7. **A colour pad after each large buffer.** One unused 64 KiB block of physical memory, so buffers never sit an exact power of two apart (finding C). The cost is 64 KiB per large buffer. That is about 3% for a 1080p plane, and it buys a 3.5× speedup.

8. **A guard page after every buffer.** A write one byte past the end faults instead of silently corrupting the next buffer.

9. **Faults are sticky.** After a bad access, every later wait on that context returns `-EIO`, because its memory may be inconsistent. CUDA behaves the same way. Other contexts keep running.

10. **A full ring returns `-EAGAIN` instead of blocking.** The caller decides whether to retry, sleep or drop work.

11. **Buffers are zeroed when allocated.** Otherwise one context could read another context's old data.

12. **Threaded CPU backend: one thread per row band, join between stages.** Blur reads neighbouring rows of the previous stage, so a stage must finish completely before the next starts. Joining the threads is that barrier. Each thread writes only its own rows, so no locks are needed.

## Findings

### A. The "driver speedup" was memory placement
The vGPU path first looked 1.7× faster than the CPU path, with identical kernels. `make layout` fixes the code and data and moves only the output buffer: at an odd multiple of 16 KiB from the input, Sobel and blur are 2.3–2.9× slower. A CPU backend using the vGPU's arena layout (`cpu-arena`, 4.72 ms) matches the vGPU (4.79 ms). So the layout explains the whole gap, and the driver costs about 1%.
**Open:** the exact hardware mechanism (cache-set or bank conflicts, or load/store aliasing). Confirming it needs hardware performance counters.

### B. LRU flooding in the TLB, fixed with large pages
With 4 KiB pages, each job streams about 2,000 pages through 16 entries, so LRU evicts every entry just before reuse: 0% hits. With 64 KiB pages, 16 entries cover 1 MiB: 92.4% hits.

### C. Large pages caused a slowdown; page colouring fixed it
Aligning to 64 KiB rounded each 1080p plane to exactly 2 MiB, so all planes were exactly a power of two apart. The pipeline ran at 15.0 ms instead of 4.2 ms, while the TLB was hitting. The colour pad (decision 7) fixes it.

### D. Shared-memory tiling did not help on the T4
Naive 0.279 ms vs tiled 0.294 ms (and 0.318 vs 0.326 in an earlier run). The L1 cache already serves the neighbour reads, and at about 59 GB/s of ~320 GB/s the kernels are not bandwidth-bound. One-byte accesses and instruction overhead dominate.

### E. Data movement dominates on the GPU
2.56 ms per frame with copies, of which 0.29 ms is compute: 89% is PCIe transfer. Pinned memory makes copies 2.3× faster (no staging copy through a page-locked buffer). Four streams overlap kernels and downloads with the next upload for a further 1.4×: 0.856 ms per frame, 3.0× better than one frame with pageable copies. That is close to the time the upload alone takes (~0.72 ms), so PCIe upload bandwidth is now the limit.

### Measurement note
The first run after idle is up to 2–5× slower (cold caches, CPU clock ramp-up). Every tool runs a discarded warm-up and reports a median or best-of-N.
