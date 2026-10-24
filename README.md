# mini vGPU

A small simulated GPU and its driver in C, with a C++ layer on top and a CUDA comparison on a real NVIDIA GPU. About 3,200 lines of code including tests. Built to understand, measure and explain how GPU system software works.

**Read the full write-up: [docs/REPORT.md](docs/REPORT.md).**

```
   edge_bench (C++)    sched_demo    tlb_demo    layout_sweep    tests
 ──────────────────────────────────────────────────────────────────────── vgpu.h
   driver:  contexts · buffers · page tables + LRU TLB · 64 KiB large pages
            command ring + doorbell · scheduler (FCFS/RR/priority/SJF/aging) · fences
 ────────────────────────────────────────────────────────────────────────
   "hardware": an engine thread, a VRAM arena, a DMA-style copy

   cuda/: the same pipeline on a real NVIDIA GPU (naive, tiled, pinned + streams)
```

## How a job runs

1. `vgpu_submit()` copies a 40-byte command into the **ring buffer** and rings the **doorbell**.
2. The **engine thread** (the "GPU") moves commands into per-context queues and the **scheduler** picks the next one.
3. The engine translates the job's GPU addresses through that context's **page table and TLB**. A bad address is a **page fault**.
4. It runs the job, then signals the job's **fence** and wakes waiters. This is what an interrupt would do on real hardware.

## Quick start

```bash
make test      # 17 tests, 189 checks
make asan      # the same under AddressSanitizer + UBSan
make tsan      # the same under ThreadSanitizer
make sched     # job order and waiting time for all five scheduling policies
make tlb       # TLB hit rate: 4 KiB vs 64 KiB pages
make bench     # four backends on a 1080p image, writes build/edges.pgm
make layout    # how much buffer placement alone changes kernel speed
```

CUDA part: open `cuda/vgpu_cuda_pipeline.ipynb` in Google Colab, choose a T4 GPU runtime, and run all cells.

## Key results

| What | Result |
|---|---|
| Scheduling | SJF gives the lowest average wait. Aging lets the lowest-priority context run earlier and prevents starvation |
| TLB | 0% hits with 4 KiB pages → **92.4%** with 64 KiB large pages |
| Memory layout | Identical code is 2–3.5× slower at some buffer distances. Fixed with a page-colouring pad |
| Driver overhead | vGPU path 4.79 ms vs the same kernels on the CPU 4.72 ms: about 1% |
| Multi-core CPU | 1.44 ms per 1080p frame on 10 threads (5.5× the naive CPU backend) |
| Real GPU (T4) | Kernels 72× faster than one CPU thread, but copies are 89% of a frame's time. Pinned memory + 4 streams: 0.856 ms per frame, 3.0× faster, now limited by PCIe upload |

## Files

| File | What it does |
|---|---|
| `include/vgpu/uapi.h` | Command format: 16-byte header + union of payloads, size-checked at compile time |
| `include/vgpu/vgpu.h` | The driver's public API |
| `src/driver/device.c` | Ring, doorbell, engine thread, fences, buffers (with large pages and colour pad), contexts |
| `src/driver/mmu.c` | Page table per context, 16-entry LRU TLB, large-page entries |
| `src/driver/page_alloc.c` | Bitmap allocator for device memory |
| `src/driver/aligned_alloc.c` | Hand-written aligned malloc/free |
| `src/driver/sched.c` | Five scheduling policies + a log of finished jobs |
| `src/driver/ops.c` | Image kernels: grayscale, 3×3 blur, Sobel (whole image or a band of rows) |
| `cpp/edge_bench.cpp` | RAII wrappers, Singleton, abstract `Backend` with 4 implementations, benchmark |
| `tools/sched_demo.c`, `tlb_demo.c`, `layout_sweep.c` | The experiments behind the report |
| `tests/test_driver.c` | Tests for every part above |
| `cuda/pipeline.cu`, `cuda/vgpu_cuda_pipeline.ipynb` | CUDA version and its Colab notebook |
| `docs/REPORT.md` | The project report: design, results, findings, lessons |
| `docs/CODE_GUIDE.md` | Walkthrough of every file, why it was built this way, evaluation, real-world uses |
| `docs/STUDY_GUIDE.md` | Theory topics to learn, a 3-week study plan, and 20 practice questions with answers |
| `docs/DESIGN_DECISIONS.md` | Trade-offs to defend in an interview |
| `docs/INTERVIEW_MAP.md` | Which file answers which interview question, plus a 60-second pitch |
| `docs/PLAN.md` | What was done, and what is left |
