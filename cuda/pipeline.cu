// pipeline.cu - the vGPU edge pipeline (grayscale -> 3x3 blur -> Sobel) on a real GPU.
//
//   nvcc -O3 -arch=native pipeline.cu -o pipeline
//   ./pipeline [width] [height] [frames]
//
// Same integer math as src/driver/ops.c, so every GPU result must be
// byte-identical to the CPU reference. Variants:
//   naive          one thread per pixel, every tap read from global memory
//   tiled          a block loads a (16+2)x(16+2) tile into shared memory once
//   with copies    tiled, plus the PCIe upload and download: the honest number
//   batch-*        many frames (think video): pageable vs pinned host memory,
//                  1 stream vs 4 streams that overlap copies with kernels
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,    \
                         #call, cudaGetErrorString(err_));                        \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

// ---- math shared by CPU and GPU (keep in sync with src/driver/ops.c) --------------

__host__ __device__ inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

__host__ __device__ inline uint8_t gray_of(uint8_t r, uint8_t g, uint8_t b)
{
    return static_cast<uint8_t>((77u * r + 150u * g + 29u * b) >> 8);
}

// n[dy + 1][dx + 1] is the pixel at (x + dx, y + dy).
__host__ __device__ inline uint8_t blur_of(const int n[3][3])
{
    int s = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            s += n[i][j];
    return static_cast<uint8_t>(s / 9);
}

__host__ __device__ inline uint8_t sobel_of(const int n[3][3])
{
    int gx = -n[0][0] + n[0][2] - 2 * n[1][0] + 2 * n[1][2] - n[2][0] + n[2][2];
    int gy = -n[0][0] - 2 * n[0][1] - n[0][2] + n[2][0] + 2 * n[2][1] + n[2][2];
    int m  = (gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy);
    return static_cast<uint8_t>(m > 255 ? 255 : m);
}

enum { OP_BLUR = 0, OP_SOBEL = 1 };

// ---- CPU reference -------------------------------------------------------------------

void cpu_pipeline(const uint8_t* rgba, uint8_t* edges, int w, int h)
{
    std::vector<uint8_t> gray(static_cast<size_t>(w) * h), blur(gray.size());
    for (size_t i = 0; i < gray.size(); i++)
        gray[i] = gray_of(rgba[4 * i], rgba[4 * i + 1], rgba[4 * i + 2]);

    auto stencil = [&](const uint8_t* src, uint8_t* dst, int op) {
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int n[3][3];
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        n[dy + 1][dx + 1] = src[clampi(y + dy, 0, h - 1) * w + clampi(x + dx, 0, w - 1)];
                dst[y * w + x] = op == OP_BLUR ? blur_of(n) : sobel_of(n);
            }
    };
    stencil(gray.data(), blur.data(), OP_BLUR);
    stencil(blur.data(), edges, OP_SOBEL);
}

// ---- kernels ---------------------------------------------------------------------------

__global__ void k_gray(const uchar4* __restrict__ rgba, uint8_t* __restrict__ gray, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uchar4 p = rgba[i];  // one 4-byte load per pixel: neighbours in a warp coalesce
        gray[i]  = gray_of(p.x, p.y, p.z);
    }
}

// Naive: each of the 9 taps is a separate global-memory read.
template <int OP>
__global__ void k_stencil_naive(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    int n[3][3];
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            n[dy + 1][dx + 1] = src[clampi(y + dy, 0, h - 1) * w + clampi(x + dx, 0, w - 1)];
    dst[y * w + x] = OP == OP_BLUR ? blur_of(n) : sobel_of(n);
}

constexpr int TILE = 16;

// Tiled: the block cooperatively loads its tile plus a 1-pixel halo into shared
// memory, so each input pixel is read from global memory about once, not 9 times.
template <int OP>
__global__ void k_stencil_tiled(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int w, int h)
{
    __shared__ uint8_t tile[TILE + 2][TILE + 2];
    int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
    int tid = threadIdx.y * TILE + threadIdx.x;

    for (int i = tid; i < (TILE + 2) * (TILE + 2); i += TILE * TILE) {
        int ty = i / (TILE + 2), tx = i % (TILE + 2);
        int gx = clampi(bx + tx - 1, 0, w - 1), gy = clampi(by + ty - 1, 0, h - 1);
        tile[ty][tx] = src[gy * w + gx];  // clamping here gives clamp-to-edge borders
    }
    __syncthreads();  // every thread must reach this: no early return above

    int x = bx + threadIdx.x, y = by + threadIdx.y;
    if (x < w && y < h) {
        int n[3][3];
        for (int dy = 0; dy < 3; dy++)
            for (int dx = 0; dx < 3; dx++)
                n[dy][dx] = tile[threadIdx.y + dy][threadIdx.x + dx];
        dst[y * w + x] = OP == OP_BLUR ? blur_of(n) : sobel_of(n);
    }
}

// ---- device buffers and launch -------------------------------------------------------------

struct Frame {
    uint8_t *rgba = nullptr, *gray = nullptr, *blur = nullptr, *edges = nullptr;

    void alloc(size_t pixels)
    {
        CUDA_CHECK(cudaMalloc(&rgba, pixels * 4));
        CUDA_CHECK(cudaMalloc(&gray, pixels));
        CUDA_CHECK(cudaMalloc(&blur, pixels));
        CUDA_CHECK(cudaMalloc(&edges, pixels));
    }
    void release()
    {
        cudaFree(rgba);
        cudaFree(gray);
        cudaFree(blur);
        cudaFree(edges);
    }
};

void launch(const Frame& f, int w, int h, bool tiled, cudaStream_t s)
{
    int n = w * h;
    k_gray<<<(n + 255) / 256, 256, 0, s>>>(reinterpret_cast<const uchar4*>(f.rgba), f.gray, n);
    dim3 block(TILE, TILE), grid((w + TILE - 1) / TILE, (h + TILE - 1) / TILE);
    if (tiled) {
        k_stencil_tiled<OP_BLUR><<<grid, block, 0, s>>>(f.gray, f.blur, w, h);
        k_stencil_tiled<OP_SOBEL><<<grid, block, 0, s>>>(f.blur, f.edges, w, h);
    } else {
        k_stencil_naive<OP_BLUR><<<grid, block, 0, s>>>(f.gray, f.blur, w, h);
        k_stencil_naive<OP_SOBEL><<<grid, block, 0, s>>>(f.blur, f.edges, w, h);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---- helpers -----------------------------------------------------------------------------------

// Same deterministic test image as bench/bench_pipeline.cpp.
std::vector<uint8_t> make_image(uint32_t w, uint32_t h)
{
    std::vector<uint8_t> img(static_cast<size_t>(w) * h * 4);
    uint32_t rng = 12345;
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            rng = rng * 1664525u + 1013904223u;
            uint8_t noise = static_cast<uint8_t>(rng >> 28);
            int64_t dx = static_cast<int64_t>(x) - w / 2, dy = static_cast<int64_t>(y) - h / 2;
            int64_t r = static_cast<int64_t>(std::min(w, h)) / 4;
            bool in_circle = dx * dx + dy * dy < r * r;
            bool in_box = (x / 64 + y / 64) % 7 == 0;
            uint8_t* p = &img[(static_cast<size_t>(y) * w + x) * 4];
            p[0] = static_cast<uint8_t>((x * 255) / w) ^ noise;
            p[1] = in_circle ? 230 : static_cast<uint8_t>((y * 255) / h);
            p[2] = in_box ? 255 : 40;
            p[3] = 255;
        }
    return img;
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename F>
double time_ms(F&& fn, int iters)
{
    std::vector<double> t;
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return median(t);
}

struct Result {
    std::string variant;
    double      ms_per_frame;
    bool        ok;
};

// ---- main ---------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    int w      = argc > 1 ? std::atoi(argv[1]) : 1920;
    int h      = argc > 2 ? std::atoi(argv[2]) : 1080;
    int frames = argc > 3 ? std::atoi(argv[3]) : 16;
    const int kIters = 20, kStreams = 4;
    size_t pixels = static_cast<size_t>(w) * h;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int mem_khz = 0, bus_bits = 0;  // attributes, not prop fields: those were removed in CUDA 13
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    double peak_gbs = 2.0 * mem_khz * 1e3 * (bus_bits / 8) / 1e9;  // DDR: 2 transfers per clock
    std::printf("GPU: %s, %d SMs, %d copy engine(s), theoretical DRAM bandwidth ~%.0f GB/s\n",
                prop.name, prop.multiProcessorCount, prop.asyncEngineCount, peak_gbs);
    std::printf("image %dx%d, batch of %d frames\n\n", w, h, frames);

    std::vector<uint8_t> rgba = make_image(w, h), ref(pixels), out(pixels);
    std::vector<Result> results;

    // 1. CPU, single thread
    double cpu_ms = time_ms([&] { cpu_pipeline(rgba.data(), ref.data(), w, h); }, 3);
    results.push_back({"cpu (1 thread)", cpu_ms, true});

    // 2-3. one frame, kernels only (timed with CUDA events on the GPU clock)
    Frame f;
    f.alloc(pixels);
    CUDA_CHECK(cudaMemcpy(f.rgba, rgba.data(), pixels * 4, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    double naive_kernel_ms = 0, tiled_kernel_ms = 0;
    for (bool tiled : {false, true}) {
        launch(f, w, h, tiled, 0);  // warm-up
        CUDA_CHECK(cudaEventRecord(e0));
        for (int i = 0; i < kIters; i++)
            launch(f, w, h, tiled, 0);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CUDA_CHECK(cudaMemcpy(out.data(), f.edges, pixels, cudaMemcpyDeviceToHost));
        double per = ms / kIters;
        (tiled ? tiled_kernel_ms : naive_kernel_ms) = per;
        results.push_back({tiled ? "gpu tiled, kernels only" : "gpu naive, kernels only", per, out == ref});
    }

    // 4. one frame end to end: upload + kernels + download, pageable host memory
    double e2e = time_ms([&] {
        CUDA_CHECK(cudaMemcpy(f.rgba, rgba.data(), pixels * 4, cudaMemcpyHostToDevice));
        launch(f, w, h, true, 0);
        CUDA_CHECK(cudaMemcpy(out.data(), f.edges, pixels, cudaMemcpyDeviceToHost));
    }, kIters);
    results.push_back({"gpu tiled, with copies", e2e, out == ref});

    // 5-7. a batch of frames. Pageable memory: the driver must first copy each frame
    // into a pinned staging buffer, and the copy cannot overlap anything.
    std::vector<uint8_t> pageable_in(pixels * 4 * frames), pageable_out(pixels * frames);
    uint8_t *pinned_in, *pinned_out;  // page-locked: the GPU's DMA engine can read it directly
    CUDA_CHECK(cudaMallocHost(&pinned_in, pixels * 4 * frames));
    CUDA_CHECK(cudaMallocHost(&pinned_out, pixels * frames));
    for (int i = 0; i < frames; i++) {
        std::memcpy(pageable_in.data() + pixels * 4 * i, rgba.data(), pixels * 4);
        std::memcpy(pinned_in + pixels * 4 * i, rgba.data(), pixels * 4);
    }
    auto all_frames_ok = [&](const uint8_t* outp) {
        for (int i = 0; i < frames; i++)
            if (std::memcmp(outp + pixels * i, ref.data(), pixels) != 0)
                return false;
        return true;
    };

    auto one_stream = [&](const uint8_t* in, uint8_t* outp) {
        for (int i = 0; i < frames; i++) {
            CUDA_CHECK(cudaMemcpy(f.rgba, in + pixels * 4 * i, pixels * 4, cudaMemcpyHostToDevice));
            launch(f, w, h, true, 0);
            CUDA_CHECK(cudaMemcpy(outp + pixels * i, f.edges, pixels, cudaMemcpyDeviceToHost));
        }
    };
    double t_pageable = time_ms([&] { one_stream(pageable_in.data(), pageable_out.data()); }, 3);
    results.push_back({"batch, pageable, 1 stream", t_pageable / frames, all_frames_ok(pageable_out.data())});
    double t_pinned = time_ms([&] { one_stream(pinned_in, pinned_out); }, 3);
    results.push_back({"batch, pinned, 1 stream", t_pinned / frames, all_frames_ok(pinned_out)});

    // Four streams, each with its own device buffers so frames in flight never share
    // memory. While stream 1 uploads frame i, stream 0 can run kernels on frame i-1.
    cudaStream_t streams[kStreams];
    Frame sf[kStreams];
    for (int s = 0; s < kStreams; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        sf[s].alloc(pixels);
    }
    std::memset(pinned_out, 0, pixels * frames);
    double t_streams = time_ms([&] {
        for (int i = 0; i < frames; i++) {
            int s = i % kStreams;
            CUDA_CHECK(cudaMemcpyAsync(sf[s].rgba, pinned_in + pixels * 4 * i, pixels * 4,
                                       cudaMemcpyHostToDevice, streams[s]));
            launch(sf[s], w, h, true, streams[s]);
            CUDA_CHECK(cudaMemcpyAsync(pinned_out + pixels * i, sf[s].edges, pixels,
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }, 3);
    results.push_back({"batch, pinned, 4 streams", t_streams / frames, all_frames_ok(pinned_out)});

    // ---- report ----
    std::printf("%-28s %12s %10s %8s\n", "variant", "ms/frame", "speedup", "correct");
    FILE* csv = std::fopen("results.csv", "w");
    std::fprintf(csv, "variant,ms_per_frame,speedup_vs_cpu,correct\n");
    for (const auto& r : results) {
        std::printf("%-28s %12.3f %9.1fx %8s\n", r.variant.c_str(), r.ms_per_frame, cpu_ms / r.ms_per_frame,
                    r.ok ? "yes" : "NO");
        // Quote the name: variants like "gpu naive, kernels only" contain commas.
        std::fprintf(csv, "\"%s\",%.4f,%.2f,%d\n", r.variant.c_str(), r.ms_per_frame, cpu_ms / r.ms_per_frame, r.ok);
    }
    std::fclose(csv);

    // Minimum DRAM traffic per frame: gray reads 4n + writes n, blur n + n, sobel n + n = 8n bytes.
    double bytes = 8.0 * pixels;
    std::printf("\neffective bandwidth: naive %.0f GB/s, tiled %.0f GB/s (peak ~%.0f GB/s)\n",
                bytes / (naive_kernel_ms * 1e6), bytes / (tiled_kernel_ms * 1e6), peak_gbs);

    for (int s = 0; s < kStreams; s++) {
        cudaStreamDestroy(streams[s]);
        sf[s].release();
    }
    cudaFreeHost(pinned_in);
    cudaFreeHost(pinned_out);
    f.release();
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    for (const auto& r : results)
        if (!r.ok)
            return 1;
    return 0;
}
