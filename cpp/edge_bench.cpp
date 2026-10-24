// edge_bench.cpp - C++ layer on top of the C driver, and the CPU vs vGPU benchmark.
//
//   ./build/edge_bench [--width W] [--height H] [--iters N] [--out edges.pgm]
//
// Shows the C++ ideas interviews ask about, in one place:
//   - RAII + move-only types (Context, Buffer): resources freed exactly once
//   - a thread-safe Singleton (Device)
//   - an abstract base class with virtual functions (Backend -> 4 implementations)
//   - a factory (make_backend)
//
// Backends:
//   cpu          plain std::vector buffers, one thread
//   cpu-arena    one thread, buffers in one arena laid out like VRAM (finding A)
//   cpu-threads  the arena layout, rows split across all CPU cores
//   vgpu         through the driver: ring, scheduler, MMU, engine thread
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C" {
#include "aligned_alloc.h"
}
#include "ops.h"
#include "vgpu/vgpu.h"

namespace {

void check(int err, const char* what)
{
    if (err < 0)
        throw std::runtime_error(std::string(what) + ": " + std::strerror(-err));
}

// ---- Singleton: one device per process -------------------------------------------
// The function-local static is initialised exactly once; C++11 makes that thread-safe.
class Device {
public:
    static Device& instance()
    {
        static Device dev;
        return dev;
    }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    vgpu_device* raw() const { return dev_; }

private:
    Device() { check(vgpu_device_create(nullptr, &dev_), "vgpu_device_create"); }
    ~Device() { vgpu_device_destroy(dev_); }
    vgpu_device* dev_ = nullptr;
};

// ---- RAII wrappers ---------------------------------------------------------------------
// Copying is deleted: two objects freeing the same context or buffer is a double free.
class Context {
public:
    explicit Context(int priority = 0)
    {
        check(vgpu_ctx_create(Device::instance().raw(), priority, &id_), "vgpu_ctx_create");
    }
    ~Context()
    {
        if (live_)
            vgpu_ctx_destroy(Device::instance().raw(), id_);
    }
    Context(Context&& o) noexcept : id_(o.id_), live_(std::exchange(o.live_, false)) {}
    Context& operator=(Context&&)    = delete;
    Context(const Context&)          = delete;
    Context& operator=(const Context&) = delete;

    uint64_t submit(const vgpu_cmd& cmd)
    {
        uint64_t fence;
        check(vgpu_submit(Device::instance().raw(), id_, &cmd, nullptr, nullptr, &fence), "vgpu_submit");
        return fence;
    }
    void wait(uint64_t fence)
    {
        check(vgpu_fence_wait(Device::instance().raw(), id_, fence, VGPU_WAIT_FOREVER), "vgpu_fence_wait");
    }
    uint32_t id() const { return id_; }

private:
    uint32_t id_   = 0;
    bool     live_ = true;
};

class Buffer {
public:
    Buffer(Context& ctx, uint64_t size) : ctx_(ctx.id())
    {
        vgpu_device* dev = Device::instance().raw();
        check(vgpu_bo_create(dev, ctx_, size, &va_), "vgpu_bo_create");
        cpu_ = vgpu_bo_map(dev, ctx_, va_);
    }
    ~Buffer()
    {
        if (!cpu_)
            return;  // moved-from: owns nothing
        vgpu_device* dev = Device::instance().raw();
        vgpu_ctx_finish(dev, ctx_, VGPU_WAIT_FOREVER);  // the engine may still be using it
        vgpu_bo_free(dev, ctx_, va_);
    }
    Buffer(Buffer&& o) noexcept : ctx_(o.ctx_), va_(o.va_), cpu_(std::exchange(o.cpu_, nullptr)) {}
    Buffer& operator=(Buffer&&)      = delete;
    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;

    uint64_t va() const { return va_; }
    void*    data() const { return cpu_; }

private:
    uint32_t ctx_;
    uint64_t va_  = 0;
    void*    cpu_ = nullptr;
};

// ---- Backends: one interface, two implementations --------------------------------------------
class Backend {
public:
    virtual ~Backend() = default;  // virtual: deleting through Backend* runs the derived destructor
    virtual std::string name() const = 0;
    // RGBA8 in -> edge map out: grayscale -> 3x3 blur -> Sobel.
    virtual void edge_pipeline(const uint8_t* rgba, uint32_t w, uint32_t h, uint8_t* edges) = 0;
};

class CpuBackend final : public Backend {
public:
    std::string name() const override { return "cpu"; }
    void edge_pipeline(const uint8_t* rgba, uint32_t w, uint32_t h, uint8_t* edges) override
    {
        gray_.resize(static_cast<size_t>(w) * h);
        blur_.resize(gray_.size());
        vgpu_op_grayscale(gray_.data(), rgba, w, h);
        vgpu_op_blur3(blur_.data(), gray_.data(), w, h);
        vgpu_op_sobel(edges, blur_.data(), w, h);
    }

private:
    std::vector<uint8_t> gray_, blur_;
};

// The same kernels, but every plane lives in one page-aligned arena with a 64 KiB
// pad between planes, like the vGPU's VRAM. Optionally splits rows across threads.
class ArenaCpuBackend final : public Backend {
public:
    explicit ArenaCpuBackend(unsigned threads) : threads_(threads ? threads : 1) {}
    ~ArenaCpuBackend() override { vgpu_aligned_free(arena_); }
    ArenaCpuBackend(const ArenaCpuBackend&)            = delete;  // owns a raw allocation
    ArenaCpuBackend& operator=(const ArenaCpuBackend&) = delete;

    std::string name() const override { return threads_ == 1 ? "cpu-arena" : "cpu-threads"; }

    void edge_pipeline(const uint8_t* rgba, uint32_t w, uint32_t h, uint8_t* edges) override
    {
        layout(w, h);
        size_t pixels = static_cast<size_t>(w) * h;
        std::memcpy(rgba_, rgba, pixels * 4);
        // Each stage needs the whole previous stage (blur reads neighbouring rows),
        // so all threads finish one stage before the next starts: join is the barrier.
        parallel_rows(h, [&](uint32_t y0, uint32_t y1) {
            vgpu_op_grayscale(gray_ + static_cast<size_t>(y0) * w, rgba_ + static_cast<size_t>(y0) * w * 4, w, y1 - y0);
        });
        parallel_rows(h, [&](uint32_t y0, uint32_t y1) { vgpu_op_blur3_rows(blur_, gray_, w, h, y0, y1); });
        parallel_rows(h, [&](uint32_t y0, uint32_t y1) { vgpu_op_sobel_rows(out_, blur_, w, h, y0, y1); });
        std::memcpy(edges, out_, pixels);
    }

private:
    void layout(uint32_t w, uint32_t h)
    {
        if (w == w_ && h == h_)
            return;
        const size_t pad = 64 * 1024, pixels = static_cast<size_t>(w) * h;
        auto next = [&](size_t offset, size_t bytes) { return (offset + bytes + pad - 1) / pad * pad + pad; };
        size_t o_gray = next(0, pixels * 4), o_blur = next(o_gray, pixels), o_out = next(o_blur, pixels);
        vgpu_aligned_free(arena_);
        arena_ = static_cast<uint8_t*>(vgpu_aligned_malloc(o_out + pixels, VGPU_PAGE_SIZE));
        if (!arena_)
            throw std::bad_alloc();
        rgba_ = arena_, gray_ = arena_ + o_gray, blur_ = arena_ + o_blur, out_ = arena_ + o_out;
        w_ = w, h_ = h;
    }

    template <typename Fn>
    void parallel_rows(uint32_t h, Fn fn)
    {
        if (threads_ == 1) {
            fn(0, h);
            return;
        }
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < threads_; t++) {
            uint32_t y0 = static_cast<uint32_t>(static_cast<uint64_t>(h) * t / threads_);
            uint32_t y1 = static_cast<uint32_t>(static_cast<uint64_t>(h) * (t + 1) / threads_);
            pool.emplace_back(fn, y0, y1);  // each thread writes only its own rows: no locks needed
        }
        for (auto& th : pool)
            th.join();
    }

    unsigned threads_;
    uint8_t* arena_ = nullptr;
    uint8_t *rgba_ = nullptr, *gray_ = nullptr, *blur_ = nullptr, *out_ = nullptr;
    uint32_t w_ = 0, h_ = 0;
};

class VgpuBackend final : public Backend {
public:
    std::string name() const override { return "vgpu"; }
    void edge_pipeline(const uint8_t* rgba, uint32_t w, uint32_t h, uint8_t* edges) override
    {
        size_t pixels = static_cast<size_t>(w) * h;
        if (w != w_ || h != h_) {  // (re)allocate device memory only when the size changes
            rgba_.reset(), gray_.reset(), blur_.reset(), out_.reset();
            rgba_.emplace(ctx_, pixels * 4);
            gray_.emplace(ctx_, pixels);
            blur_.emplace(ctx_, pixels);
            out_.emplace(ctx_, pixels);
            w_ = w, h_ = h;
        }
        std::memcpy(rgba_->data(), rgba, pixels * 4);  // upload
        // One context = one in-order queue, so each step sees the previous step's output.
        ctx_.submit(vgpu_cmd_image(VGPU_OP_GRAYSCALE, gray_->va(), rgba_->va(), w, h));
        ctx_.submit(vgpu_cmd_image(VGPU_OP_BLUR3, blur_->va(), gray_->va(), w, h));
        ctx_.wait(ctx_.submit(vgpu_cmd_image(VGPU_OP_SOBEL, out_->va(), blur_->va(), w, h)));
        std::memcpy(edges, out_->data(), pixels);  // download
    }

private:
    Context               ctx_;  // declared first, so it is destroyed after the buffers
    std::optional<Buffer> rgba_, gray_, blur_, out_;
    uint32_t              w_ = 0, h_ = 0;
};

std::unique_ptr<Backend> make_backend(const std::string& name)
{
    if (name == "cpu")
        return std::make_unique<CpuBackend>();
    if (name == "cpu-arena")
        return std::make_unique<ArenaCpuBackend>(1);
    if (name == "cpu-threads")
        return std::make_unique<ArenaCpuBackend>(std::max(1u, std::thread::hardware_concurrency()));
    if (name == "vgpu")
        return std::make_unique<VgpuBackend>();
    throw std::invalid_argument("unknown backend: " + name);
}

// ---- benchmark ----------------------------------------------------------------------------------

// Deterministic test image: gradients, a circle, boxes and a little noise.
std::vector<uint8_t> make_image(uint32_t w, uint32_t h)
{
    std::vector<uint8_t> img(static_cast<size_t>(w) * h * 4);
    uint32_t rng = 12345;
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            rng = rng * 1664525u + 1013904223u;  // LCG
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

void write_pgm(const std::string& path, const uint8_t* data, uint32_t w, uint32_t h)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::perror(path.c_str());
        return;
    }
    std::fprintf(f, "P5\n%u %u\n255\n", w, h);
    std::fwrite(data, 1, static_cast<size_t>(w) * h, f);
    std::fclose(f);
}

}  // namespace

int main(int argc, char** argv)
{
    uint32_t w = 1920, h = 1080;
    int iters = 10;
    std::string out;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if (k == "--width")       w     = static_cast<uint32_t>(std::atoi(argv[i + 1]));
        else if (k == "--height") h     = static_cast<uint32_t>(std::atoi(argv[i + 1]));
        else if (k == "--iters")  iters = std::atoi(argv[i + 1]);
        else if (k == "--out")    out   = argv[i + 1];
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    if (w == 0 || h == 0 || iters <= 0) {
        std::fprintf(stderr, "width, height and iters must be positive\n");
        return 2;
    }

    std::vector<uint8_t> rgba = make_image(w, h);
    size_t pixels = static_cast<size_t>(w) * h;
    std::printf("image %ux%u, %d iterations, %u CPU threads available\n\n%-12s %12s %12s\n", w, h, iters,
                std::thread::hardware_concurrency(), "backend", "median ms", "Mpixel/s");

    std::vector<uint8_t> reference;
    for (const char* name : {"cpu", "cpu-arena", "cpu-threads", "vgpu"}) {
        std::unique_ptr<Backend> backend = make_backend(name);
        std::vector<uint8_t> edges(pixels);
        std::vector<double> ms;
        backend->edge_pipeline(rgba.data(), w, h, edges.data());  // warm-up
        for (int i = 0; i < iters; i++) {
            auto t0 = std::chrono::steady_clock::now();
            backend->edge_pipeline(rgba.data(), w, h, edges.data());  // virtual call
            auto t1 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        double median = ms[ms.size() / 2];
        std::printf("%-12s %12.2f %12.1f\n", backend->name().c_str(), median, pixels / median / 1e3);

        if (reference.empty())
            reference = edges;
        else if (edges != reference) {
            std::fprintf(stderr, "MISMATCH: %s output differs from cpu\n", name);
            return 1;
        }
    }
    std::printf("\noutputs identical: yes\n");

    if (!out.empty()) {
        write_pgm(out, reference.data(), w, h);
        std::printf("edge map written to %s\n", out.c_str());
    }
    return 0;
}
