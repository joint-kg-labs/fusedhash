// Common infrastructure of the CUDA engines (FusedHash and SortMerge).
//
//  * memory: cudaMallocAsync on the default stream, with the pool's release threshold set to
//    "never", so that freed blocks are reused by later allocations, and its high watermark
//    gives the peak GPU memory of a run; an allocation failure prints "OOM" and exits
//    with code 3, which the harness interprets as out of memory;
//  * input: binary files of int32 rows (2 columns for graphs, 3 for weighted graphs and
//    shareholdings), written by data/prepare.py;
//  * output: one line "RESULT {json}" on stdout, parsed by bench/harness.py.
#pragma once
#include <cuda_runtime.h>
#include <unistd.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#define CK(call)                                                                              \
    do {                                                                                      \
        cudaError_t e_ = (call);                                                              \
        if (e_ != cudaSuccess) fail_cuda(e_, #call, __FILE__, __LINE__);                      \
    } while (0)

[[noreturn]] inline void fail_cuda(cudaError_t e, const char* what, const char* file, int line) {
    if (e == cudaErrorMemoryAllocation) {
        printf("OOM\n");
        fprintf(stderr, "out of GPU memory at %s:%d (%s)\n", file, line, what);
        fflush(stdout); fflush(stderr);
        _exit(3);
    }
    fprintf(stderr, "CUDA error '%s' at %s:%d (%s)\n", cudaGetErrorString(e), file, line, what);
    fflush(stdout); fflush(stderr);
    _exit(2);
}

inline void check_launch(const char* name) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) fail_cuda(e, name, __FILE__, __LINE__);
}

// ------------------------------------------------------------------ memory

inline void init_pool() {
    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaMemPool_t pool;
    CK(cudaDeviceGetDefaultMemPool(&pool, dev));
    uint64_t never = UINT64_MAX;
    CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &never));
}

inline void trim_pool() {                      // gives cached blocks back (between partitions)
    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaMemPool_t pool;
    CK(cudaDeviceGetDefaultMemPool(&pool, dev));
    CK(cudaDeviceSynchronize());
    CK(cudaMemPoolTrimTo(pool, 0));
}

// peak of GPU memory: high watermarks of the default pool (every allocation of the engines goes
// through it), used = allocated and not freed, reserved = taken from the device by the pool
inline cudaMemPool_t default_pool() {
    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaMemPool_t pool;
    CK(cudaDeviceGetDefaultMemPool(&pool, dev));
    return pool;
}

inline void reset_peak() {
    CK(cudaDeviceSynchronize());
    uint64_t zero = 0;
    CK(cudaMemPoolSetAttribute(default_pool(), cudaMemPoolAttrUsedMemHigh, &zero));
    CK(cudaMemPoolSetAttribute(default_pool(), cudaMemPoolAttrReservedMemHigh, &zero));
}

inline uint64_t pool_peak(bool used) {
    CK(cudaDeviceSynchronize());
    uint64_t v = 0;
    CK(cudaMemPoolGetAttribute(default_pool(), used ? cudaMemPoolAttrUsedMemHigh : cudaMemPoolAttrReservedMemHigh, &v));
    return v;
}

template <class T>
struct DBuf {                                  // device array owned by the host
    T* p = nullptr;
    size_t n = 0;
    DBuf() = default;
    explicit DBuf(size_t k) { alloc(k); }
    DBuf(const DBuf&) = delete;
    DBuf& operator=(const DBuf&) = delete;
    DBuf(DBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    DBuf& operator=(DBuf&& o) noexcept {
        if (this != &o) { release(); p = o.p; n = o.n; o.p = nullptr; o.n = 0; }
        return *this;
    }
    ~DBuf() { release(); }
    void alloc(size_t k) {
        release();
        if (k) CK(cudaMallocAsync((void**)&p, k * sizeof(T), 0));
        n = k;
    }
    void release() {
        if (p) CK(cudaFreeAsync(p, 0));
        p = nullptr;
        n = 0;
    }
    void zero() { if (n) CK(cudaMemsetAsync(p, 0, n * sizeof(T), 0)); }
    void fill_bytes(int v) { if (n) CK(cudaMemsetAsync(p, v, n * sizeof(T), 0)); }
    // grows to at least k elements, keeping the first `keep` elements
    void grow(size_t k, size_t keep) {
        if (k <= n) return;
        T* q = nullptr;
        CK(cudaMallocAsync((void**)&q, k * sizeof(T), 0));
        if (keep) CK(cudaMemcpyAsync(q, p, keep * sizeof(T), cudaMemcpyDeviceToDevice, 0));
        if (p) CK(cudaFreeAsync(p, 0));
        p = q;
        n = k;
    }
    T get(size_t i) const {
        T v;
        CK(cudaMemcpy(&v, p + i, sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
    void set(size_t i, T v) { CK(cudaMemcpy(p + i, &v, sizeof(T), cudaMemcpyHostToDevice)); }
};

template <class T>
DBuf<T> to_device(const std::vector<T>& h) {
    DBuf<T> d(h.size());
    if (!h.empty()) CK(cudaMemcpy(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// temporary storage of CUB, reused across calls
struct CubTemp {
    DBuf<uint8_t> buf;
    void* get(size_t bytes) {
        if (bytes > buf.n) buf.alloc(bytes + bytes / 4 + 1024);
        return buf.p;
    }
};
inline CubTemp& cub_temp() { static CubTemp t; return t; }

template <class T>
void exclusive_sum(const T* in, T* out, size_t n) {
    if (!n) return;
    size_t bytes = 0;
    CK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, n));
    void* tmp = cub_temp().get(bytes);
    CK(cub::DeviceScan::ExclusiveSum(tmp, bytes, in, out, n));
}

template <class T>
void inclusive_sum(const T* in, T* out, size_t n) {
    if (!n) return;
    size_t bytes = 0;
    CK(cub::DeviceScan::InclusiveSum(nullptr, bytes, in, out, n));
    void* tmp = cub_temp().get(bytes);
    CK(cub::DeviceScan::InclusiveSum(tmp, bytes, in, out, n));
}

template <class T>
T device_sum(const T* in, size_t n) {
    if (!n) return T(0);
    DBuf<T> out(1);
    size_t bytes = 0;
    CK(cub::DeviceReduce::Sum(nullptr, bytes, in, out.p, n));
    void* tmp = cub_temp().get(bytes);
    CK(cub::DeviceReduce::Sum(tmp, bytes, in, out.p, n));
    return out.get(0);
}

// CUB primitives take int item counts in several CUDA versions: beyond 2^31 - 1 items a
// materialized step is treated as running out of memory (exit code 3), never truncated
inline void check_items(uint64_t n, const char* what) {
    if (n > 0x7FFFFFFFull) {
        printf("OOM\n");
        fprintf(stderr, "%s: %llu items exceed the 2^31-1 limit of the CUB primitives\n", what,
                (unsigned long long)n);
        fflush(stdout); fflush(stderr);
        _exit(3);
    }
}

inline unsigned grid_for(uint64_t n, unsigned block = 256) {
    uint64_t g = (n + block - 1) / block;
    return (unsigned)(g ? g : 1);
}

inline int bits_for(uint64_t v) {              // number of bits to represent 0..v
    int b = 1;
    while (b < 64 && (v >> b)) ++b;
    return b;
}

// ------------------------------------------------------------------ hashing

__device__ __forceinline__ uint32_t hash32(uint32_t k) {       // lowbias32 (Wellons)
    k ^= k >> 16; k *= 0x7feb352dU; k ^= k >> 15; k *= 0x846ca68bU; k ^= k >> 16;
    return k;
}

__device__ __forceinline__ uint64_t hash64(uint64_t k) {       // splitmix64 finalizer
    k ^= k >> 30; k *= 0xbf58476d1ce4e5b9ULL; k ^= k >> 27; k *= 0x94d049bb133111ebULL; k ^= k >> 31;
    return k;
}

// ------------------------------------------------------------------ CSR index of a binary EDB

// off[k]..off[k+1] indexes the successors val[...] of k (and their weights w, if any).
// Built on the GPU: count degrees, exclusive scan, scatter with an atomic cursor per key
// (the order of the successors of a key is irrelevant to every program).
__global__ void k_degree(const uint32_t* key, uint64_t n, uint32_t* deg) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&deg[key[i]], 1u);
}

__global__ void k_scatter_csr(const uint32_t* key, const uint32_t* val, const uint32_t* w, uint64_t n,
                              uint32_t* cursor, uint32_t* oval, uint32_t* ow) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t p = atomicAdd(&cursor[key[i]], 1u);
    oval[p] = val[i];
    if (w) ow[p] = w[i];
}

struct CSR {
    uint32_t K = 0;
    DBuf<uint32_t> off, val, w;                // off has K+1 entries
    void build(const uint32_t* key, const uint32_t* v, const uint32_t* wt, uint64_t n, uint32_t K_) {
        K = K_;
        DBuf<uint32_t> deg(K + 1);
        deg.zero();
        if (n) { k_degree<<<grid_for(n), 256>>>(key, n, deg.p); check_launch("k_degree"); }
        off.alloc(K + 1);
        exclusive_sum(deg.p, off.p, (size_t)K + 1);
        DBuf<uint32_t> cursor(K + 1);
        CK(cudaMemcpyAsync(cursor.p, off.p, (K + 1) * sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));
        val.alloc(n);
        if (wt) w.alloc(n);
        if (n) {
            k_scatter_csr<<<grid_for(n), 256>>>(key, v, wt, n, cursor.p, val.p, wt ? w.p : nullptr);
            check_launch("k_scatter_csr");
        }
    }
};

// ------------------------------------------------------------------ input

struct Table {                                 // rows of int32 read from a .bin file
    size_t rows = 0;
    int cols = 0;
    std::vector<uint32_t> data;                // row-major
    std::vector<uint32_t> col(int c) const {
        std::vector<uint32_t> v(rows);
        for (size_t i = 0; i < rows; ++i) v[i] = data[i * cols + c];
        return v;
    }
    uint32_t max_of(int c) const {
        uint32_t m = 0;
        for (size_t i = 0; i < rows; ++i) m = data[i * cols + c] > m ? data[i * cols + c] : m;
        return m;
    }
};

inline Table read_bin(const std::string& path, int cols) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    Table t;
    t.cols = cols;
    t.rows = (size_t)bytes / (sizeof(uint32_t) * cols);
    t.data.resize(t.rows * cols);
    if (t.rows && fread(t.data.data(), sizeof(uint32_t), t.data.size(), f) != t.data.size()) {
        fprintf(stderr, "short read on %s\n", path.c_str());
        exit(1);
    }
    fclose(f);
    return t;
}

// ------------------------------------------------------------------ command line and output

struct Args {
    std::map<std::string, std::string> kv;
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0) { fprintf(stderr, "bad argument %s\n", a.c_str()); exit(1); }
            auto eq = a.find('=');
            if (eq != std::string::npos) kv[a.substr(2, eq - 2)] = a.substr(eq + 1);
            else if (i + 1 < argc) kv[a.substr(2)] = argv[++i];
            else kv[a.substr(2)] = "1";
        }
    }
    std::string str(const std::string& k, const std::string& d = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? d : it->second;
    }
    long long num(const std::string& k, long long d) const {
        auto it = kv.find(k);
        return it == kv.end() ? d : atoll(it->second.c_str());
    }
    double real(const std::string& k, double d) const {
        auto it = kv.find(k);
        return it == kv.end() ? d : atof(it->second.c_str());
    }
};

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline double sync_now() {
    CK(cudaDeviceSynchronize());
    return now_s();
}

struct Json {                                  // flat JSON object for the RESULT line
    std::string s = "{";
    bool first = true;
    Json& add(const char* k, const std::string& v, bool quote = true) {
        s += first ? "" : ", ";
        first = false;
        s += "\""; s += k; s += "\": ";
        if (quote) s += "\"";
        s += v;
        if (quote) s += "\"";
        return *this;
    }
    Json& num(const char* k, double v) { char b[64]; snprintf(b, sizeof b, "%.6f", v); return add(k, b, false); }
    Json& num(const char* k, unsigned long long v) { return add(k, std::to_string(v), false); }
    Json& num(const char* k, long long v) { return add(k, std::to_string(v), false); }
    Json& num(const char* k, int v) { return add(k, std::to_string(v), false); }
    void print() { s += "}"; printf("RESULT %s\n", s.c_str()); fflush(stdout); }
};

inline void print_device() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    fprintf(stderr, "device: %s, %d SMs, %.1f GiB\n", p.name, p.multiProcessorCount,
            p.totalGlobalMem / 1073741824.0);
}

// ------------------------------------------------------------------ fingerprint of a set of facts
// An order-independent fingerprint of a set of binary facts (x, y): the sum mod 2^64 of
// splitmix64(x << 32 | y). A missing, extra or different fact changes it (with probability
// 1 - 2^-64), which a count of the facts does not catch. Computed only with --fingerprint=1 (the
// correctness tests), outside the measured work of the benchmarks; tests/reference.py computes the
// same on the CPU.
__device__ __forceinline__ unsigned long long fp_mix(unsigned long long z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// facts (x[i], y[i]) for i < n; x == nullptr: x = i (a dense array of values, y[i] == 0xFFFFFFFF
// skipped: a vertex not reached)
__global__ void k_fingerprint(const uint32_t* x, const uint32_t* y, uint64_t n, unsigned long long* out) {
    unsigned long long s = 0;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t b = y[i];
        if (!x && b == 0xFFFFFFFFu) continue;
        uint32_t a = x ? x[i] : (uint32_t)i;
        s += fp_mix(((unsigned long long)a << 32) | b);
    }
    atomicAdd(out, s);
}

// facts packed as keys x * K + y
__global__ void k_fingerprint_keys(const unsigned long long* keys, uint64_t n, uint32_t K, unsigned long long* out) {
    unsigned long long s = 0;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        unsigned long long k = keys[i];
        s += fp_mix(((k / K) << 32) | (k % K));
    }
    atomicAdd(out, s);
}

inline unsigned long long fingerprint(const uint32_t* x, const uint32_t* y, uint64_t n) {
    DBuf<unsigned long long> out(1);
    out.zero();
    unsigned g = grid_for(n) < 4096 ? grid_for(n) : 4096;
    if (n) { k_fingerprint<<<g, 256>>>(x, y, n, out.p); check_launch("k_fingerprint"); }
    return out.get(0);
}

inline unsigned long long fingerprint_keys(const unsigned long long* keys, uint64_t n, uint32_t K) {
    DBuf<unsigned long long> out(1);
    out.zero();
    unsigned g = grid_for(n) < 4096 ? grid_for(n) : 4096;
    if (n) { k_fingerprint_keys<<<g, 256>>>(keys, n, K, out.p); check_launch("k_fingerprint_keys"); }
    return out.get(0);
}
