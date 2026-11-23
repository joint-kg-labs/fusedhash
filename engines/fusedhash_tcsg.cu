// FusedHash for TC and SG on NVIDIA GPUs.
//
// One semi-naive iteration = one fused kernel. Each thread takes a tuple of Delta (or one
// candidate, with --split output), enumerates its candidates through the CSR index of the EDB,
// inserts each candidate into the set that holds all of full, and appends it to full if the
// insertion succeeded. Delta of the next iteration is the window of full appended meanwhile.
//
//   TC   T(x,y) :- E(x,y).                 T(x,y) :- T(x,z), E(z,y).
//   SG   SG(x,y) :- E(p,x), E(p,y), x != y.  SG(x,y) :- E(a,x), SG(a,b), E(b,y) [, x != y if --neq 1]
//
// The set that holds full (chosen at run time, --bitmap auto):
//   hash32  key x*K+y in one 32-bit word (K <= 65535), CAS on the word;
//   hash64  key (x<<32)|y in one 64-bit word, native 64-bit CAS (more than 65,535 constants);
//   bitmap  one bit per pair (x,y): rows of W = ceil(K/32) words, 32W rows (the word-aligned
//           layout of the word-level join), atomicOr; selected when the hash table would grow to
//           at least the size of the bitmap (--bitmap auto), never (off) or from the start (force).
// Append to full (--append): one atomicAdd on the 64-bit counter per new fact (atomic), or one
// per warp through a ballot within the coalesced group of active threads (warp).
// Work distribution (--split): one thread per tuple of Delta (delta) or per candidate (output);
// defaults: delta for TC, output for SG (the paper uses output for both, with --loop 16384; see
// bench/fusedhash_final.py).
// Looping kernel (--loop L, 0 = off): when Delta and the last iteration are small, a single block
// of 1024 threads runs successive iterations, synchronized with __syncthreads (which also makes
// the global writes of the block visible to the block), and returns to the host only at the
// fixpoint, when an iteration would produce more than L candidates, when full or the hash table
// would exceed their capacity, or after --loop-maxit iterations.
// Word-level join (--words c, 0 = off): in bitmap mode, an iteration whose tuple-level join would
// produce more than c times the words read by the word-level join runs on whole rows of the bit
// matrix: OR of the rows of the neighbours (32 pairs per operation, no atomics) and 32x32 block
// transposes through warp ballots. TC: T_new = T(rowor(T(Delta), pred)); SG: T(rowor(T(rowor(
// Delta, par)), par)). The new pairs (candidates AND NOT full) are appended to full, so Delta
// stays a window of full. Needs three working bitmaps (--words-max-gib bounds their memory).
// Plans (--plan P, TC only): the seed facts T(x,y) :- E(x,y) are split by x mod P and each
// partition is a separate fixpoint with its own set and full (homomorphic decomposability).
// The RESULT line reports the peak of GPU memory (high watermarks of the memory pool: used and
// reserved) and the facts of the largest partition.
//
// Usage: fusedhash_tcsg --prog tc|sg --input g.bin [--base b.bin] [--plan 1] [--bitmap auto|off|force]
//        --base (TC only): the seed facts come from b.bin instead of g.bin (CSDA)
//        [--append atomic|warp] [--split delta|output] [--neq 0|1] [--reps 2] [--verbose 0]
//        [--chunk 268435456] [--bitmap-max-gib 24] [--loop 262144] [--loop-maxit 4096]
//        [--words 1] [--words-max-gib 16]
#include "common.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

enum Mode { HASH32 = 0, HASH64 = 1, BITS = 2 };
static const char* mode_name(int m) { return m == HASH32 ? "hash32" : m == HASH64 ? "hash64" : "bitmap"; }

constexpr uint32_t EMPTY32 = 0xFFFFFFFFu;
constexpr unsigned long long EMPTY64 = 0xFFFFFFFFFFFFFFFFull;

struct SetView {
    uint32_t* t32;
    unsigned long long* t64;
    uint32_t* bits;
    uint64_t mask;
    uint64_t row;                                          // bits per row of the bitmap: 32 * W
    uint32_t K;
    int mode;
};

struct FullView {
    uint32_t* fx;
    uint32_t* fy;
    unsigned long long* size;
};

// true if (x,y) was not in the set and this thread inserted it
__device__ __forceinline__ bool insert(const SetView& s, uint32_t x, uint32_t y) {
    if (s.mode == BITS) {
        uint64_t b = (uint64_t)x * s.row + y;              // word-aligned rows (word-level join)
        uint32_t bit = 1u << (uint32_t)(b & 31);
        return (atomicOr(&s.bits[b >> 5], bit) & bit) == 0;
    }
    if (s.mode == HASH32) {
        uint32_t key = x * s.K + y;
        uint64_t h = hash32(key) & s.mask;
        while (true) {
            uint32_t prev = atomicCAS(&s.t32[h], EMPTY32, key);
            if (prev == EMPTY32) return true;
            if (prev == key) return false;
            h = (h + 1) & s.mask;                          // linear probing
        }
    }
    unsigned long long key = ((unsigned long long)x << 32) | y;
    uint64_t h = hash64(key) & s.mask;
    while (true) {
        unsigned long long prev = atomicCAS(&s.t64[h], EMPTY64, key);
        if (prev == EMPTY64) return true;
        if (prev == key) return false;
        h = (h + 1) & s.mask;
    }
}

template <int WARP>
__device__ __forceinline__ void append(const FullView& f, bool fresh, uint32_t x, uint32_t y) {
    if (WARP) {                                            // one atomic per group of active threads
        cg::coalesced_group g = cg::coalesced_threads();
        unsigned m = g.ballot(fresh);
        if (!m) return;
        unsigned long long base = 0;
        if (g.thread_rank() == 0) base = atomicAdd(f.size, (unsigned long long)__popc(m));
        base = g.shfl(base, 0);
        if (fresh) {
            unsigned rank = __popc(m & ((1u << g.thread_rank()) - 1u));
            f.fx[base + rank] = x;
            f.fy[base + rank] = y;
        }
    } else if (fresh) {
        unsigned long long p = atomicAdd(f.size, 1ull);
        f.fx[p] = x;
        f.fy[p] = y;
    }
}

template <int WARP>
__device__ __forceinline__ void emit(const SetView& s, const FullView& f, uint32_t x, uint32_t y) {
    bool fresh = insert(s, x, y);
    append<WARP>(f, fresh, x, y);
}

// ------------------------------------------------------------------ kernels

template <int WARP>
__global__ void k_seed(SetView s, FullView f, const uint32_t* ex, const uint32_t* ey, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) emit<WARP>(s, f, ex[i], ey[i]);
}

__global__ void k_rehash(SetView s, const uint32_t* fx, const uint32_t* fy, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) insert(s, fx[i], fy[i]);
}

// candidates of each tuple of Delta = full[d0, d0+nd)
__global__ void k_count_tc(const uint32_t* fy, uint64_t d0, uint64_t nd, const uint32_t* off,
                           unsigned long long* cnt) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= nd) return;
    uint32_t z = fy[d0 + t];
    cnt[t] = off[z + 1] - off[z];
}

__global__ void k_count_sg(const uint32_t* fx, const uint32_t* fy, uint64_t d0, uint64_t nd,
                           const uint32_t* off, unsigned long long* cnt) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= nd) return;
    uint32_t a = fx[d0 + t], b = fy[d0 + t];
    cnt[t] = (unsigned long long)(off[a + 1] - off[a]) * (off[b + 1] - off[b]);
}

// base rule of SG, candidates of each edge (p, x): the children of p
__global__ void k_count_sgbase(const uint32_t* ep, uint64_t n, const uint32_t* off, unsigned long long* cnt) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint32_t p = ep[t];
    cnt[t] = off[p + 1] - off[p];
}

// one thread per tuple of Delta[a, b) (positions relative to d0)
template <int WARP>
__global__ void k_tc_delta(SetView s, FullView f, const uint32_t* off, const uint32_t* adj,
                           uint64_t d0, uint64_t a, uint64_t b) {
    uint64_t t = a + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= b) return;
    uint32_t x = f.fx[d0 + t], z = f.fy[d0 + t];
    for (uint32_t j = off[z], e = off[z + 1]; j < e; ++j) emit<WARP>(s, f, x, adj[j]);
}

template <int WARP>
__global__ void k_sg_delta(SetView s, FullView f, const uint32_t* off, const uint32_t* adj,
                           uint64_t d0, uint64_t a, uint64_t b, int neq) {
    uint64_t t = a + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= b) return;
    uint32_t pa = f.fx[d0 + t], pb = f.fy[d0 + t];
    uint32_t ib = off[pb], eb = off[pb + 1];
    for (uint32_t i = off[pa], ea = off[pa + 1]; i < ea; ++i) {
        uint32_t x = adj[i];
        for (uint32_t j = ib; j < eb; ++j) {
            uint32_t y = adj[j];
            if (!neq || x != y) emit<WARP>(s, f, x, y);
        }
    }
}

template <int WARP>
__global__ void k_sgbase_delta(SetView s, FullView f, const uint32_t* off, const uint32_t* adj,
                               const uint32_t* ep, const uint32_t* ex, uint64_t a, uint64_t b) {
    uint64_t t = a + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= b) return;
    uint32_t p = ep[t], x = ex[t];
    for (uint32_t j = off[p], e = off[p + 1]; j < e; ++j) {
        uint32_t y = adj[j];
        if (y != x) emit<WARP>(s, f, x, y);
    }
}

// one thread per candidate: incl = inclusive scan of the counts of Delta; the candidates of this
// launch are the global numbers [base, base+total); tuple j owns [excl(j), incl[j])
__device__ __forceinline__ uint64_t owner(const unsigned long long* incl, uint64_t a, uint64_t b, uint64_t g) {
    uint64_t lo = a, hi = b;                               // smallest j in [a,b) with incl[j] > g
    while (lo < hi) {
        uint64_t m = (lo + hi) >> 1;
        if (incl[m] > g) hi = m; else lo = m + 1;
    }
    return lo;
}

template <int WARP>
__global__ void k_tc_output(SetView s, FullView f, const uint32_t* off, const uint32_t* adj,
                            const unsigned long long* incl, uint64_t d0, uint64_t a, uint64_t b,
                            uint64_t base, uint64_t total) {
    uint64_t n = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (n >= total) return;
    uint64_t g = base + n;
    uint64_t j = owner(incl, a, b, g);
    uint64_t ex = j ? incl[j - 1] : 0;
    uint32_t x = f.fx[d0 + j], z = f.fy[d0 + j];
    emit<WARP>(s, f, x, adj[off[z] + (uint32_t)(g - ex)]);
}

template <int WARP>
__global__ void k_sg_output(SetView s, FullView f, const uint32_t* off, const uint32_t* adj,
                            const unsigned long long* incl, uint64_t d0, uint64_t a, uint64_t b,
                            uint64_t base, uint64_t total, int neq) {
    uint64_t n = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (n >= total) return;
    uint64_t g = base + n;
    uint64_t j = owner(incl, a, b, g);
    uint64_t ex = j ? incl[j - 1] : 0;
    uint32_t pa = f.fx[d0 + j], pb = f.fy[d0 + j];
    uint64_t r = g - ex, db = off[pb + 1] - off[pb];
    uint32_t x = adj[off[pa] + (uint32_t)(r / db)], y = adj[off[pb] + (uint32_t)(r % db)];
    if (!neq || x != y) emit<WARP>(s, f, x, y);
}

// cut points of the chunks: cut[k] = first j with incl[j] > (k+1)*chunk
__global__ void k_cuts(const unsigned long long* incl, uint64_t nd, uint64_t chunk, uint64_t ncuts,
                       unsigned long long* cut) {
    uint64_t k = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (k >= ncuts) return;
    cut[k] = owner(incl, 0, nd, (k + 1) * chunk);
}

// ------------------------------------------------------------------ looping kernel (small Delta)
// One block runs successive iterations over Delta = full[d0, F). ctl: [d0, F] in input, [d0, F,
// iterations run] in output; log: (|Delta|, candidates, new facts) per iteration. The block stops at
// the fixpoint, when an iteration would produce more than maxwork candidates (the whole GPU is
// better), when full would exceed limit (capacity of full and, for the hash table, its load), or
// after maxit iterations. __syncthreads makes the appends of an iteration visible to the block.
constexpr int LOOP_THREADS = 1024;

__device__ __forceinline__ unsigned long long block_sum(unsigned long long v, unsigned long long* red) {
    unsigned lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, d);
    if (lane == 0) red[wid] = v;
    __syncthreads();
    unsigned long long t = 0;
    for (unsigned i = 0; i < nw; ++i) t += red[i];
    __syncthreads();
    return t;
}

template <int WARP, int SG>
__global__ void __launch_bounds__(LOOP_THREADS)
k_loop(SetView s, FullView f, const uint32_t* off, const uint32_t* adj, unsigned long long* ctl,
       unsigned long long* log, unsigned long long maxwork, unsigned long long limit, int maxit, int neq) {
    __shared__ unsigned long long red[32];
    __shared__ unsigned long long shF;
    uint64_t d0 = ctl[0], F = ctl[1];
    int it = 0;
    while (it < maxit && F > d0) {
        unsigned long long w = 0;
        for (uint64_t i = d0 + threadIdx.x; i < F; i += blockDim.x) {
            unsigned long long work;
            if (SG) {
                uint32_t a = f.fx[i], b = f.fy[i];
                work = (unsigned long long)(off[a + 1] - off[a]) * (off[b + 1] - off[b]);
            } else {
                uint32_t z = f.fy[i];
                work = off[z + 1] - off[z];
            }
            w = min(w + work, maxwork + 1);
        }
        unsigned long long tot = block_sum(w, red);
        if (tot > maxwork || F + tot > limit) break;       // uniform: every thread has the same tot
        for (uint64_t i = d0 + threadIdx.x; i < F; i += blockDim.x) {
            if (SG) {
                uint32_t a = f.fx[i], b = f.fy[i];
                uint32_t ib = off[b], eb = off[b + 1];
                for (uint32_t p = off[a], ea = off[a + 1]; p < ea; ++p) {
                    uint32_t x = adj[p];
                    for (uint32_t j = ib; j < eb; ++j) {
                        uint32_t y = adj[j];
                        if (!neq || x != y) emit<WARP>(s, f, x, y);
                    }
                }
            } else {
                uint32_t x = f.fx[i], z = f.fy[i];
                for (uint32_t j = off[z], e = off[z + 1]; j < e; ++j) emit<WARP>(s, f, x, adj[j]);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) shF = *(volatile unsigned long long*)f.size;
        __syncthreads();
        uint64_t nF = shF;
        if (threadIdx.x == 0) { log[3 * it] = F - d0; log[3 * it + 1] = tot; log[3 * it + 2] = nF - F; }
        d0 = F;
        F = nF;
        ++it;
    }
    if (threadIdx.x == 0) { ctl[0] = d0; ctl[1] = F; ctl[2] = (unsigned long long)it; }
}

// ------------------------------------------------------------------ word-level join (dense Delta)
// Relations as bit matrices with rows of W words (32W rows); n = 32 W^2 words. Grids are whole
// warps (multiples of 256 threads) and no thread returns before the warp-wide operations.

// 32x32 block transpose, one warp per block: block (bi, bj) (rows 32bi.., word bj) -> (bj, bi)
__global__ void k_bits_transpose(const uint32_t* in, uint32_t* out, uint64_t W) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, t = tid >> 5;
    unsigned lane = threadIdx.x & 31;
    if (t >= W * W) return;                                // uniform within the warp
    uint64_t bi = t / W, bj = t % W;
    uint32_t v = in[(32 * bi + lane) * W + bj], r = 0;
    for (unsigned k = 0; k < 32; ++k) {
        uint32_t m = __ballot_sync(0xFFFFFFFFu, (v >> k) & 1u);
        if (lane == k) r = m;
    }
    out[(32 * bj + lane) * W + bi] = r;
}

// one thread per word (r, w): OR of the rows of the neighbours of r (rows >= K: no neighbours)
__global__ void k_bits_rowor(const uint32_t* in, uint32_t* out, const uint32_t* off, const uint32_t* nbr,
                             uint64_t W, uint32_t K, uint64_t n) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t r = tid / W, w = tid % W;
    uint32_t acc = 0;
    if (r < K)
        for (uint32_t j = off[r], e = off[r + 1]; j < e; ++j) acc |= in[(uint64_t)nbr[j] * W + w];
    out[tid] = acc;
}

// cand = cand AND NOT full (and without the diagonal if neq); counts the new pairs
__global__ void k_bits_count(uint32_t* cand, const uint32_t* full, unsigned long long* cnt, uint64_t W,
                             int neq, uint64_t n) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    unsigned c = 0;
    if (tid < n) {
        uint32_t v = cand[tid] & ~full[tid];
        uint64_t r = tid / W;
        if (neq && (r >> 5) == tid % W) v &= ~(1u << (r & 31));
        cand[tid] = v;
        c = __popc(v);
    }
    c = __reduce_add_sync(0xFFFFFFFFu, c);
    if ((threadIdx.x & 31) == 0 && c) atomicAdd(cnt, (unsigned long long)c);
}

// full |= cand, Delta = cand, and the new pairs at the end of the list of full (one atomic per warp)
__global__ void k_bits_append(const uint32_t* cand, uint32_t* full, uint32_t* delta, uint32_t* fx, uint32_t* fy,
                              unsigned long long* size, uint64_t W, uint64_t n) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    unsigned lane = threadIdx.x & 31;
    uint32_t v = tid < n ? cand[tid] : 0u;
    if (tid < n) { delta[tid] = v; if (v) full[tid] |= v; }
    unsigned c = __popc(v), inc = c;
    for (int d = 1; d < 32; d <<= 1) {
        unsigned t = __shfl_up_sync(0xFFFFFFFFu, inc, d);
        if (lane >= (unsigned)d) inc += t;
    }
    unsigned tot = __shfl_sync(0xFFFFFFFFu, inc, 31);
    unsigned long long base = 0;
    if (lane == 0 && tot) base = atomicAdd(size, (unsigned long long)tot);
    base = __shfl_sync(0xFFFFFFFFu, base, 0) + (inc - c);
    uint32_t x = (uint32_t)(tid / W), y0 = (uint32_t)(tid % W) << 5;
    while (v) {
        uint32_t b = __ffs(v) - 1;
        fx[base] = x;
        fy[base] = y0 + b;
        ++base;
        v &= v - 1;
    }
}

// Delta = full[d0, d0+n) as a bitmap (delta zeroed before)
__global__ void k_bits_scatter(uint32_t* delta, const uint32_t* fx, const uint32_t* fy, uint64_t W, uint64_t d0,
                               uint64_t n) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint32_t x = fx[d0 + tid], y = fy[d0 + tid];
    atomicOr(&delta[(uint64_t)x * W + (y >> 5)], 1u << (y & 31));
}

// ------------------------------------------------------------------ store: full + set

struct Options {
    std::string bitmap = "auto", append = "atomic", split;
    int neq = 0, verbose = 0;
    uint64_t chunk = 1ull << 28;
    double load = 0.5;
    uint64_t bitmap_max_bytes = 24ull << 30;
    uint64_t loop = 262144;                                // looping kernel: max candidates per iteration
    int loop_maxit = 4096;                                 // iterations per launch (size of the log)
    double words = 1.0;                                    // word-level join: candidates / words threshold
    uint64_t words_max_bytes = 16ull << 30;                // memory of the three working bitmaps
    int fingerprint = 0;                                   // the fingerprint of the facts (tests)
};

struct Store {
    uint32_t K;
    int mode, wide;
    const Options& o;
    DBuf<uint32_t> fx, fy, t32, bits;
    DBuf<unsigned long long> t64, size;
    uint64_t cap_tab = 0, F = 0;
    uint64_t W, bits_words;                                // bitmap: 32W rows of W words
    int rehashes = 0;

    Store(uint32_t K_, uint64_t n0, const Options& o_) : K(K_), o(o_) {
        wide = K > 65535;
        mode = wide ? HASH64 : HASH32;
        W = ((uint64_t)K + 31) / 32;
        bits_words = 32 * W * W;
        bool bitmap_ok = o.bitmap != "off" && bits_words * 4 <= o.bitmap_max_bytes;
        n0 = n0 ? n0 : 1;
        fx.alloc(pow2(4 * n0));
        fy.alloc(fx.n);
        cap_tab = pow2((uint64_t)(n0 / o.load) + 1);
        if (bitmap_ok && (o.bitmap == "force" || bits_words * 4 <= cap_tab * slot_bytes())) mode = BITS;
        new_table();
        size.alloc(1);
        size.zero();
    }
    static uint64_t pow2(uint64_t n) { uint64_t p = 1024; while (p < n) p <<= 1; return p; }
    uint64_t slot_bytes() const { return wide ? 8 : 4; }
    uint64_t set_bytes() const { return mode == BITS ? bits_words * 4 : cap_tab * slot_bytes(); }
    void new_table() {
        t32.release(); t64.release(); bits.release();
        if (mode == BITS) { bits.alloc(bits_words); bits.zero(); }
        else if (mode == HASH32) { t32.alloc(cap_tab); t32.fill_bytes(0xFF); }
        else { t64.alloc(cap_tab); t64.fill_bytes(0xFF); }
    }
    SetView set() const { return SetView{t32.p, t64.p, bits.p, cap_tab - 1, 32 * W, K, mode}; }
    FullView full() const { return FullView{fx.p, fy.p, size.p}; }
    uint64_t sync_size() { F = size.get(0); return F; }
    // largest |full| reachable without growing: capacity of full and, for the hash table, its load
    uint64_t limit() const {
        if (mode == BITS) return fx.n;
        uint64_t t = (uint64_t)(o.load * cap_tab);
        return t < fx.n ? t : fx.n;
    }
    // room for F + extra facts in full and in the set
    void ensure(uint64_t extra) {
        uint64_t need = F + extra;
        if (need > fx.n) { uint64_t c = pow2(need); fx.grow(c, F); fy.grow(c, F); }
        if (mode == BITS) return;
        if (need > o.load * cap_tab) {
            cap_tab = pow2((uint64_t)(need / o.load) + 1);
            bool bitmap_ok = o.bitmap == "auto" && bits_words * 4 <= o.bitmap_max_bytes;
            if (bitmap_ok && bits_words * 4 <= cap_tab * slot_bytes()) mode = BITS;
            new_table();
            ++rehashes;
            if (F) { k_rehash<<<grid_for(F), 256>>>(set(), fx.p, fy.p, F); check_launch("k_rehash"); }
        }
    }
};

// ------------------------------------------------------------------ semi-naive loop

struct Stats {
    int iters = 0, loop_iters = 0, word_iters = 0;
    uint64_t candidates = 0;
};

// CSR built at its first use: the index of the word-level join (predecessors for TC, parents for
// SG) costs nothing in the computations where the word-level join never runs
struct LazyCSR {
    const uint32_t *key, *val;
    uint64_t n;
    uint32_t K;
    CSR csr;
    bool built = false;
    CSR& get() {
        if (!built) { csr.build(key, val, nullptr, n, K); built = true; }
        return csr;
    }
};

// word-level join of one store: passes = sequence of transposes (nullptr) and row ORs over a CSR
// that maps the bitmap of Delta to the bitmap of the candidates; keeps the bitmap of Delta (valid
// while consecutive iterations stay word-level) and two working bitmaps
struct WordJoin {
    Store& st;
    std::vector<LazyCSR*> passes;
    int neq;
    double threshold;
    DBuf<uint32_t> D, A, B;
    bool valid = false, ok = false;
    double cost = 0;
    WordJoin(Store& st_, std::vector<LazyCSR*> p, int neq_, const Options& o) : st(st_), passes(p), neq(neq_),
                                                                              threshold(o.words) {
        uint64_t n = st.bits_words, nnz = 0;
        int nt = 0;
        for (auto c : passes) { if (c) nnz += c->n; else ++nt; }
        cost = (double)nnz * st.W + (double)(2 * nt + 3) * n;          // words read and written
        ok = o.words > 0 && o.bitmap != "off" && 12 * n <= o.words_max_bytes;
    }
    bool use(uint64_t R) const { return ok && st.mode == BITS && (double)R > threshold * cost; }
    void step(uint64_t d0, uint64_t F) {
        uint64_t n = st.bits_words, W = st.W;
        if (!D.p) { D.alloc(n); A.alloc(n); B.alloc(n); }
        if (!valid) {                                      // Delta = full[d0, F) as a bitmap
            D.zero();
            k_bits_scatter<<<grid_for(F - d0), 256>>>(D.p, st.fx.p, st.fy.p, W, d0, F - d0);
            check_launch("k_bits_scatter");
        }
        uint32_t* cur = D.p;
        for (auto c : passes) {
            uint32_t* out = cur != A.p ? A.p : B.p;
            if (!c) k_bits_transpose<<<grid_for(n), 256>>>(cur, out, W);
            else {
                CSR& g = c->get();
                k_bits_rowor<<<grid_for(n), 256>>>(cur, out, g.off.p, g.val.p, W, st.K, n);
            }
            check_launch("k_bits_pass");
            cur = out;
        }
        DBuf<unsigned long long> cnt(1);
        cnt.zero();
        k_bits_count<<<grid_for(n), 256>>>(cur, st.bits.p, cnt.p, W, neq, n);
        check_launch("k_bits_count");
        st.ensure(cnt.get(0));                             // only the list: the bitmap covers the domain
        k_bits_append<<<grid_for(n), 256>>>(cur, st.bits.p, D.p, st.fx.p, st.fy.p, st.size.p, W, n);
        check_launch("k_bits_append");
        st.sync_size();
        valid = true;
    }
};

template <int WARP>
struct Engine {
    const Options& o;
    const CSR& E;
    std::string prog, split;

    // counts the candidates of each tuple of Delta = full[d0, d0+nd) (inclusive scan in incl); returns R
    uint64_t count(Store& st, uint64_t d0, uint64_t nd, DBuf<unsigned long long>& incl) {
        check_items(nd, "delta");
        DBuf<unsigned long long> cnt(nd);
        incl.alloc(nd);
        if (prog == "tc") k_count_tc<<<grid_for(nd), 256>>>(st.fy.p, d0, nd, E.off.p, cnt.p);
        else k_count_sg<<<grid_for(nd), 256>>>(st.fx.p, st.fy.p, d0, nd, E.off.p, cnt.p);
        check_launch("k_count");
        inclusive_sum(cnt.p, incl.p, nd);
        return incl.get(nd - 1);
    }

    // runs the kernel of one iteration over Delta = full[d0, d0+nd), in chunks of <= o.chunk candidates
    void iteration(Store& st, uint64_t d0, uint64_t nd, const DBuf<unsigned long long>& incl, uint64_t R) {
        std::vector<uint64_t> cuts{0};
        if (R > o.chunk) {
            uint64_t nc = (R - 1) / o.chunk;
            DBuf<unsigned long long> dcut(nc);
            k_cuts<<<grid_for(nc), 256>>>(incl.p, nd, o.chunk, nc, dcut.p);
            check_launch("k_cuts");
            std::vector<unsigned long long> h(nc);
            CK(cudaMemcpy(h.data(), dcut.p, nc * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            for (auto c : h) if (c > cuts.back() && c < nd) cuts.push_back(c);
        }
        cuts.push_back(nd);
        for (size_t k = 0; k + 1 < cuts.size(); ++k) {
            uint64_t a = cuts[k], b = cuts[k + 1];
            uint64_t base = a ? incl.get(a - 1) : 0, r = incl.get(b - 1) - base;
            if (r == 0) continue;
            st.ensure(r);
            SetView sv = st.set();
            FullView fv = st.full();
            if (split == "output") {
                if (prog == "tc") k_tc_output<WARP><<<grid_for(r), 256>>>(sv, fv, E.off.p, E.val.p, incl.p, d0, a, b, base, r);
                else k_sg_output<WARP><<<grid_for(r), 256>>>(sv, fv, E.off.p, E.val.p, incl.p, d0, a, b, base, r, o.neq);
            } else {
                if (prog == "tc") k_tc_delta<WARP><<<grid_for(b - a), 256>>>(sv, fv, E.off.p, E.val.p, d0, a, b);
                else k_sg_delta<WARP><<<grid_for(b - a), 256>>>(sv, fv, E.off.p, E.val.p, d0, a, b, o.neq);
            }
            check_launch("k_step");
            st.sync_size();
        }
    }

    // looping kernel from Delta = full[d0, F): returns the iterations it ran (0 = none) and updates d0
    int loop(Store& st, uint64_t& d0, Stats& stats, DBuf<unsigned long long>& ctl, DBuf<unsigned long long>& log) {
        st.ensure(o.loop);                                 // headroom: the kernel grows nothing
        unsigned long long h[3] = {d0, st.F, 0};
        CK(cudaMemcpy(ctl.p, h, sizeof h, cudaMemcpyHostToDevice));
        bool sg = prog == "sg";
        auto kern = sg ? k_loop<WARP, 1> : k_loop<WARP, 0>;
        kern<<<1, LOOP_THREADS>>>(st.set(), st.full(), E.off.p, E.val.p, ctl.p, log.p, o.loop, st.limit(),
                                  o.loop_maxit, o.neq);
        check_launch("k_loop");
        CK(cudaMemcpy(h, ctl.p, sizeof h, cudaMemcpyDeviceToHost));
        int n = (int)h[2];
        if (n) {
            std::vector<unsigned long long> rows(3 * n);
            CK(cudaMemcpy(rows.data(), log.p, rows.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) stats.candidates += rows[3 * i + 1];
            stats.iters += n;
            stats.loop_iters += n;
            d0 = h[0];
            st.F = h[1];
        }
        return n;
    }

    // fixpoint from the facts already in the store (full[0, F) is the first Delta)
    void fixpoint(Store& st, Stats& stats, WordJoin* word) {
        uint64_t d0 = 0, last_R = 0;
        bool retry = true;
        DBuf<unsigned long long> ctl, log;
        if (o.loop) { ctl.alloc(3); log.alloc(3 * (size_t)o.loop_maxit); }
        while (st.F > d0) {
            if (o.loop && retry && st.F - d0 <= o.loop && last_R <= 2 * o.loop) {
                double ta = now_s();
                int n = loop(st, d0, stats, ctl, log);
                if (word) word->valid = false;
                if (n) {
                    if (o.verbose)
                        fprintf(stderr, "  it %4d-%d: %d iterations in one kernel, |full|=%llu %.4fs\n",
                                stats.iters - n + 1, stats.iters, n, (unsigned long long)st.F, now_s() - ta);
                    retry = n == o.loop_maxit;             // stopped by work or space: one normal round
                    continue;
                }
            }
            retry = true;
            double ta = now_s();
            uint64_t F0 = st.F, nd = F0 - d0;
            ++stats.iters;
            DBuf<unsigned long long> incl;
            uint64_t R = count(st, d0, nd, incl);
            stats.candidates += R;
            bool words = word && word->use(R);
            if (words) {
                incl.release();
                word->step(d0, F0);
                ++stats.word_iters;
            } else {
                if (word) word->valid = false;
                iteration(st, d0, nd, incl, R);
            }
            d0 = F0;
            last_R = R;
            if (o.verbose)
                fprintf(stderr, "  it %4d: |delta|=%12llu candidates=%14llu new=%12llu set=%s%s %.4fs\n", stats.iters,
                        (unsigned long long)nd, (unsigned long long)R, (unsigned long long)(st.F - d0),
                        mode_name(st.mode), words ? " words" : "", now_s() - ta);
        }
    }

    // SG base rule: SG(x,y) :- E(p,x), E(p,y), x != y, chunked over the edges (p, x)
    void sg_base(Store& st, const DBuf<uint32_t>& ep, const DBuf<uint32_t>& ex, uint64_t m) {
        if (!m) return;
        check_items(m, "edges");
        DBuf<unsigned long long> cnt(m), incl(m);
        k_count_sgbase<<<grid_for(m), 256>>>(ep.p, m, E.off.p, cnt.p);
        check_launch("k_count_sgbase");
        inclusive_sum(cnt.p, incl.p, m);
        uint64_t R = incl.get(m - 1);
        std::vector<uint64_t> cuts{0};
        if (R > o.chunk) {
            uint64_t nc = (R - 1) / o.chunk;
            DBuf<unsigned long long> dcut(nc);
            k_cuts<<<grid_for(nc), 256>>>(incl.p, m, o.chunk, nc, dcut.p);
            check_launch("k_cuts");
            std::vector<unsigned long long> h(nc);
            CK(cudaMemcpy(h.data(), dcut.p, nc * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            for (auto c : h) if (c > cuts.back() && c < m) cuts.push_back(c);
        }
        cuts.push_back(m);
        for (size_t k = 0; k + 1 < cuts.size(); ++k) {
            uint64_t a = cuts[k], b = cuts[k + 1];
            uint64_t r = incl.get(b - 1) - (a ? incl.get(a - 1) : 0);
            if (!r) continue;
            st.ensure(r);
            k_sgbase_delta<WARP><<<grid_for(b - a), 256>>>(st.set(), st.full(), E.off.p, E.val.p, ep.p, ex.p, a, b);
            check_launch("k_sgbase");
            st.sync_size();
        }
    }
};

// ------------------------------------------------------------------ programs

struct RunResult {
    uint64_t facts = 0, largest = 0;                       // largest = facts of the largest partition
    int iters = 0, mode = 0, loop_iters = 0, word_iters = 0;
    uint64_t set_bytes = 0, candidates = 0;
    unsigned long long fp = 0;                             // fingerprint of the facts (--fingerprint=1)
    void add(const Store& st, const Stats& stats) {
        facts += st.F;
        largest = st.F > largest ? st.F : largest;
        iters = stats.iters > iters ? stats.iters : iters;
        loop_iters += stats.loop_iters;
        word_iters += stats.word_iters;
        candidates += stats.candidates;
        mode = st.mode > mode ? st.mode : mode;
        set_bytes = st.set_bytes() > set_bytes ? st.set_bytes() : set_bytes;
    }
};

template <int WARP>
RunResult run_tc(const Table& g, uint32_t K, int P, const Options& o, const std::string& split,
                 const Table* base = nullptr) {
    // base: the seed facts T(x,y) :- B(x,y) of another relation (CSDA: NullNode(x,y) :- NullEdge(x,y),
    // NullNode(x,y) :- NullNode(x,w), Edge(w,y)); without it the seed is the edges themselves (TC)
    std::vector<uint32_t> src = g.col(0), dst = g.col(1);
    std::vector<uint32_t> bsrc = base ? base->col(0) : src, bdst = base ? base->col(1) : dst;
    DBuf<uint32_t> dsrc = to_device(src), ddst = to_device(dst);
    CSR E;
    E.build(dsrc.p, ddst.p, nullptr, g.rows, K);
    LazyCSR pred{ddst.p, dsrc.p, g.rows, K};               // predecessors of each y: T_new^T = OR_{z in pred(y)} Delta^T[z]
    Engine<WARP> eng{o, E, "tc", split};
    RunResult res;
    for (int p = 0; p < P; ++p) {
        std::vector<uint32_t> sx, sy;                      // seed of partition p: x mod P == p
        if (P == 1) { sx = bsrc; sy = bdst; }
        else for (size_t i = 0; i < bsrc.size(); ++i) if (bsrc[i] % P == (uint32_t)p) { sx.push_back(bsrc[i]); sy.push_back(bdst[i]); }
        if (sx.empty()) continue;
        Store st(K, sx.size(), o);
        {
            DBuf<uint32_t> dx = to_device(sx), dy = to_device(sy);
            st.ensure(sx.size());
            if (WARP) k_seed<1><<<grid_for(sx.size()), 256>>>(st.set(), st.full(), dx.p, dy.p, sx.size());
            else k_seed<0><<<grid_for(sx.size()), 256>>>(st.set(), st.full(), dx.p, dy.p, sx.size());
            check_launch("k_seed");
            st.sync_size();
        }
        Stats stats;
        WordJoin word(st, {nullptr, &pred, nullptr}, 0, o);
        eng.fixpoint(st, stats, &word);
        res.add(st, stats);
        if (o.fingerprint) res.fp += fingerprint(st.fx.p, st.fy.p, st.F);   // a sum: additive over partitions
        if (o.verbose) fprintf(stderr, " partition %d: %llu facts\n", p, (unsigned long long)st.F);
    }
    return res;
}

template <int WARP>
RunResult run_sg(const Table& g, uint32_t K, const Options& o, const std::string& split) {
    std::vector<uint32_t> src = g.col(0), dst = g.col(1);  // edge (p, x): x is a child of p
    DBuf<uint32_t> dsrc = to_device(src), ddst = to_device(dst);
    CSR E;
    E.build(dsrc.p, ddst.p, nullptr, g.rows, K);
    LazyCSR par{ddst.p, dsrc.p, g.rows, K};                // parents of each x
    Engine<WARP> eng{o, E, "sg", split};
    Store st(K, g.rows, o);
    eng.sg_base(st, dsrc, ddst, g.rows);
    Stats stats;
    // M = OR_{a in par(x)} Delta[a];  S_new^T = OR_{b in par(y)} M^T[b]
    WordJoin word(st, {&par, nullptr, &par, nullptr}, o.neq, o);
    eng.fixpoint(st, stats, &word);
    RunResult res;
    res.add(st, stats);
    if (o.fingerprint) res.fp = fingerprint(st.fx.p, st.fy.p, st.F);
    return res;
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    std::string prog = a.str("prog", "tc"), input = a.str("input");
    if (input.empty()) { fprintf(stderr, "missing --input\n"); return 1; }
    Options o;
    o.bitmap = a.str("bitmap", "auto");
    o.append = a.str("append", "atomic");
    o.neq = (int)a.num("neq", 0);
    o.verbose = (int)a.num("verbose", 0);
    o.chunk = (uint64_t)a.num("chunk", 1ll << 28);
    o.load = a.real("load", 0.5);
    o.bitmap_max_bytes = (uint64_t)(a.real("bitmap-max-gib", 24) * 1073741824.0);
    o.loop = (uint64_t)a.num("loop", 262144);
    o.loop_maxit = (int)a.num("loop-maxit", 4096);
    o.words = a.real("words", 1.0);
    o.words_max_bytes = (uint64_t)(a.real("words-max-gib", 16) * 1073741824.0);
    o.fingerprint = (int)a.num("fingerprint", 0);
    std::string split = a.str("split", prog == "tc" ? "delta" : "output");
    int P = (int)a.num("plan", 1), reps = (int)a.num("reps", 2);
    if (prog == "sg" && P != 1) { fprintf(stderr, "SG is not decomposable: only --plan 1\n"); return 1; }
    int warp = o.append == "warp";

    print_device();
    init_pool();
    Table g = read_bin(input, 2);
    uint32_t K = std::max(g.max_of(0), g.max_of(1)) + 1;
    std::string base_path = a.str("base", "");
    if (!base_path.empty() && prog != "tc") { fprintf(stderr, "--base is for TC only\n"); return 1; }
    Table b;
    if (!base_path.empty()) {
        b = read_bin(base_path, 2);
        K = std::max(K, std::max(b.max_of(0), b.max_of(1)) + 1);
    }
    const Table* bp = base_path.empty() ? nullptr : &b;
    bool warming = true;                                   // the warm-up graph has no base

    auto run = [&](const Table& t, uint32_t k) {
        const Table* bb = warming ? nullptr : bp;
        if (prog == "tc") return warp ? run_tc<1>(t, k, P, o, split, bb) : run_tc<0>(t, k, P, o, split, bb);
        return warp ? run_sg<1>(t, k, o, split) : run_sg<0>(t, k, o, split);
    };
    {                                                      // warm-up: context, pool, kernels
        Table w;
        w.cols = 2;
        for (uint32_t i = 0; i + 1 < 2000; ++i) { w.data.push_back(i); w.data.push_back(i + 1); }
        w.rows = w.data.size() / 2;
        run(w, 2000);
        CK(cudaDeviceSynchronize());
    }
    warming = false;
    double best = 1e30;
    uint64_t peak_used = 0, peak_reserved = 0;             // of the best run
    RunResult r;
    for (int i = 0; i < reps; ++i) {
        reset_peak();
        double t0 = sync_now();
        RunResult ri = run(g, K);
        double t = sync_now() - t0;
        if (i && (ri.facts != r.facts || ri.fp != r.fp)) { fprintf(stderr, "runs disagree: %llu vs %llu\n",
                                                (unsigned long long)ri.facts, (unsigned long long)r.facts); return 4; }
        r = ri;
        if (t < best) { best = t; peak_used = pool_peak(true); peak_reserved = pool_peak(false); }
        if (P > 1) trim_pool();
    }
    Json j;
    j.add("engine", "fusedhash").add("prog", prog).num("time", best).num("result", (unsigned long long)r.facts)
     .num("iters", r.iters).num("plan", P).add("set_mode", mode_name(r.mode))
     .num("set_bytes", (unsigned long long)r.set_bytes).num("candidates", (unsigned long long)r.candidates)
     .add("bitmap", o.bitmap).add("append", o.append).add("split", split).num("neq", o.neq)
     .num("K", (unsigned long long)K).num("edges", (unsigned long long)g.rows)
     .num("base_rows", (unsigned long long)(bp ? b.rows : 0))
     .num("loop", (unsigned long long)o.loop).num("words", o.words)
     .num("loop_iters", r.loop_iters).num("word_iters", r.word_iters)
     .num("largest_part", (unsigned long long)r.largest)
     .num("peak_used_gib", peak_used / 1073741824.0).num("peak_reserved_gib", peak_reserved / 1073741824.0);
    if (o.fingerprint) j.num("fingerprint", r.fp);
    j.print();
    return 0;
}
