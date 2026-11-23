// FusedHash for monotonic aggregation on NVIDIA GPUs.
//
//   CC    CC(x, mmin<x>) :- E(x,_).        CC(y, mmin<v>) :- CC(x,v), E(x,y).
//   SSSP  D(s, 0).                          D(y, mmin<d+w>) :- D(x,d), E(x,y,w).
//         dense state lab[v]: old = atomicMin(lab[y], c); if c < old the value of y improved,
//         and y joins the next frontier once, thanks to a per-vertex stamp (atomicExch with the
//         iteration number). Threads read the latest lab[x] (asynchronous iteration).
//   ctrl  C(x,x) :- Company(x).            C(x,z) :- C(x,y), Own(y,z,w), msum(w, <y>) > 0.5.
//         hash map (x,z) -> running sum: old = atomicAdd(S[x,z], w); the thread that sees
//         old <= T < old + w appends C(x,z). Shares in millionths (exact integer sums).
//
// Usage: fusedhash_agg --prog cc|sssp|ctrl --input f.bin [--source s] [--split delta|output]
//        [--reps 2] [--verbose 0] [--chunk 268435456]
// Input: cc/sssp read rows (u, v, w) (w ignored by cc); ctrl reads rows (y, z, w).
#include "common.cuh"

constexpr uint32_t INF = 0xFFFFFFFFu;
constexpr uint32_t THRESHOLD = 500000;              // 0.5 in millionths
constexpr unsigned long long EMPTY64 = 0xFFFFFFFFFFFFFFFFull;

// ------------------------------------------------------------------ mmin (CC, SSSP)

__device__ __forceinline__ void relax(uint32_t* lab, uint32_t* stamp, uint32_t* nxt, unsigned long long* nsize,
                                      uint32_t it, uint32_t y, uint32_t c) {
    if (c < *((volatile uint32_t*)&lab[y])) {       // cheap filter before the atomic
        uint32_t old = atomicMin(&lab[y], c);
        if (c < old && atomicExch(&stamp[y], it) != it) {
            unsigned long long p = atomicAdd(nsize, 1ull);
            nxt[p] = y;
        }
    }
}

__global__ void k_min_delta(uint32_t* lab, uint32_t* stamp, const uint32_t* cur, uint64_t nd, uint32_t* nxt,
                            unsigned long long* nsize, uint32_t it, const uint32_t* off, const uint32_t* ev,
                            const uint32_t* ew, int weighted) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= nd) return;
    uint32_t x = cur[t];
    uint32_t d = *((volatile uint32_t*)&lab[x]);
    for (uint32_t j = off[x], e = off[x + 1]; j < e; ++j)
        relax(lab, stamp, nxt, nsize, it, ev[j], weighted ? d + ew[j] : d);
}

__global__ void k_min_count(const uint32_t* cur, uint64_t nd, const uint32_t* off, unsigned long long* cnt) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t < nd) cnt[t] = off[cur[t] + 1] - off[cur[t]];
}

__device__ __forceinline__ uint64_t owner_incl(const unsigned long long* incl, uint64_t a, uint64_t b, uint64_t g) {
    uint64_t lo = a, hi = b;                        // smallest j in [a,b) with incl[j] > g
    while (lo < hi) {
        uint64_t m = (lo + hi) >> 1;
        if (incl[m] > g) hi = m; else lo = m + 1;
    }
    return lo;
}

__global__ void k_min_output(uint32_t* lab, uint32_t* stamp, const uint32_t* cur, uint64_t nd, uint32_t* nxt,
                             unsigned long long* nsize, uint32_t it, const uint32_t* off, const uint32_t* ev,
                             const uint32_t* ew, int weighted, const unsigned long long* incl, uint64_t total) {
    uint64_t n = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (n >= total) return;
    uint64_t t = owner_incl(incl, 0, nd, n);
    uint64_t ex = t ? incl[t - 1] : 0;
    uint32_t x = cur[t];
    uint32_t j = off[x] + (uint32_t)(n - ex);
    uint32_t d = *((volatile uint32_t*)&lab[x]);
    relax(lab, stamp, nxt, nsize, it, ev[j], weighted ? d + ew[j] : d);
}

__global__ void k_sources(const uint32_t* off, uint32_t K, uint32_t* lab, uint32_t* cur, unsigned long long* n) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= K || off[v + 1] == off[v]) return;
    lab[v] = v;                                     // CC(x, x) :- E(x, _)
    cur[atomicAdd(n, 1ull)] = v;
}

__global__ void k_summary(const uint32_t* lab, uint32_t K, unsigned long long* out) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= K || lab[v] == INF) return;
    atomicAdd(&out[0], 1ull);
    atomicAdd(&out[1], (unsigned long long)lab[v]);
}

struct Result { unsigned long long a = 0, b = 0; int iters = 0; uint64_t candidates = 0; unsigned long long fp = 0; };
static int FINGERPRINT = 0;                       // --fingerprint=1: the fingerprint of the facts (tests)

static Result run_min(const Table& t, bool weighted, uint32_t source, const std::string& split, int verbose) {
    uint32_t K = std::max(t.max_of(0), t.max_of(1)) + 1;
    if (weighted && source >= K) { fprintf(stderr, "source %u >= K %u\n", source, K); exit(1); }
    std::vector<uint32_t> u = t.col(0), v = t.col(1), w = t.col(2);
    DBuf<uint32_t> du = to_device(u), dv = to_device(v), dw = to_device(w);
    CSR E;
    E.build(du.p, dv.p, weighted ? dw.p : nullptr, t.rows, K);
    du.release(); dv.release(); dw.release();
    DBuf<uint32_t> lab(K), stamp(K), cur(K), nxt(K);
    DBuf<unsigned long long> nsize(1);
    lab.fill_bytes(0xFF);
    stamp.zero();
    nsize.zero();
    uint64_t nd;
    if (weighted) { lab.set(source, 0); cur.set(0, source); nd = 1; }
    else {
        k_sources<<<grid_for(K), 256>>>(E.off.p, K, lab.p, cur.p, nsize.p);
        check_launch("k_sources");
        nd = nsize.get(0);
    }
    Result r;
    const uint32_t* ew = weighted ? E.w.p : E.val.p;
    while (nd) {
        ++r.iters;
        double ta = now_s();
        nsize.zero();
        uint64_t total = 0;
        if (split == "output") {
            check_items(nd, "frontier");
            DBuf<unsigned long long> cnt(nd), incl(nd);
            k_min_count<<<grid_for(nd), 256>>>(cur.p, nd, E.off.p, cnt.p);
            inclusive_sum(cnt.p, incl.p, nd);
            total = incl.get(nd - 1);
            if (total) k_min_output<<<grid_for(total), 256>>>(lab.p, stamp.p, cur.p, nd, nxt.p, nsize.p, r.iters,
                                                              E.off.p, E.val.p, ew, weighted, incl.p, total);
        } else {
            k_min_delta<<<grid_for(nd), 256>>>(lab.p, stamp.p, cur.p, nd, nxt.p, nsize.p, r.iters, E.off.p,
                                               E.val.p, ew, weighted);
        }
        check_launch("k_min_step");
        r.candidates += total;
        uint64_t n_new = nsize.get(0);
        if (verbose) fprintf(stderr, "  it %5d: |frontier|=%10llu improved=%10llu %.4fs\n", r.iters,
                             (unsigned long long)nd, (unsigned long long)n_new, sync_now() - ta);
        std::swap(cur, nxt);
        nd = n_new;
    }
    DBuf<unsigned long long> sum(2);
    sum.zero();
    k_summary<<<grid_for(K), 256>>>(lab.p, K, sum.p);
    check_launch("k_summary");
    r.a = sum.get(0);
    r.b = sum.get(1);
    if (FINGERPRINT) r.fp = fingerprint(nullptr, lab.p, K);    // (vertex, label), the reached ones
    return r;
}

// ------------------------------------------------------------------ msum with threshold (company control)

struct MapView {
    unsigned long long* keys;
    uint32_t* sums;
    unsigned long long* nkeys;
    uint64_t mask;
};

// finds or inserts the group (x, z) and returns its slot
__device__ __forceinline__ uint64_t slot(const MapView& m, uint32_t x, uint32_t z) {
    unsigned long long key = ((unsigned long long)x << 32) | z;
    uint64_t h = hash64(key) & m.mask;
    while (true) {
        unsigned long long prev = atomicCAS(&m.keys[h], EMPTY64, key);
        if (prev == EMPTY64) { atomicAdd(m.nkeys, 1ull); return h; }
        if (prev == key) return h;
        h = (h + 1) & m.mask;
    }
}

__device__ __forceinline__ void contribute(const MapView& m, uint32_t* fx, uint32_t* fy, unsigned long long* size,
                                           uint32_t x, uint32_t z, uint32_t w) {
    if (z == x) return;                             // C(x, x) already holds
    uint64_t h = slot(m, x, z);
    uint32_t old = atomicAdd(&m.sums[h], w);
    if (old <= THRESHOLD && old + w > THRESHOLD) {  // this thread makes the sum cross the threshold
        unsigned long long p = atomicAdd(size, 1ull);
        fx[p] = x;
        fy[p] = z;
    }
}

__global__ void k_ctrl_count(const uint32_t* fy, uint64_t d0, uint64_t nd, const uint32_t* off, unsigned long long* cnt) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t < nd) { uint32_t y = fy[d0 + t]; cnt[t] = off[y + 1] - off[y]; }
}

__global__ void k_ctrl_delta(MapView m, uint32_t* fx, uint32_t* fy, unsigned long long* size, const uint32_t* off,
                             const uint32_t* oz, const uint32_t* ow, uint64_t d0, uint64_t a, uint64_t b) {
    uint64_t t = a + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= b) return;
    uint32_t x = fx[d0 + t], y = fy[d0 + t];
    for (uint32_t j = off[y], e = off[y + 1]; j < e; ++j) contribute(m, fx, fy, size, x, oz[j], ow[j]);
}

__global__ void k_ctrl_output(MapView m, uint32_t* fx, uint32_t* fy, unsigned long long* size, const uint32_t* off,
                              const uint32_t* oz, const uint32_t* ow, const unsigned long long* incl, uint64_t d0,
                              uint64_t a, uint64_t b, uint64_t base, uint64_t total) {
    uint64_t n = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (n >= total) return;
    uint64_t g = base + n;
    uint64_t t = owner_incl(incl, a, b, g);
    uint64_t ex = t ? incl[t - 1] : 0;
    uint32_t x = fx[d0 + t], y = fy[d0 + t];
    uint32_t j = off[y] + (uint32_t)(g - ex);
    contribute(m, fx, fy, size, x, oz[j], ow[j]);
}

__global__ void k_map_rehash(const unsigned long long* okeys, const uint32_t* osums, uint64_t ocap, MapView m) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= ocap || okeys[i] == EMPTY64) return;
    unsigned long long k = okeys[i];
    uint64_t h = slot(m, (uint32_t)(k >> 32), (uint32_t)k);
    m.sums[h] = osums[i];
}

__global__ void k_diag(uint32_t* fx, uint32_t* fy, uint32_t K) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < K) { fx[v] = v; fy[v] = v; }
}

__global__ void k_cuts(const unsigned long long* incl, uint64_t nd, uint64_t chunk, uint64_t ncuts,
                       unsigned long long* cut) {
    uint64_t k = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (k < ncuts) cut[k] = owner_incl(incl, 0, nd, (k + 1) * chunk);
}

struct MapStore {
    DBuf<uint32_t> fx, fy, sums;
    DBuf<unsigned long long> keys, size, nkeys;
    uint64_t cap = 0, F = 0, nk = 0;
    double load = 0.5;
    static uint64_t pow2(uint64_t n) { uint64_t p = 1024; while (p < n) p <<= 1; return p; }
    MapView view() const { return MapView{keys.p, sums.p, nkeys.p, cap - 1}; }
    void init(uint32_t K) {
        fx.alloc(pow2(4ull * K)); fy.alloc(fx.n);
        k_diag<<<grid_for(K), 256>>>(fx.p, fy.p, K);
        size.alloc(1); size.set(0, K); F = K;
        nkeys.alloc(1); nkeys.zero();
        cap = pow2((uint64_t)(K / load) + 1);
        keys.alloc(cap); keys.fill_bytes(0xFF);
        sums.alloc(cap); sums.zero();
    }
    void sync() { F = size.get(0); nk = nkeys.get(0); }
    void ensure(uint64_t extra) {
        if (F + extra > fx.n) { uint64_t c = pow2(F + extra); fx.grow(c, F); fy.grow(c, F); }
        uint64_t need = nk + extra;
        if (need > load * cap) {
            DBuf<unsigned long long> okeys = std::move(keys);
            DBuf<uint32_t> osums = std::move(sums);
            uint64_t ocap = cap;
            cap = pow2((uint64_t)(need / load) + 1);
            keys.alloc(cap); keys.fill_bytes(0xFF);
            sums.alloc(cap); sums.zero();
            nkeys.zero();
            k_map_rehash<<<grid_for(ocap), 256>>>(okeys.p, osums.p, ocap, view());
            check_launch("k_map_rehash");
            nk = nkeys.get(0);
        }
    }
};

static Result run_ctrl(const Table& t, const std::string& split, uint64_t chunk, int verbose) {
    uint32_t K = std::max(t.max_of(0), t.max_of(1)) + 1;
    std::vector<uint32_t> y = t.col(0), z = t.col(1), w = t.col(2);
    DBuf<uint32_t> dy = to_device(y), dz = to_device(z), dw = to_device(w);
    CSR own;                                         // shares of each owner y: off[y].. -> (z, w)
    own.build(dy.p, dz.p, dw.p, t.rows, K);
    dy.release(); dz.release(); dw.release();
    MapStore st;
    st.init(K);
    Result r;
    uint64_t d0 = 0;
    while (st.F > d0) {
        ++r.iters;
        double ta = now_s();
        uint64_t F0 = st.F, nd = F0 - d0;
        check_items(nd, "delta");
        DBuf<unsigned long long> cnt(nd), incl(nd);
        k_ctrl_count<<<grid_for(nd), 256>>>(st.fy.p, d0, nd, own.off.p, cnt.p);
        inclusive_sum(cnt.p, incl.p, nd);
        uint64_t R = incl.get(nd - 1);
        r.candidates += R;
        std::vector<uint64_t> cuts{0};
        if (R > chunk) {
            uint64_t nc = (R - 1) / chunk;
            DBuf<unsigned long long> dcut(nc);
            k_cuts<<<grid_for(nc), 256>>>(incl.p, nd, chunk, nc, dcut.p);
            check_launch("k_cuts");
            std::vector<unsigned long long> h(nc);
            CK(cudaMemcpy(h.data(), dcut.p, nc * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            for (auto c : h) if (c > cuts.back() && c < nd) cuts.push_back(c);
        }
        cuts.push_back(nd);
        for (size_t k = 0; k + 1 < cuts.size(); ++k) {
            uint64_t a = cuts[k], b = cuts[k + 1];
            uint64_t base = a ? incl.get(a - 1) : 0, rr = incl.get(b - 1) - base;
            if (!rr) continue;
            st.ensure(rr);
            if (split == "output")
                k_ctrl_output<<<grid_for(rr), 256>>>(st.view(), st.fx.p, st.fy.p, st.size.p, own.off.p, own.val.p,
                                                     own.w.p, incl.p, d0, a, b, base, rr);
            else
                k_ctrl_delta<<<grid_for(b - a), 256>>>(st.view(), st.fx.p, st.fy.p, st.size.p, own.off.p, own.val.p,
                                                       own.w.p, d0, a, b);
            check_launch("k_ctrl_step");
            st.sync();
        }
        d0 = F0;
        if (verbose) fprintf(stderr, "  it %4d: |delta|=%10llu shares=%12llu new=%10llu groups=%llu %.4fs\n", r.iters,
                             (unsigned long long)nd, (unsigned long long)R, (unsigned long long)(st.F - d0),
                             (unsigned long long)st.nk, now_s() - ta);
    }
    r.a = st.F;
    r.b = st.nk;
    if (FINGERPRINT) r.fp = fingerprint(st.fx.p, st.fy.p, st.F);
    return r;
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    std::string prog = a.str("prog", "cc"), input = a.str("input");
    if (input.empty()) { fprintf(stderr, "missing --input\n"); return 1; }
    std::string split = a.str("split", "delta");
    int reps = (int)a.num("reps", 2), verbose = (int)a.num("verbose", 0);
    FINGERPRINT = (int)a.num("fingerprint", 0);
    uint32_t source = (uint32_t)a.num("source", 0);
    uint64_t chunk = (uint64_t)a.num("chunk", 1ll << 28);
    print_device();
    init_pool();
    Table t = read_bin(input, 3);
    auto run = [&](const Table& x, uint32_t s) {
        if (prog == "ctrl") return run_ctrl(x, split, chunk, verbose);
        return run_min(x, prog == "sssp", s, split, verbose);
    };
    {                                                // warm-up on a small chain
        Table w;
        w.cols = 3;
        for (uint32_t i = 0; i + 1 < 2000; ++i) { w.data.push_back(i); w.data.push_back(i + 1); w.data.push_back(600000); }
        w.rows = w.data.size() / 3;
        run(w, 0);
        CK(cudaDeviceSynchronize());
    }
    double best = 1e30;
    Result r;
    uint64_t peak_used = 0, peak_reserved = 0;             // GPU memory of the best run (memory pool)
    for (int i = 0; i < reps; ++i) {
        reset_peak();
        double t0 = sync_now();
        Result ri = run(t, source);
        double el = sync_now() - t0;
        if (i && (ri.a != r.a || ri.fp != r.fp || (prog != "ctrl" && ri.b != r.b))) { fprintf(stderr, "runs disagree\n"); return 4; }
        r = ri;
        if (el < best) { best = el; peak_used = pool_peak(true); peak_reserved = pool_peak(false); }
    }
    Json j;
    j.add("engine", "fusedhash").add("prog", prog).num("time", best).num("iters", r.iters)
     .num("candidates", (unsigned long long)r.candidates).add("split", split);
    if (prog == "ctrl") j.num("result", r.a).num("map_keys", r.b);
    else j.num("result", r.a).num("checksum", r.b);
    if (FINGERPRINT) j.num("fingerprint", r.fp);
    j.num("peak_used_gib", peak_used / 1073741824.0).num("peak_reserved_gib", peak_reserved / 1073741824.0);
    j.print();
    return 0;
}
