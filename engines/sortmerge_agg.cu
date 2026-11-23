// SortMerge for monotonic aggregation on NVIDIA GPUs: the
// extension of the sort-based design of FVlog to aggregates (FVlog itself has none).
//
// Per iteration: (1) two-phase join with materialized (group, value) candidates; (2) radix sort
// by group and segmented reduction (min or sum; CUB ReduceByKey); (3) lookup of each group in
// the aggregated relation full, sorted by group: an existing group is updated in place, a new
// group is flagged; (4) the new groups are merged into full (eager merge, full stays sorted),
// and the value -> position index of full is rebuilt.
//   CC / SSSP  full = (vertex, value) sorted by vertex; index: offk[v] = position of v (dense).
//   ctrl       full = (x*K+z, sum) sorted by key; lookup by binary search.
//
// Usage: sortmerge_agg --prog cc|sssp|ctrl --input f.bin [--source s] [--reps 2] [--verbose 0]
#include "common.cuh"

constexpr uint32_t THRESHOLD = 500000;

struct MinOp {
    __host__ __device__ uint32_t operator()(const uint32_t& a, const uint32_t& b) const { return b < a ? b : a; }
};
struct SumOp {
    __host__ __device__ uint32_t operator()(const uint32_t& a, const uint32_t& b) const { return a + b; }
};

__device__ __forceinline__ uint64_t owner_excl(const unsigned long long* excl, uint64_t m, uint64_t g) {
    uint64_t lo = 0, hi = m;
    while (hi - lo > 1) {
        uint64_t c = (lo + hi) >> 1;
        if (excl[c] <= g) lo = c; else hi = c;
    }
    return lo;
}

__global__ void k_count(const uint32_t* key, uint64_t n, const uint32_t* off, unsigned long long* cnt) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) cnt[i] = off[key[i] + 1] - off[key[i]];
}

// join of CC/SSSP: (oy, oc) = (neighbor, value of the source [+ weight])
__global__ void k_join_relax(const unsigned long long* excl, uint64_t m, uint64_t total, const uint32_t* dk,
                             const uint32_t* dv, const uint32_t* off, const uint32_t* ev, const uint32_t* ew,
                             int weighted, uint32_t* oy, uint32_t* oc) {
    uint64_t n = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (n >= total) return;
    uint64_t j = owner_excl(excl, m, n);
    uint32_t r = off[dk[j]] + (uint32_t)(n - excl[j]);
    oy[n] = ev[r];
    oc[n] = weighted ? dv[j] + ew[r] : dv[j];
}

// CC/SSSP: group y with candidate c against full (index offk: 0 or 1 rows per vertex)
__global__ void k_min_apply(const uint32_t* y, const uint32_t* c, uint64_t n, const uint32_t* offk, uint32_t* fv,
                            uint8_t* fresh, uint8_t* better, uint32_t* ins) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t p = offk[y[i]];
    ins[i] = p;
    if (offk[y[i] + 1] > p) {
        fresh[i] = 0;
        bool b = c[i] < fv[p];
        if (b) fv[p] = c[i];                        // unique groups after the reduction: no conflicts
        better[i] = b;
    } else {
        fresh[i] = 1;
        better[i] = 1;
    }
}

__global__ void k_hist32(const uint32_t* ins, uint64_t m, uint32_t* h) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < m) atomicAdd(&h[ins[i]], 1u);
}

__global__ void k_hist64(const unsigned long long* ins, uint64_t m, uint32_t* h) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < m) atomicAdd(&h[ins[i]], 1u);
}

template <class K, class P>
__global__ void k_merge2(const K* fk, const uint32_t* fv, uint64_t nf, const uint32_t* cum, const K* dk,
                         const uint32_t* dv, const P* ins, uint64_t m, K* ok, uint32_t* ov) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t < nf) { uint64_t p = t + cum[t]; ok[p] = fk[t]; ov[p] = fv[t]; }
    else if (t < nf + m) { uint64_t i = t - nf, p = ins[i] + i; ok[p] = dk[i]; ov[p] = dv[i]; }
}

__global__ void k_mark(const uint32_t* fk, uint64_t nf, uint32_t* pres) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < nf) pres[fk[i]] = 1;
}

__global__ void k_sum_values(const uint32_t* v, uint64_t n, unsigned long long* out) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, (unsigned long long)v[i]);
}

__global__ void k_has_edges(const uint32_t* off, uint32_t K, uint8_t* flag) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < K) flag[v] = off[v + 1] > off[v];
}

__global__ void k_iota(uint32_t* a, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) a[i] = (uint32_t)i;
}

template <class T>
static uint64_t select_flagged(const T* in, const uint8_t* flags, T* out, uint64_t n) {
    if (!n) return 0;
    check_items(n, "select");
    DBuf<unsigned long long> ns(1);
    size_t bytes = 0;
    CK(cub::DeviceSelect::Flagged(nullptr, bytes, in, flags, out, ns.p, n));
    CK(cub::DeviceSelect::Flagged(cub_temp().get(bytes), bytes, in, flags, out, ns.p, n));
    return ns.get(0);
}

template <class K, class Op>
static uint64_t sort_reduce(DBuf<K>& keys, DBuf<uint32_t>& vals, uint64_t n, int key_bits, Op op,
                            DBuf<K>& uk, DBuf<uint32_t>& uv) {
    check_items(n, "candidates");
    DBuf<K> k2(n);
    DBuf<uint32_t> v2(n);
    cub::DoubleBuffer<K> dk(keys.p, k2.p);
    cub::DoubleBuffer<uint32_t> dv(vals.p, v2.p);
    size_t bytes = 0;
    CK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, dk, dv, n, 0, key_bits));
    CK(cub::DeviceRadixSort::SortPairs(cub_temp().get(bytes), bytes, dk, dv, n, 0, key_bits));
    uk.alloc(n);
    uv.alloc(n);
    DBuf<unsigned long long> nr(1);
    bytes = 0;
    CK(cub::DeviceReduce::ReduceByKey(nullptr, bytes, dk.Current(), uk.p, dv.Current(), uv.p, nr.p, op, n));
    CK(cub::DeviceReduce::ReduceByKey(cub_temp().get(bytes), bytes, dk.Current(), uk.p, dv.Current(), uv.p, nr.p, op, n));
    return nr.get(0);
}

struct Result { unsigned long long a = 0, b = 0; int iters = 0; uint64_t candidates = 0; unsigned long long fp = 0; };
static int FINGERPRINT = 0;                       // --fingerprint=1: the fingerprint of the facts (tests)

// ------------------------------------------------------------------ CC / SSSP

static Result run_min(const Table& t, bool weighted, uint32_t source, int verbose) {
    uint32_t K = std::max(t.max_of(0), t.max_of(1)) + 1;
    if (weighted && source >= K) { fprintf(stderr, "source %u >= K %u\n", source, K); exit(1); }
    std::vector<uint32_t> u = t.col(0), v = t.col(1), w = t.col(2);
    DBuf<uint32_t> du = to_device(u), dv_ = to_device(v), dw = to_device(w);
    CSR E;
    E.build(du.p, dv_.p, weighted ? dw.p : nullptr, t.rows, K);
    du.release(); dv_.release(); dw.release();
    const uint32_t* ew = weighted ? E.w.p : E.val.p;
    int kb = bits_for(K - 1);
    DBuf<uint32_t> fk, fv, offk(K + 1), pres(K + 1);
    uint64_t nf;
    if (weighted) { fk.alloc(1); fv.alloc(1); fk.set(0, source); fv.set(0, 0); nf = 1; }
    else {                                           // CC(x, x) :- E(x, _), sorted by vertex
        DBuf<uint8_t> flag(K);
        k_has_edges<<<grid_for(K), 256>>>(E.off.p, K, flag.p);
        DBuf<uint32_t> ids(K);
        k_iota<<<grid_for(K), 256>>>(ids.p, K);
        fk.alloc(K);
        nf = select_flagged(ids.p, flag.p, fk.p, K);
        fv.alloc(K);
        CK(cudaMemcpyAsync(fv.p, fk.p, nf * sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));
    }
    auto rebuild_index = [&]() {
        pres.zero();
        k_mark<<<grid_for(nf), 256>>>(fk.p, nf, pres.p);
        exclusive_sum(pres.p, offk.p, (size_t)K + 1);
    };
    rebuild_index();
    DBuf<uint32_t> dk(nf), dvv(nf);                  // Delta = full
    CK(cudaMemcpyAsync(dk.p, fk.p, nf * sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));
    CK(cudaMemcpyAsync(dvv.p, fv.p, nf * sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));
    uint64_t nd = nf;
    Result r;
    while (nd) {
        ++r.iters;
        double ta = now_s();
        DBuf<unsigned long long> cnt(nd), excl(nd);
        k_count<<<grid_for(nd), 256>>>(dk.p, nd, E.off.p, cnt.p);
        exclusive_sum(cnt.p, excl.p, nd);
        uint64_t total = excl.get(nd - 1) + cnt.get(nd - 1);
        r.candidates += total;
        if (!total) break;
        DBuf<uint32_t> oy(total), oc(total);
        k_join_relax<<<grid_for(total), 256>>>(excl.p, nd, total, dk.p, dvv.p, E.off.p, E.val.p, ew, weighted,
                                               oy.p, oc.p);
        check_launch("k_join_relax");
        cnt.release(); excl.release();
        DBuf<uint32_t> uy, uc;
        uint64_t n = sort_reduce(oy, oc, total, kb, MinOp{}, uy, uc);
        oy.release(); oc.release();
        DBuf<uint8_t> fresh(n), better(n);
        DBuf<uint32_t> ins(n);
        k_min_apply<<<grid_for(n), 256>>>(uy.p, uc.p, n, offk.p, fv.p, fresh.p, better.p, ins.p);
        check_launch("k_min_apply");
        DBuf<uint32_t> ndk(n), ndv(n);               // next Delta: the improved groups
        uint64_t nd2 = select_flagged(uy.p, better.p, ndk.p, n);
        select_flagged(uc.p, better.p, ndv.p, n);
        DBuf<uint32_t> nk(n), nv(n), nins(n);        // new groups, merged into full
        uint64_t m = select_flagged(uy.p, fresh.p, nk.p, n);
        select_flagged(uc.p, fresh.p, nv.p, n);
        select_flagged(ins.p, fresh.p, nins.p, n);
        if (m) {
            DBuf<uint32_t> h(nf + 1), cum(nf + 1);
            h.zero();
            k_hist32<<<grid_for(m), 256>>>(nins.p, m, h.p);
            check_items(nf + 1, "full");
            inclusive_sum(h.p, cum.p, nf + 1);
            DBuf<uint32_t> ok(nf + m), ov(nf + m);
            k_merge2<uint32_t, uint32_t><<<grid_for(nf + m), 256>>>(fk.p, fv.p, nf, cum.p, nk.p, nv.p, nins.p, m,
                                                                    ok.p, ov.p);
            check_launch("k_merge2");
            fk = std::move(ok);
            fv = std::move(ov);
            nf += m;
            rebuild_index();
        }
        dk = std::move(ndk);
        dvv = std::move(ndv);
        nd = nd2;
        if (verbose) fprintf(stderr, "  it %5d: candidates=%12llu groups=%10llu improved=%10llu new=%9llu %.4fs\n",
                             r.iters, (unsigned long long)total, (unsigned long long)n, (unsigned long long)nd,
                             (unsigned long long)m, sync_now() - ta);
    }
    DBuf<unsigned long long> s(1);
    s.zero();
    if (nf) k_sum_values<<<grid_for(nf), 256>>>(fv.p, nf, s.p);
    r.a = nf;
    r.b = s.get(0);
    if (FINGERPRINT) r.fp = fingerprint(fk.p, fv.p, nf);
    return r;
}

// ------------------------------------------------------------------ company control

__global__ void k_join_ctrl(const unsigned long long* excl, uint64_t m, uint64_t total, const uint32_t* dx,
                            const uint32_t* dy, const uint32_t* off, const uint32_t* oz, const uint32_t* ow, uint32_t K,
                            unsigned long long* key, uint32_t* w, uint8_t* keep) {
    uint64_t n = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (n >= total) return;
    uint64_t j = owner_excl(excl, m, n);
    uint32_t r = off[dy[j]] + (uint32_t)(n - excl[j]);
    uint32_t x = dx[j], z = oz[r];
    key[n] = (unsigned long long)x * K + z;
    w[n] = ow[r];
    keep[n] = z != x;
}

__global__ void k_find(const unsigned long long* u, uint64_t n, const unsigned long long* sk, uint64_t nf,
                       uint8_t* fresh, unsigned long long* ins) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long k = u[i];
    uint64_t lo = 0, hi = nf;
    while (lo < hi) { uint64_t c = (lo + hi) >> 1; if (sk[c] < k) lo = c + 1; else hi = c; }
    ins[i] = lo;
    fresh[i] = (lo < nf && sk[lo] == k) ? 0 : 1;
}

__global__ void k_acc_apply(const uint8_t* fresh, const unsigned long long* ins, const uint32_t* s, uint64_t n,
                            uint32_t* sv, uint8_t* cross) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (fresh[i]) { cross[i] = s[i] > THRESHOLD; return; }
    uint32_t old = sv[ins[i]], nw = old + s[i];
    sv[ins[i]] = nw;
    cross[i] = old <= THRESHOLD && nw > THRESHOLD;
}

__global__ void k_decode(const unsigned long long* k, uint64_t n, uint32_t K, uint32_t* x, uint32_t* y) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) { x[i] = (uint32_t)(k[i] / K); y[i] = (uint32_t)(k[i] % K); }
}

static Result run_ctrl(const Table& t, int verbose) {
    uint32_t K = std::max(t.max_of(0), t.max_of(1)) + 1;
    std::vector<uint32_t> y = t.col(0), z = t.col(1), w = t.col(2);
    DBuf<uint32_t> dy0 = to_device(y), dz0 = to_device(z), dw0 = to_device(w);
    CSR own;
    own.build(dy0.p, dz0.p, dw0.p, t.rows, K);
    dy0.release(); dz0.release(); dw0.release();
    int kb = bits_for((uint64_t)K * K - 1);
    DBuf<unsigned long long> sk;                     // aggregated relation S: key x*K+z, sum
    DBuf<uint32_t> sv;
    uint64_t nf = 0;
    DBuf<uint32_t> dx(K), dy(K);                     // Delta = C(x, x)
    k_iota<<<grid_for(K), 256>>>(dx.p, K);
    k_iota<<<grid_for(K), 256>>>(dy.p, K);
    uint64_t nd = K;
    Result r;
    r.a = K;
    if (FINGERPRINT) r.fp = fingerprint(dx.p, dy.p, K);           // C(x, x)
    while (nd) {
        ++r.iters;
        double ta = now_s();
        DBuf<unsigned long long> cnt(nd), excl(nd);
        k_count<<<grid_for(nd), 256>>>(dy.p, nd, own.off.p, cnt.p);
        exclusive_sum(cnt.p, excl.p, nd);
        uint64_t total = excl.get(nd - 1) + cnt.get(nd - 1);
        r.candidates += total;
        if (!total) break;
        DBuf<unsigned long long> jk(total);
        DBuf<uint32_t> jw(total);
        DBuf<uint8_t> keep(total);
        k_join_ctrl<<<grid_for(total), 256>>>(excl.p, nd, total, dx.p, dy.p, own.off.p, own.val.p, own.w.p, K,
                                              jk.p, jw.p, keep.p);
        check_launch("k_join_ctrl");
        cnt.release(); excl.release();
        DBuf<unsigned long long> fk(total);          // z != x
        DBuf<uint32_t> fw(total);
        uint64_t nj = select_flagged(jk.p, keep.p, fk.p, total);
        select_flagged(jw.p, keep.p, fw.p, total);
        jk.release(); jw.release(); keep.release();
        if (!nj) break;
        DBuf<unsigned long long> uk;
        DBuf<uint32_t> us;
        uint64_t n = sort_reduce(fk, fw, nj, kb, SumOp{}, uk, us);
        fk.release(); fw.release();
        DBuf<uint8_t> fresh(n), cross(n);
        DBuf<unsigned long long> ins(n);
        k_find<<<grid_for(n), 256>>>(uk.p, n, sk.p, nf, fresh.p, ins.p);
        k_acc_apply<<<grid_for(n), 256>>>(fresh.p, ins.p, us.p, n, sv.p, cross.p);
        check_launch("k_acc_apply");
        DBuf<unsigned long long> ck(n);              // next Delta: the groups that crossed the threshold
        uint64_t nd2 = select_flagged(uk.p, cross.p, ck.p, n);
        DBuf<unsigned long long> nk(n), nins(n);     // new groups, merged into S
        DBuf<uint32_t> nv(n);
        uint64_t m = select_flagged(uk.p, fresh.p, nk.p, n);
        select_flagged(us.p, fresh.p, nv.p, n);
        select_flagged(ins.p, fresh.p, nins.p, n);
        if (m) {
            DBuf<uint32_t> h(nf + 1), cum(nf + 1);
            h.zero();
            k_hist64<<<grid_for(m), 256>>>(nins.p, m, h.p);
            check_items(nf + 1, "full");
            inclusive_sum(h.p, cum.p, nf + 1);
            DBuf<unsigned long long> ok(nf + m);
            DBuf<uint32_t> ov(nf + m);
            k_merge2<unsigned long long, unsigned long long><<<grid_for(nf + m), 256>>>(
                sk.p, sv.p, nf, cum.p, nk.p, nv.p, nins.p, m, ok.p, ov.p);
            check_launch("k_merge2");
            sk = std::move(ok);
            sv = std::move(ov);
            nf += m;
        }
        dx.alloc(nd2);
        dy.alloc(nd2);
        if (nd2) k_decode<<<grid_for(nd2), 256>>>(ck.p, nd2, K, dx.p, dy.p);
        nd = nd2;
        r.a += nd2;
        if (FINGERPRINT) r.fp += fingerprint(dx.p, dy.p, nd2);   // the facts of this Delta, each new
        if (verbose) fprintf(stderr, "  it %4d: shares=%12llu groups=%10llu new groups=%10llu delta=%9llu |S|=%llu %.4fs\n",
                             r.iters, (unsigned long long)total, (unsigned long long)n, (unsigned long long)m,
                             (unsigned long long)nd2, (unsigned long long)nf, sync_now() - ta);
    }
    r.b = nf;
    return r;
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    std::string prog = a.str("prog", "cc"), input = a.str("input");
    if (input.empty()) { fprintf(stderr, "missing --input\n"); return 1; }
    int reps = (int)a.num("reps", 2), verbose = (int)a.num("verbose", 0);
    uint32_t source = (uint32_t)a.num("source", 0);
    FINGERPRINT = (int)a.num("fingerprint", 0);
    print_device();
    init_pool();
    Table t = read_bin(input, 3);
    auto run = [&](const Table& x, uint32_t s) {
        return prog == "ctrl" ? run_ctrl(x, verbose) : run_min(x, prog == "sssp", s, verbose);
    };
    {
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
    j.add("engine", "sortmerge").add("prog", prog).num("time", best).num("iters", r.iters)
     .num("candidates", (unsigned long long)r.candidates);
    if (prog == "ctrl") j.num("result", r.a).num("map_keys", r.b);
    else j.num("result", r.a).num("checksum", r.b);
    if (FINGERPRINT) j.num("fingerprint", r.fp);
    j.num("peak_used_gib", peak_used / 1073741824.0).num("peak_reserved_gib", peak_reserved / 1073741824.0);
    j.print();
    return 0;
}
