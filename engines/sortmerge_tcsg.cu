// SortMerge for TC and SG on NVIDIA GPUs: our re-implementation of the sort-based design of
// FVlog. The original FVlog is also run in the experiments;
// this engine is kept to compare the two designs with the same code base, and it is the basis
// of the sort-based aggregation engine (sortmerge_agg.cu).
//
// Per iteration, as in FVlog: (1) two-phase join with materialized output (count, prefix sum,
// one thread per output tuple writing the projected tuple); (2) radix sort of the candidates and
// removal of duplicates; (3) difference against full; (4) eager merge of Delta into full, which
// stays sorted and contiguous. Deviations, all in favor of the sort-based design:
//   * a tuple (x,y) is packed into the 64-bit key x*K+y, whose order is the lexicographic order
//     of (x,y): one radix sort (CUB, over the significant bits only) and one merge replace the
//     column-wise passes of FVlog;
//   * the difference is a binary search of each candidate in full (O(log |full|)), which also
//     gives its insertion position; the merge is a merge path in O(|full| + |Delta|);
//   * the join writes the projected keys directly.
//
// Usage: sortmerge_tcsg --prog tc|sg --input g.bin [--plan 1] [--neq 0|1] [--reps 2] [--verbose 0]
#include "common.cuh"

__global__ void k_split_keys(const unsigned long long* k, uint64_t n, uint32_t K, uint32_t* hi, uint32_t* lo) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    hi[i] = (uint32_t)(k[i] / K);
    lo[i] = (uint32_t)(k[i] % K);
}

// counts of a join on the key column `key` against the CSR `off`
__global__ void k_join_count(const uint32_t* key, uint64_t n, const uint32_t* off, unsigned long long* cnt) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    cnt[i] = off[key[i] + 1] - off[key[i]];
}

__device__ __forceinline__ uint64_t owner_excl(const unsigned long long* excl, uint64_t m, uint64_t g) {
    uint64_t lo = 0, hi = m;                           // last j with excl[j] <= g
    while (hi - lo > 1) {
        uint64_t c = (lo + hi) >> 1;
        if (excl[c] <= g) lo = c; else hi = c;
    }
    return lo;
}

// Algorithm 1, write phase, projecting (keep[j], adj[...]) into the packed key keep*K + adj
__global__ void k_join_expand_key(const unsigned long long* excl, uint64_t m, uint64_t total, const uint32_t* key,
                                  const uint32_t* keep, const uint32_t* off, const uint32_t* adj, uint32_t K,
                                  unsigned long long* out) {
    uint64_t n = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (n >= total) return;
    uint64_t j = owner_excl(excl, m, n);
    uint32_t y = adj[off[key[j]] + (uint32_t)(n - excl[j])];
    out[n] = (unsigned long long)keep[j] * K + y;
}

// the same, projecting into two columns (keep[j], adj[...]) (first join of SG)
__global__ void k_join_expand_cols(const unsigned long long* excl, uint64_t m, uint64_t total, const uint32_t* key,
                                   const uint32_t* keep, const uint32_t* off, const uint32_t* adj,
                                   uint32_t* ok, uint32_t* ov) {
    uint64_t n = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (n >= total) return;
    uint64_t j = owner_excl(excl, m, n);
    ok[n] = keep[j];
    ov[n] = adj[off[key[j]] + (uint32_t)(n - excl[j])];
}

struct NeqKey {
    uint32_t K;
    __host__ __device__ bool operator()(const unsigned long long& k) const { return k / K != k % K; }
};

// Algorithm 2 (sorted variant): fresh = key absent from full; ins = lower bound in full
__global__ void k_diff(const unsigned long long* u, uint64_t n, const unsigned long long* full, uint64_t nf,
                       uint8_t* fresh, unsigned long long* ins) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long k = u[i];
    uint64_t lo = 0, hi = nf;
    while (lo < hi) {
        uint64_t c = (lo + hi) >> 1;
        if (full[c] < k) lo = c + 1; else hi = c;
    }
    ins[i] = lo;
    fresh[i] = (lo < nf && full[lo] == k) ? 0 : 1;
}

// merge path: h[j] = number of Delta tuples inserted before full[j]
__global__ void k_merge_hist(const unsigned long long* ins, uint64_t m, uint32_t* h) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < m) atomicAdd(&h[ins[i]], 1u);
}

// full[t] goes to t + cum[t] (cum inclusive); Delta[i] to ins[i] + i
__global__ void k_merge_scatter(const unsigned long long* full, uint64_t nf, const uint32_t* cum,
                                const unsigned long long* d, const unsigned long long* ins, uint64_t m,
                                unsigned long long* out) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t < nf) out[t + cum[t]] = full[t];
    else if (t < nf + m) { uint64_t i = t - nf; out[ins[i] + i] = d[i]; }
}

struct Rel {
    uint32_t K;
    int key_bits;
    DBuf<unsigned long long> full, spare;              // double buffer for the eager merge
    uint64_t nf = 0;
    explicit Rel(uint32_t K_) : K(K_) { key_bits = bits_for((uint64_t)K * K - 1); }

    // dedup of the candidates, difference with full, eager merge; returns Delta (sorted)
    DBuf<unsigned long long> close(DBuf<unsigned long long>& cand, uint64_t n, uint64_t& n_unique) {
        DBuf<unsigned long long> delta;
        n_unique = 0;
        if (!n) return delta;
        check_items(n, "candidates");
        DBuf<unsigned long long> alt(n);               // radix sort (CUB, significant bits only)
        cub::DoubleBuffer<unsigned long long> db(cand.p, alt.p);
        size_t bytes = 0;
        CK(cub::DeviceRadixSort::SortKeys(nullptr, bytes, db, n, 0, key_bits));
        CK(cub::DeviceRadixSort::SortKeys(cub_temp().get(bytes), bytes, db, n, 0, key_bits));
        unsigned long long* sorted = db.Current();
        DBuf<unsigned long long> u(n);                 // unique
        DBuf<unsigned long long> nsel(1);
        bytes = 0;
        CK(cub::DeviceSelect::Unique(nullptr, bytes, sorted, u.p, nsel.p, n));
        CK(cub::DeviceSelect::Unique(cub_temp().get(bytes), bytes, sorted, u.p, nsel.p, n));
        uint64_t nu = nsel.get(0);
        n_unique = nu;
        alt.release();
        cand.release();
        DBuf<uint8_t> fresh(nu);                       // difference
        DBuf<unsigned long long> ins(nu);
        k_diff<<<grid_for(nu), 256>>>(u.p, nu, full.p, nf, fresh.p, ins.p);
        check_launch("k_diff");
        delta.alloc(nu);
        DBuf<unsigned long long> dins(nu);
        bytes = 0;
        CK(cub::DeviceSelect::Flagged(nullptr, bytes, u.p, fresh.p, delta.p, nsel.p, nu));
        CK(cub::DeviceSelect::Flagged(cub_temp().get(bytes), bytes, u.p, fresh.p, delta.p, nsel.p, nu));
        uint64_t m = nsel.get(0);
        CK(cub::DeviceSelect::Flagged(cub_temp().get(bytes), bytes, ins.p, fresh.p, dins.p, nsel.p, nu));
        if (m) {                                       // eager merge (merge path)
            DBuf<uint32_t> h(nf + 1), cum(nf + 1);
            h.zero();
            k_merge_hist<<<grid_for(m), 256>>>(dins.p, m, h.p);
            check_launch("k_merge_hist");
            check_items(nf + 1, "full");
            inclusive_sum(h.p, cum.p, nf + 1);
            spare.grow(nf + m, 0);
            if (spare.n < nf + m) spare.alloc(nf + m);
            k_merge_scatter<<<grid_for(nf + m), 256>>>(full.p, nf, cum.p, delta.p, dins.p, m, spare.p);
            check_launch("k_merge_scatter");
            std::swap(full, spare);
            nf += m;
            if (spare.n < nf) spare.grow(full.n, 0);   // keep the spare as large as full
        }
        delta.n = m;                                   // logical size (capacity is nu)
        return delta;
    }
};

struct RunResult { uint64_t facts = 0; int iters = 0; uint64_t candidates = 0; unsigned long long fp = 0; };
static int FINGERPRINT = 0;                             // --fingerprint=1: the fingerprint of the facts (tests)

// Delta(x,z) join E(z,y) -> keys x*K+y
static DBuf<unsigned long long> step_tc(const DBuf<unsigned long long>& delta, uint64_t nd, const CSR& E, uint32_t K,
                                        uint64_t& total) {
    DBuf<uint32_t> dx(nd), dz(nd);
    k_split_keys<<<grid_for(nd), 256>>>(delta.p, nd, K, dx.p, dz.p);
    check_items(nd, "delta");
    DBuf<unsigned long long> cnt(nd), excl(nd);
    k_join_count<<<grid_for(nd), 256>>>(dz.p, nd, E.off.p, cnt.p);
    exclusive_sum(cnt.p, excl.p, nd);
    total = excl.get(nd - 1) + cnt.get(nd - 1);
    DBuf<unsigned long long> out(total);
    if (total) k_join_expand_key<<<grid_for(total), 256>>>(excl.p, nd, total, dz.p, dx.p, E.off.p, E.val.p, K, out.p);
    check_launch("step_tc");
    return out;
}

// SG: Delta(a,b) join E(a,x) -> (b, x) materialized; join E(b,y) -> keys x*K+y [, x != y]
static DBuf<unsigned long long> step_sg(const DBuf<unsigned long long>& delta, uint64_t nd, const CSR& E, uint32_t K,
                                        int neq, uint64_t& total) {
    DBuf<uint32_t> da(nd), dbb(nd);
    k_split_keys<<<grid_for(nd), 256>>>(delta.p, nd, K, da.p, dbb.p);
    DBuf<unsigned long long> cnt(nd), excl(nd);
    k_join_count<<<grid_for(nd), 256>>>(da.p, nd, E.off.p, cnt.p);
    exclusive_sum(cnt.p, excl.p, nd);
    uint64_t t1 = excl.get(nd - 1) + cnt.get(nd - 1);
    total = 0;
    if (!t1) return DBuf<unsigned long long>();
    DBuf<uint32_t> b1(t1), x1(t1);
    k_join_expand_cols<<<grid_for(t1), 256>>>(excl.p, nd, t1, da.p, dbb.p, E.off.p, E.val.p, b1.p, x1.p);
    da.release(); dbb.release();
    check_items(t1, "intermediate join");
    DBuf<unsigned long long> cnt2(t1), excl2(t1);
    k_join_count<<<grid_for(t1), 256>>>(b1.p, t1, E.off.p, cnt2.p);
    exclusive_sum(cnt2.p, excl2.p, t1);
    uint64_t t2 = excl2.get(t1 - 1) + cnt2.get(t1 - 1);
    DBuf<unsigned long long> out(t2);
    if (t2) k_join_expand_key<<<grid_for(t2), 256>>>(excl2.p, t1, t2, b1.p, x1.p, E.off.p, E.val.p, K, out.p);
    check_launch("step_sg");
    total = t2;
    if (neq && t2) {                                   // filter x != y
        DBuf<unsigned long long> kept(t2), nsel(1);
        size_t bytes = 0;
        CK(cub::DeviceSelect::If(nullptr, bytes, out.p, kept.p, nsel.p, t2, NeqKey{K}));
        CK(cub::DeviceSelect::If(cub_temp().get(bytes), bytes, out.p, kept.p, nsel.p, t2, NeqKey{K}));
        total = nsel.get(0);
        kept.n = total;
        return kept;
    }
    return out;
}

static void loop(Rel& rel, DBuf<unsigned long long> delta, int is_sg, const CSR& E, uint32_t K, int neq,
                 int verbose, RunResult& r) {
    uint64_t nd = delta.n;
    while (nd) {
        ++r.iters;
        double ta = now_s();
        uint64_t total = 0, nu = 0;
        DBuf<unsigned long long> cand = is_sg ? step_sg(delta, nd, E, K, neq, total) : step_tc(delta, nd, E, K, total);
        r.candidates += total;
        delta.release();
        delta = rel.close(cand, total, nu);
        nd = delta.n;
        if (verbose)
            fprintf(stderr, "  it %4d: candidates=%14llu unique=%12llu new=%12llu |full|=%llu %.4fs\n", r.iters,
                    (unsigned long long)total, (unsigned long long)nu, (unsigned long long)nd,
                    (unsigned long long)rel.nf, sync_now() - ta);
    }
}

static RunResult run_tc(const Table& g, uint32_t K, int P, int verbose, const Table* base = nullptr) {
    // base: the seed facts of another relation (CSDA), else the edges themselves (TC)
    std::vector<uint32_t> src = g.col(0), dst = g.col(1);
    std::vector<uint32_t> bsrc = base ? base->col(0) : src, bdst = base ? base->col(1) : dst;
    DBuf<uint32_t> dsrc = to_device(src), ddst = to_device(dst);
    CSR E;
    E.build(dsrc.p, ddst.p, nullptr, g.rows, K);
    RunResult r;
    for (int p = 0; p < P; ++p) {
        std::vector<unsigned long long> seed;
        for (size_t i = 0; i < bsrc.size(); ++i)
            if (P == 1 || bsrc[i] % P == (uint32_t)p) seed.push_back((unsigned long long)bsrc[i] * K + bdst[i]);
        if (seed.empty()) continue;
        Rel rel(K);
        DBuf<unsigned long long> s = to_device(seed);
        uint64_t nu = 0;
        DBuf<unsigned long long> delta = rel.close(s, seed.size(), nu);
        RunResult rp;
        loop(rel, std::move(delta), 0, E, K, 0, verbose, rp);
        r.facts += rel.nf;
        if (FINGERPRINT) r.fp += fingerprint_keys(rel.full.p, rel.nf, K);
        r.iters = rp.iters > r.iters ? rp.iters : r.iters;
        r.candidates += rp.candidates;
    }
    return r;
}

static RunResult run_sg(const Table& g, uint32_t K, int neq, int verbose) {
    std::vector<uint32_t> src = g.col(0), dst = g.col(1);  // edge (p, x): x is a child of p
    DBuf<uint32_t> dsrc = to_device(src), ddst = to_device(dst);
    CSR E;
    E.build(dsrc.p, ddst.p, nullptr, g.rows, K);
    // base rule: E(p,x) join E(p,y), x != y -> keys x*K+y
    uint64_t m = g.rows;
    if (!m) return RunResult();
    DBuf<unsigned long long> cnt(m), excl(m);
    k_join_count<<<grid_for(m), 256>>>(dsrc.p, m, E.off.p, cnt.p);
    exclusive_sum(cnt.p, excl.p, m);
    uint64_t t = excl.get(m - 1) + cnt.get(m - 1);
    DBuf<unsigned long long> out(t), kept(t), nsel(1);
    if (t) k_join_expand_key<<<grid_for(t), 256>>>(excl.p, m, t, dsrc.p, ddst.p, E.off.p, E.val.p, K, out.p);
    size_t bytes = 0;
    uint64_t nb = 0;
    if (t) {
        CK(cub::DeviceSelect::If(nullptr, bytes, out.p, kept.p, nsel.p, t, NeqKey{K}));
        CK(cub::DeviceSelect::If(cub_temp().get(bytes), bytes, out.p, kept.p, nsel.p, t, NeqKey{K}));
        nb = nsel.get(0);
    }
    out.release(); cnt.release(); excl.release();
    Rel rel(K);
    uint64_t nu = 0;
    DBuf<unsigned long long> delta = rel.close(kept, nb, nu);
    RunResult r;
    r.candidates = t;
    loop(rel, std::move(delta), 1, E, K, neq, verbose, r);
    r.facts = rel.nf;
    if (FINGERPRINT) r.fp = fingerprint_keys(rel.full.p, rel.nf, K);
    return r;
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    std::string prog = a.str("prog", "tc"), input = a.str("input");
    if (input.empty()) { fprintf(stderr, "missing --input\n"); return 1; }
    int P = (int)a.num("plan", 1), reps = (int)a.num("reps", 2), neq = (int)a.num("neq", 0);
    int verbose = (int)a.num("verbose", 0);
    FINGERPRINT = (int)a.num("fingerprint", 0);
    if (prog == "sg" && P != 1) { fprintf(stderr, "SG is not decomposable: only --plan 1\n"); return 1; }
    print_device();
    init_pool();
    Table g = read_bin(input, 2);
    uint32_t K = std::max(g.max_of(0), g.max_of(1)) + 1;
    std::string base_path = a.str("base", "");             // TC only: the seed facts from another file (CSDA)
    if (!base_path.empty() && prog != "tc") { fprintf(stderr, "--base is for TC only\n"); return 1; }
    Table b;
    if (!base_path.empty()) {
        b = read_bin(base_path, 2);
        K = std::max(K, std::max(b.max_of(0), b.max_of(1)) + 1);
    }
    const Table* bp = base_path.empty() ? nullptr : &b;
    bool warming = true;
    auto run = [&](const Table& t, uint32_t k) {
        return prog == "tc" ? run_tc(t, k, P, verbose, warming ? nullptr : bp) : run_sg(t, k, neq, verbose);
    };
    {                                                  // warm-up
        Table w;
        w.cols = 2;
        for (uint32_t i = 0; i + 1 < 2000; ++i) { w.data.push_back(i); w.data.push_back(i + 1); }
        w.rows = w.data.size() / 2;
        run(w, 2000);
        CK(cudaDeviceSynchronize());
    }
    warming = false;
    double best = 1e30;
    RunResult r;
    uint64_t peak_used = 0, peak_reserved = 0;             // GPU memory of the best run (memory pool)
    for (int i = 0; i < reps; ++i) {
        reset_peak();
        double t0 = sync_now();
        RunResult ri = run(g, K);
        double t = sync_now() - t0;
        if (i && (ri.facts != r.facts || ri.fp != r.fp)) { fprintf(stderr, "runs disagree\n"); return 4; }
        r = ri;
        if (t < best) { best = t; peak_used = pool_peak(true); peak_reserved = pool_peak(false); }
        if (P > 1) trim_pool();
    }
    Json j;
    j.add("engine", "sortmerge").add("prog", prog).num("time", best).num("result", (unsigned long long)r.facts)
     .num("iters", r.iters).num("plan", P).num("candidates", (unsigned long long)r.candidates).num("neq", neq)
     .num("K", (unsigned long long)K).num("edges", (unsigned long long)g.rows)
     .num("base_rows", (unsigned long long)(bp ? b.rows : 0));
    if (FINGERPRINT) j.num("fingerprint", r.fp);
    j.num("peak_used_gib", peak_used / 1073741824.0).num("peak_reserved_gib", peak_reserved / 1073741824.0);
    j.print();
    return 0;
}
