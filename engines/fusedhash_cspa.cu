// FusedHash for CSPA (context-sensitive points-to analysis, rules of GDlog) on NVIDIA GPUs
// (appendix of the paper: non-linear recursion).
//
//   VF(y,x) <- assign(y,x).         VF(x,x) <- assign(x,_).   VF(x,x) <- assign(_,x).
//   MA(x,x) <- assign(_,x).         MA(x,x) <- assign(x,_).
//   r1 VF(x,y) <- VF(x,z), VF(z,y).                  r2 VA(x,y) <- VF(z,x), VF(z,y).
//   r3 VF(x,y) <- assign(x,z), MA(z,y).              r4 MA(x,w) <- deref(y,x), VA(y,z), deref(z,w).
//   r5 VA(x,y) <- VF(z,x), MA(z,w), VF(w,y).
//
// Each relation has a hash table with 64-bit keys (x<<32)|y and an append-only full. VF (small) is
// indexed by CSR rebuilt at each iteration, by column 1 for the facts before Delta (old) and for
// Delta, by column 2 for the old facts; MA (large) has lock-free adjacency lists per column (head[k] =
// latest fact with key k, next[i] = previous fact with the same key), linked by a separate kernel at
// the end of each iteration. Semi-naive variants with each derivation made once (the atoms before the
// one read as Delta read old; r2, symmetric, is one variant emitting both orientations), one fused
// kernel each; r5 with Delta on MA, whose work is the product of two degrees, runs one thread per
// candidate. Tables grow optimistically: an insertion that finds the table or full full gives up and
// raises a flag (a claimed slot without room in full becomes a tombstone); the host enlarges and
// runs the same chunk again, which is idempotent.
//
// Usage: fusedhash_cspa --input <dir with assign.bin, dereference.bin> [--reps 2] [--verbose 0]
#include "common.cuh"
#include <functional>

constexpr unsigned long long EMPTY64 = 0xFFFFFFFFFFFFFFFFull;
constexpr unsigned long long TOMB64 = 0xFFFFFFFFFFFFFFFEull;
constexpr uint32_t NIL = 0xFFFFFFFFu;
constexpr double MAX_LOAD = 0.8;

struct Tgt {                                        // target relation of a kernel
    unsigned long long* keys;
    uint32_t *fx, *fy;
    unsigned long long *size, *nkeys;
    uint32_t* ovf;
    uint64_t mask, capf, maxkeys;
};

__device__ __forceinline__ void emit(const Tgt& t, uint32_t x, uint32_t y) {
    unsigned long long key = ((unsigned long long)x << 32) | y;
    uint64_t h = hash64(key) & t.mask;
    while (true) {
        unsigned long long cur = *((volatile unsigned long long*)&t.keys[h]);
        if (cur == EMPTY64) {
            if (*((volatile unsigned long long*)t.nkeys) >= t.maxkeys) { *t.ovf = 1; return; }
            unsigned long long prev = atomicCAS(&t.keys[h], EMPTY64, key);
            if (prev == EMPTY64) {
                atomicAdd(t.nkeys, 1ull);
                unsigned long long p = atomicAdd(t.size, 1ull);
                if (p < t.capf) { t.fx[p] = x; t.fy[p] = y; }
                else { t.keys[h] = TOMB64; *t.ovf = 1; }      // no room in full: undo, retried later
                return;
            }
            cur = prev;
        }
        if (cur == key) return;                              // already present
        h = (h + 1) & t.mask;                                // another key or a tombstone
    }
}

__global__ void k_rehash(unsigned long long* keys, uint64_t mask, const uint32_t* fx, const uint32_t* fy, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long key = ((unsigned long long)fx[i] << 32) | fy[i];
    uint64_t h = hash64(key) & mask;
    while (atomicCAS(&keys[h], EMPTY64, key) != EMPTY64) h = (h + 1) & mask;
}

__global__ void k_insert_pairs(Tgt t, const uint32_t* px, const uint32_t* py, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) emit(t, px[i], py[i]);
}

// the base rules on the GPU: the ids that occur in assign, VF(x,x) and MA(x,x) for each of them
__global__ void k_mark(const uint32_t* v, uint64_t n, uint8_t* seen) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) seen[v[i]] = 1;
}

__global__ void k_insert_diag(Tgt t, const uint8_t* seen, uint32_t K) {
    uint64_t v = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (v < K && seen[v]) emit(t, (uint32_t)v, (uint32_t)v);
}

__global__ void k_off_deg(const uint32_t* off, uint32_t K, uint32_t* deg) {   // degrees of a CSR
    uint64_t k = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (k < K) deg[k] = off[k + 1] - off[k];
}

__global__ void k_link(const uint32_t* fx, const uint32_t* fy, uint32_t* h1, uint32_t* n1, uint32_t* d1, uint32_t* h2,
                       uint32_t* n2, uint32_t* d2, uint64_t a, uint64_t b) {
    uint64_t i = a + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= b) return;
    uint32_t x = fx[i], y = fy[i];
    n1[i] = atomicExch(&h1[x], (uint32_t)i);
    atomicAdd(&d1[x], 1u);
    n2[i] = atomicExch(&h2[y], (uint32_t)i);
    atomicAdd(&d2[y], 1u);
}

// ---- VF is indexed by CSR, rebuilt at each iteration (VF is small: a few million facts): by its first
// column for the facts before Delta (old, o1) and for Delta (d1), by its second column for the old
// facts (o2). Full VF by column 1 = o1 then d1. Each derivation is made once: in a rule with two or
// three intensional atoms, the atoms before the one read as Delta read old (the semi-naive split old /
// Delta); the symmetric r2 runs as one variant that emits both orientations.
struct VFIdx {
    const uint32_t *o1off, *o1val, *d1off, *d1val, *o2off, *o2val;
};

__device__ __forceinline__ uint32_t dgr(const uint32_t* off, uint32_t k) { return off[k + 1] - off[k]; }

// r1, Delta first: VF(x,y) <- Delta VF(x,z), VF(z,y)
__global__ void k_r1a(Tgt t, const uint32_t* sx, const uint32_t* sy, uint64_t a, uint64_t b, VFIdx v) {
    uint64_t i0 = a + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i0 >= b) return;
    uint32_t x = sx[i0], z = sy[i0];
    for (uint32_t j = v.o1off[z], e = v.o1off[z + 1]; j < e; ++j) emit(t, x, v.o1val[j]);
    for (uint32_t j = v.d1off[z], e = v.d1off[z + 1]; j < e; ++j) emit(t, x, v.d1val[j]);
}

// r1, Delta second: VF(x,y) <- old VF(x,z), Delta VF(z,y)  (old by column 2: z -> x)
__global__ void k_r1b(Tgt t, const uint32_t* sx, const uint32_t* sy, uint64_t a, uint64_t b, VFIdx v) {
    uint64_t i0 = a + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i0 >= b) return;
    uint32_t z = sx[i0], y = sy[i0];
    for (uint32_t j = v.o2off[z], e = v.o2off[z + 1]; j < e; ++j) emit(t, v.o2val[j], y);
}

// the Delta tuple k of the candidate g (incl: inclusive sums of the candidates of the tuples) and its rank r
__device__ __forceinline__ uint64_t owner_rank(const unsigned long long* incl, uint64_t nd, uint64_t g, uint64_t& r) {
    uint64_t lo = 0, hi = nd;
    while (lo < hi) { uint64_t m = (lo + hi) >> 1; if (incl[m] > g) hi = m; else lo = m + 1; }
    r = g - (lo ? incl[lo - 1] : 0);
    return lo;
}

// r2, one thread per candidate: Delta VF(z,x) has |old VF(z)| + |Delta VF(z)| candidates; an old y gives
// (x,y) and (y,x), a Delta y gives (x,y)
__global__ void k_r2_out(Tgt t, const uint32_t* sx, const uint32_t* sy, uint64_t d0, const unsigned long long* incl,
                         uint64_t nd, uint64_t g0, uint64_t g1, VFIdx v) {
    uint64_t g = g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (g >= g1) return;
    uint64_t r;
    uint64_t k = owner_rank(incl, nd, g, r);
    uint32_t z = sx[d0 + k], x = sy[d0 + k];
    uint32_t zo = dgr(v.o1off, z);
    if (r < zo) { uint32_t y = v.o1val[v.o1off[z] + r]; emit(t, x, y); emit(t, y, x); }
    else emit(t, x, v.d1val[v.d1off[z] + (r - zo)]);
}

// r4, one thread per candidate: Delta VA(y,z) has |deref(y)| * |deref(z)| candidates (x, w)
__global__ void k_r4_out(Tgt t, const uint32_t* sx, const uint32_t* sy, uint64_t d0, const unsigned long long* incl,
                         uint64_t nd, uint64_t g0, uint64_t g1, const uint32_t* doff, const uint32_t* dval) {
    uint64_t g = g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (g >= g1) return;
    uint64_t r;
    uint64_t k = owner_rank(incl, nd, g, r);
    uint32_t y = sx[d0 + k], z = sy[d0 + k];
    uint32_t nz = doff[z + 1] - doff[z];
    emit(t, dval[doff[y] + (uint32_t)(r / nz)], dval[doff[z] + (uint32_t)(r % nz)]);
}

// r3: VF(x,y) <- assign(x,z), Delta MA(z,y): CSR of assign by its second column (z -> x)
__global__ void k_assign_ma(Tgt t, const uint32_t* sx, const uint32_t* sy, uint64_t a, uint64_t b, const uint32_t* aoff,
                            const uint32_t* aval) {
    uint64_t i0 = a + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i0 >= b) return;
    uint32_t z = sx[i0], y = sy[i0];
    for (uint32_t j = aoff[z], e = aoff[z + 1]; j < e; ++j) emit(t, aval[j], y);
}

// r5: VA(x,y) <- VF(z,x), MA(z,w), VF(w,y)
// Delta on VF(z,x): MA(z,w) full (list of MA by column 1), VF(w,y) full
__global__ void k_r5a(Tgt t, const uint32_t* sx, const uint32_t* sy, uint64_t a, uint64_t b, const uint32_t* mh1,
                      const uint32_t* mn1, const uint32_t* mfy, VFIdx v) {
    uint64_t i0 = a + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i0 >= b) return;
    uint32_t z = sx[i0], x = sy[i0];
    for (uint32_t i = mh1[z]; i != NIL; i = mn1[i]) {
        uint32_t w = mfy[i];
        for (uint32_t j = v.o1off[w], e = v.o1off[w + 1]; j < e; ++j) emit(t, x, v.o1val[j]);
        for (uint32_t j = v.d1off[w], e = v.d1off[w + 1]; j < e; ++j) emit(t, x, v.d1val[j]);
    }
}

// Delta on MA(z,w): VF(z,x) old, VF(w,y) full; one thread per candidate (x, y): the candidate g of the
// chunk belongs to the Delta tuple k whose range [incl[k-1], incl[k]) holds it, and is the pair
// (i, j) = (r / |VF(w)|, r % |VF(w)|) of the old successors of z and the successors of w (old then Delta)
__global__ void k_r5b(Tgt t, const uint32_t* mx, const uint32_t* my, uint64_t d0, const unsigned long long* incl,
                      uint64_t nd, uint64_t g0, uint64_t g1, VFIdx v) {
    uint64_t g = g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (g >= g1) return;
    uint64_t lo = 0, hi = nd;
    while (lo < hi) { uint64_t m = (lo + hi) >> 1; if (incl[m] > g) hi = m; else lo = m + 1; }
    uint64_t r = g - (lo ? incl[lo - 1] : 0);
    uint32_t z = mx[d0 + lo], w = my[d0 + lo];
    uint32_t wo = dgr(v.o1off, w), wn = wo + dgr(v.d1off, w);
    uint32_t i = (uint32_t)(r / wn), j = (uint32_t)(r % wn);
    uint32_t x = v.o1val[v.o1off[z] + i];
    uint32_t y = j < wo ? v.o1val[v.o1off[w] + j] : v.d1val[v.d1off[w] + j - wo];
    emit(t, x, y);
}

// Delta on VF(w,y): MA(z,w) old (list of MA by column 2: w -> z, positions before its Delta), VF(z,x) old
__global__ void k_r5c(Tgt t, const uint32_t* sx, const uint32_t* sy, uint64_t a, uint64_t b, const uint32_t* mh2,
                      const uint32_t* mn2, const uint32_t* mfx, uint64_t md0, VFIdx v) {
    uint64_t i0 = a + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i0 >= b) return;
    uint32_t w = sx[i0], y = sy[i0];
    for (uint32_t i = mh2[w]; i != NIL; i = mn2[i]) {
        if (i >= md0) continue;
        uint32_t z = mfx[i];
        for (uint32_t j = v.o1off[z], e = v.o1off[z + 1]; j < e; ++j) emit(t, v.o1val[j], y);
    }
}

// ---- work per tuple of Delta (to split Delta into chunks of bounded work, and r5 into candidates)
__global__ void k_c_full1(const uint32_t* keys, uint64_t d0, uint64_t nd, VFIdx v, unsigned long long* out) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;      // |full VF(k, _)|, k = keys[d0 + t]
    if (t < nd) { uint32_t k = keys[d0 + t]; out[t] = dgr(v.o1off, k) + dgr(v.d1off, k); }
}

__global__ void k_c_old2(const uint32_t* keys, uint64_t d0, uint64_t nd, VFIdx v, unsigned long long* out) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;      // |old VF(_, k)|
    if (t < nd) out[t] = dgr(v.o2off, keys[d0 + t]);
}

__global__ void k_c_deg(const uint32_t* keys, uint64_t d0, uint64_t nd, const uint32_t* deg, unsigned long long* out) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t < nd) out[t] = deg[keys[d0 + t]];
}

__global__ void k_c_prod(const uint32_t* kx, const uint32_t* ky, uint64_t d0, uint64_t nd, const uint32_t* dx,
                         const uint32_t* dy, unsigned long long* out) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t < nd) out[t] = (unsigned long long)dx[kx[d0 + t]] * dy[ky[d0 + t]];
}

__global__ void k_c_r5a(const uint32_t* keys, uint64_t d0, uint64_t nd, const uint32_t* mh1, const uint32_t* mn1,
                        const uint32_t* mfy, VFIdx v, unsigned long long* out) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= nd) return;
    unsigned long long s = 0;
    for (uint32_t i = mh1[keys[d0 + t]]; i != NIL; i = mn1[i]) { uint32_t w = mfy[i]; s += dgr(v.o1off, w) + dgr(v.d1off, w); }
    out[t] = s;
}

__global__ void k_c_r5b(const uint32_t* mx, const uint32_t* my, uint64_t d0, uint64_t nd, VFIdx v, unsigned long long* out) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;      // |old VF(z, _)| * |full VF(w, _)|
    if (t >= nd) return;
    uint32_t z = mx[d0 + t], w = my[d0 + t];
    out[t] = (unsigned long long)dgr(v.o1off, z) * (dgr(v.o1off, w) + dgr(v.d1off, w));
}

__global__ void k_c_r5c(const uint32_t* keys, uint64_t d0, uint64_t nd, const uint32_t* mh2, const uint32_t* mn2,
                        const uint32_t* mfx, uint64_t md0, VFIdx v, unsigned long long* out) {
    uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= nd) return;
    unsigned long long s = 0;
    for (uint32_t i = mh2[keys[d0 + t]]; i != NIL; i = mn2[i]) if (i < md0) s += dgr(v.o1off, mfx[i]);
    out[t] = s;
}

__device__ __forceinline__ uint64_t owner_incl(const unsigned long long* incl, uint64_t a, uint64_t b, uint64_t g) {
    uint64_t lo = a, hi = b;
    while (lo < hi) { uint64_t m = (lo + hi) >> 1; if (incl[m] > g) hi = m; else lo = m + 1; }
    return lo;
}

__global__ void k_cuts(const unsigned long long* incl, uint64_t nd, uint64_t chunk, uint64_t ncuts,
                       unsigned long long* cut) {
    uint64_t k = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (k < ncuts) cut[k] = owner_incl(incl, 0, nd, (k + 1) * chunk);
}

// ------------------------------------------------------------------ relations

static uint64_t pow2(uint64_t n) { uint64_t p = 1ull << 20; while (p < n) p <<= 1; return p; }

struct Rel {
    const char* name;
    uint32_t K;
    bool lists;
    DBuf<unsigned long long> keys, size, nkeys;
    DBuf<uint32_t> fx, fy, ovf, h1, h2, d1, d2, n1, n2;
    uint64_t cap_tab = 1ull << 20, cap_full = 1ull << 20, F = 0, nk = 0, d0 = 0, linked = 0;

    Rel(const char* n, uint32_t K_, bool l) : name(n), K(K_), lists(l) {
        keys.alloc(cap_tab); keys.fill_bytes(0xFF);
        fx.alloc(cap_full); fy.alloc(cap_full);
        size.alloc(1); size.zero(); nkeys.alloc(1); nkeys.zero(); ovf.alloc(1); ovf.zero();
        if (lists) {
            h1.alloc(K); h1.fill_bytes(0xFF); h2.alloc(K); h2.fill_bytes(0xFF);
            d1.alloc(K); d1.zero(); d2.alloc(K); d2.zero();
            n1.alloc(cap_full); n2.alloc(cap_full);
        }
    }
    Tgt tgt() {
        return Tgt{keys.p, fx.p, fy.p, size.p, nkeys.p, ovf.p, cap_tab - 1, cap_full, (uint64_t)(MAX_LOAD * cap_tab)};
    }
    void sync() {
        F = std::min<uint64_t>(size.get(0), cap_full);
        size.set(0, F);                              // after an overflow size can exceed cap_full
        nk = nkeys.get(0);
    }
    void grow(uint64_t need) {                       // room for `need` facts (with a 2x margin)
        if (need > cap_full) {
            uint64_t c = pow2(2 * need);
            fx.grow(c, F); fy.grow(c, F);
            if (lists) { n1.grow(c, F); n2.grow(c, F); }
            cap_full = c;
        }
        // a larger table when the facts need it, and also when the table is near its limit of keys (the
        // tombstones of insertions without room in full count as keys until a rehash removes them)
        if (need > 0.5 * MAX_LOAD * cap_tab || nk >= 0.9 * MAX_LOAD * cap_tab) {
            uint64_t f = cap_tab < (1ull << 28) ? 4 : 2;     // fewer rehashes while the table is small
            cap_tab = pow2((uint64_t)(f * need / MAX_LOAD) + 1);
            keys.alloc(cap_tab);
            keys.fill_bytes(0xFF);
            if (F) { k_rehash<<<grid_for(F), 256>>>(keys.p, cap_tab - 1, fx.p, fy.p, F); check_launch("k_rehash"); }
            nkeys.set(0, F);                         // tombstones disappear with the rehash
            nk = F;
        }
    }
    void link() {                                    // links full[linked, F) to the lists
        if (lists && F > linked) {
            k_link<<<grid_for(F - linked), 256>>>(fx.p, fy.p, h1.p, n1.p, d1.p, h2.p, n2.p, d2.p, linked, F);
            check_launch("k_link");
        }
        linked = F;
    }
    // launches a kernel towards this relation, growing and repeating on overflow
    void run(uint64_t n_est, const std::function<void()>& launch) {
        grow(F + n_est);
        while (true) {
            ovf.zero();
            launch();
            check_launch(name);
            sync();
            if (!ovf.get(0)) return;
            grow(F >= cap_full ? std::max<uint64_t>(2 * F, cap_full) : F + std::max<uint64_t>(F, 1ull << 20));
        }
    }
};

struct Sizes { uint64_t vf = 0, va = 0, ma = 0; int iters = 0; uint64_t candidates = 0;
               unsigned long long fp_vf = 0, fp_va = 0, fp_ma = 0; };   // fingerprints of the relations
static int FINGERPRINT = 0;                         // --fingerprint=1 (tests)

static Sizes cspa(const Table& asg, const Table& der, uint64_t chunk, int verbose) {
    uint32_t K = std::max({asg.max_of(0), asg.max_of(1), der.max_of(0), der.max_of(1)}) + 1;
    std::vector<uint32_t> a0 = asg.col(0), a1 = asg.col(1), d0v = der.col(0), d1v = der.col(1);
    DBuf<uint32_t> da0 = to_device(a0), da1 = to_device(a1), dd0 = to_device(d0v), dd1 = to_device(d1v);
    CSR A2, D1;                                      // assign(x,z): z -> x;  deref(y,x): y -> x
    A2.build(da1.p, da0.p, nullptr, asg.rows, K);
    D1.build(dd0.p, dd1.p, nullptr, der.rows, K);
    DBuf<uint32_t> a_deg2(K), d_deg1(K);
    auto degrees = [&](const CSR& c, DBuf<uint32_t>& deg) {
        k_off_deg<<<grid_for(K), 256>>>(c.off.p, K, deg.p);
        check_launch("k_off_deg");
    };
    degrees(A2, a_deg2);
    degrees(D1, d_deg1);
    Rel VF("VF", K, false), VA("VA", K, false), MA("MA", K, true);

    // base rules: VF(y,x) <- assign(y,x); VF(x,x), MA(x,x) for the ids of assign (on the GPU)
    {
        DBuf<uint8_t> seen(K);
        seen.zero();
        if (asg.rows) {
            k_mark<<<grid_for(asg.rows), 256>>>(da0.p, asg.rows, seen.p);
            k_mark<<<grid_for(asg.rows), 256>>>(da1.p, asg.rows, seen.p);
            check_launch("k_mark");
        }
        VF.run(asg.rows + K, [&] {
            if (asg.rows) k_insert_pairs<<<grid_for(asg.rows), 256>>>(VF.tgt(), da0.p, da1.p, asg.rows);
            k_insert_diag<<<grid_for(K), 256>>>(VF.tgt(), seen.p, K); });
        MA.run(K, [&] { k_insert_diag<<<grid_for(K), 256>>>(MA.tgt(), seen.p, K); });
    }
    VF.link();
    MA.link();

    Sizes res;
    // one semi-naive variant: Delta of src = [d0, F0); count(d0, nd, out) writes the candidates of
    // each tuple; step(a, b) launches the kernel on Delta positions [a, b) (absolute)
    auto variant = [&](Rel& src, uint64_t sd0, uint64_t sF, Rel& tgt,
                       const std::function<void(uint64_t, uint64_t, unsigned long long*)>& count,
                       const std::function<void(uint64_t, uint64_t)>& step) {
        uint64_t nd = sF - sd0;
        if (!nd) return;
        check_items(nd, "delta");
        DBuf<unsigned long long> cnt(nd), incl(nd);
        count(sd0, nd, cnt.p);
        inclusive_sum(cnt.p, incl.p, nd);
        uint64_t total = incl.get(nd - 1);
        res.candidates += total;
        if (!total) return;
        std::vector<uint64_t> cuts{0};
        if (total > chunk) {
            uint64_t nc = (total - 1) / chunk;
            DBuf<unsigned long long> dc(nc);
            k_cuts<<<grid_for(nc), 256>>>(incl.p, nd, chunk, nc, dc.p);
            check_launch("k_cuts");
            std::vector<unsigned long long> h(nc);
            CK(cudaMemcpy(h.data(), dc.p, nc * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            for (auto c : h) if (c > cuts.back() && c < nd) cuts.push_back(c);
        }
        cuts.push_back(nd);
        for (size_t k = 0; k + 1 < cuts.size(); ++k) {
            uint64_t a = sd0 + cuts[k], b = sd0 + cuts[k + 1];
            tgt.run(1ull << 20, [&] { step(a, b); });
        }
    };
    // a variant with one thread per candidate: launch(g0, g1, incl, nd) runs the candidates [g0, g1)
    auto variant_out = [&](uint64_t sd0, uint64_t sF, Rel& tgt,
                           const std::function<void(uint64_t, uint64_t, unsigned long long*)>& count,
                           const std::function<void(uint64_t, uint64_t, const unsigned long long*, uint64_t)>& launch) {
        uint64_t nd = sF - sd0;
        if (!nd) return;
        check_items(nd, "delta");
        DBuf<unsigned long long> cnt(nd), incl(nd);
        count(sd0, nd, cnt.p);
        inclusive_sum(cnt.p, incl.p, nd);
        uint64_t total = incl.get(nd - 1);
        res.candidates += total;
        for (uint64_t g0 = 0; g0 < total; g0 += chunk) {
            uint64_t g1 = std::min(total, g0 + chunk);
            tgt.run(1ull << 20, [&] { launch(g0, g1, incl.p, nd); });
        }
    };
    auto G = [](uint64_t a, uint64_t b) { return grid_for(b - a); };

    while (true) {
        uint64_t vd0 = VF.d0, vF = VF.F, ad0 = VA.d0, aF = VA.F, md0 = MA.d0, mF = MA.F;
        if (vd0 >= vF && ad0 >= aF && md0 >= mF) break;
        ++res.iters;
        double ta = now_s();
        // the indexes of VF for this iteration: old = [0, vd0), Delta = [vd0, vF)
        CSR O1, D1v, O2;
        O1.build(VF.fx.p, VF.fy.p, nullptr, vd0, K);
        D1v.build(VF.fx.p + vd0, VF.fy.p + vd0, nullptr, vF - vd0, K);
        O2.build(VF.fy.p, VF.fx.p, nullptr, vd0, K);
        VFIdx vi{O1.off.p, O1.val.p, D1v.off.p, D1v.val.p, O2.off.p, O2.val.p};
        double tv = sync_now();                     // --verbose 2: the time of each variant
        auto tick = [&](const char* what) { if (verbose > 1) { double t = sync_now(); fprintf(stderr, "    %-10s %.4fs\n", what, t - tv); tv = t; } };
        // r1, Delta first: VF(x,y) <- Delta VF(x,z), VF(z,y)
        variant(VF, vd0, vF, VF, [&](uint64_t d, uint64_t n, unsigned long long* o) {
            k_c_full1<<<grid_for(n), 256>>>(VF.fy.p, d, n, vi, o); },
            [&](uint64_t a, uint64_t b) { k_r1a<<<G(a, b), 256>>>(VF.tgt(), VF.fx.p, VF.fy.p, a, b, vi); });
        tick("r1a");
        // r1, Delta second: VF(x,y) <- old VF(x,z), Delta VF(z,y)
        variant(VF, vd0, vF, VF, [&](uint64_t d, uint64_t n, unsigned long long* o) {
            k_c_old2<<<grid_for(n), 256>>>(VF.fx.p, d, n, vi, o); },
            [&](uint64_t a, uint64_t b) { k_r1b<<<G(a, b), 256>>>(VF.tgt(), VF.fx.p, VF.fy.p, a, b, vi); });
        tick("r1b");
        // r3: VF(x,y) <- assign(x,z), Delta MA(z,y)
        variant(MA, md0, mF, VF, [&](uint64_t d, uint64_t n, unsigned long long* o) {
            k_c_deg<<<grid_for(n), 256>>>(MA.fx.p, d, n, a_deg2.p, o); },
            [&](uint64_t a, uint64_t b) { k_assign_ma<<<G(a, b), 256>>>(VF.tgt(), MA.fx.p, MA.fy.p, a, b, A2.off.p, A2.val.p); });
        tick("r3");
        // r2: VA(x,y) <- VF(z,x), VF(z,y): one variant on Delta VF(z,x), both orientations; one thread per candidate
        variant_out(vd0, vF, VA, [&](uint64_t d, uint64_t n, unsigned long long* o) {
            k_c_full1<<<grid_for(n), 256>>>(VF.fx.p, d, n, vi, o); },
            [&](uint64_t g0, uint64_t g1, const unsigned long long* incl, uint64_t nd) {
                k_r2_out<<<grid_for(g1 - g0), 256>>>(VA.tgt(), VF.fx.p, VF.fy.p, vd0, incl, nd, g0, g1, vi); });
        tick("r2");
        // r5, Delta on VF(z,x): MA full, VF full
        variant(VF, vd0, vF, VA, [&](uint64_t d, uint64_t n, unsigned long long* o) {
            k_c_r5a<<<grid_for(n), 256>>>(VF.fx.p, d, n, MA.h1.p, MA.n1.p, MA.fy.p, vi, o); },
            [&](uint64_t a, uint64_t b) { k_r5a<<<G(a, b), 256>>>(VA.tgt(), VF.fx.p, VF.fy.p, a, b, MA.h1.p, MA.n1.p, MA.fy.p, vi); });
        tick("r5a");
        // r5, Delta on MA(z,w): VF old, VF full; one thread per candidate
        variant_out(md0, mF, VA, [&](uint64_t d, uint64_t n, unsigned long long* o) {
            k_c_r5b<<<grid_for(n), 256>>>(MA.fx.p, MA.fy.p, d, n, vi, o); },
            [&](uint64_t g0, uint64_t g1, const unsigned long long* incl, uint64_t nd) {
                k_r5b<<<grid_for(g1 - g0), 256>>>(VA.tgt(), MA.fx.p, MA.fy.p, md0, incl, nd, g0, g1, vi); });
        tick("r5b");
        // r5, Delta on VF(w,y): VF old, MA old
        variant(VF, vd0, vF, VA, [&](uint64_t d, uint64_t n, unsigned long long* o) {
            k_c_r5c<<<grid_for(n), 256>>>(VF.fx.p, d, n, MA.h2.p, MA.n2.p, MA.fx.p, md0, vi, o); },
            [&](uint64_t a, uint64_t b) { k_r5c<<<G(a, b), 256>>>(VA.tgt(), VF.fx.p, VF.fy.p, a, b, MA.h2.p, MA.n2.p, MA.fx.p, md0, vi); });
        tick("r5c");
        // r4: MA(x,w) <- deref(y,x), Delta VA(y,z), deref(z,w)
        variant_out(ad0, aF, MA, [&](uint64_t d, uint64_t n, unsigned long long* o) {
            k_c_prod<<<grid_for(n), 256>>>(VA.fx.p, VA.fy.p, d, n, d_deg1.p, d_deg1.p, o); },
            [&](uint64_t g0, uint64_t g1, const unsigned long long* incl, uint64_t nd) {
                k_r4_out<<<grid_for(g1 - g0), 256>>>(MA.tgt(), VA.fx.p, VA.fy.p, ad0, incl, nd, g0, g1, D1.off.p, D1.val.p); });
        tick("r4");
        // end of the iteration: next Deltas, and new facts linked into the lists
        VF.d0 = vF; VA.d0 = aF; MA.d0 = mF;
        MA.link();
        if (verbose) fprintf(stderr, "  it %3d: VF=%11llu VA=%12llu MA=%11llu %.3fs\n", res.iters,
                             (unsigned long long)VF.F, (unsigned long long)VA.F, (unsigned long long)MA.F,
                             sync_now() - ta);
    }
    res.vf = VF.F; res.va = VA.F; res.ma = MA.F;
    if (FINGERPRINT) {
        res.fp_vf = fingerprint(VF.fx.p, VF.fy.p, VF.F);
        res.fp_va = fingerprint(VA.fx.p, VA.fy.p, VA.F);
        res.fp_ma = fingerprint(MA.fx.p, MA.fy.p, MA.F);
    }
    return res;
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    std::string dir = a.str("input");
    if (dir.empty()) { fprintf(stderr, "missing --input\n"); return 1; }
    int reps = (int)a.num("reps", 2), verbose = (int)a.num("verbose", 0);
    uint64_t chunk = (uint64_t)a.num("chunk", 1ll << 27);
    FINGERPRINT = (int)a.num("fingerprint", 0);
    print_device();
    init_pool();
    Table asg = read_bin(dir + "/assign.bin", 2), der = read_bin(dir + "/dereference.bin", 2);
    {                                                // warm-up on a small program
        Table wa, wd;
        wa.cols = wd.cols = 2;
        for (uint32_t i = 0; i < 1000; ++i) { wa.data.push_back(i); wa.data.push_back((i * 7) % 1000);
                                              wd.data.push_back(i); wd.data.push_back((i * 13) % 1000); }
        wa.rows = wd.rows = 1000;
        cspa(wa, wd, chunk, 0);
        CK(cudaDeviceSynchronize());
    }
    double best = 1e30;
    Sizes r;
    uint64_t peak_used = 0, peak_reserved = 0;             // GPU memory of the best run (memory pool)
    for (int i = 0; i < reps; ++i) {
        reset_peak();
        double t0 = sync_now();
        Sizes ri = cspa(asg, der, chunk, verbose);
        double el = sync_now() - t0;
        if (i && (ri.vf != r.vf || ri.va != r.va || ri.ma != r.ma || ri.fp_vf != r.fp_vf || ri.fp_va != r.fp_va
                   || ri.fp_ma != r.fp_ma)) { fprintf(stderr, "runs disagree\n"); return 4; }
        r = ri;
        if (el < best) { best = el; peak_used = pool_peak(true); peak_reserved = pool_peak(false); }
    }
    Json j;
    j.add("engine", "fusedhash").add("prog", "cspa").num("time", best).num("VF", (unsigned long long)r.vf)
     .num("VA", (unsigned long long)r.va).num("MA", (unsigned long long)r.ma)
     .num("result", (unsigned long long)(r.vf + r.va + r.ma)).num("iters", r.iters)
     .num("candidates", (unsigned long long)r.candidates);
    if (FINGERPRINT) j.num("fp_VF", r.fp_vf).num("fp_VA", r.fp_va).num("fp_MA", r.fp_ma);
    j.num("peak_used_gib", peak_used / 1073741824.0).num("peak_reserved_gib", peak_reserved / 1073741824.0);
    j.print();
    return 0;
}
