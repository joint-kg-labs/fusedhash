// CAS contention on one hash-table slot (Section 2 of the paper): how much it costs X threads to test the
// same key in one slot.  T = 2^24 threads in groups of X, each group on its own cell (T/X cells):
//   dup      the X threads of a group insert the SAME key into one slot (the duplicates of TC/SG): one CAS
//            wins, the others read the key back and stop;
//   collide  X DIFFERENT keys with the same start slot: linear probing in a region of pow2(2X) slots;
//   bitmap   X threads atomicOr on the same word (the same bit when X > 32).
// "adjacent": the threads of a group are consecutive (same warp/block); "spread": strided over the grid.
// Kernel time with CUDA events (median of REPS launches; the table is reset before each, outside the time).
// Output: CSV on stdout, the columns of results/a100/cas_contention.csv.
//
//   make (in engines/) && bin/cas_contention > results/cas_contention.csv
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static const unsigned EMPTY = 0xFFFFFFFFu;

__device__ __forceinline__ bool cas_insert(unsigned* t, unsigned base, unsigned R, unsigned key, unsigned& tries) {
    unsigned off = 0;
    while (true) {
        ++tries;
        unsigned prev = atomicCAS(&t[base + off], EMPTY, key);
        if (prev == EMPTY) return true;            // inserted now
        if (prev == key) return false;             // already there
        off = (off + 1) & (R - 1);                 // another key: next slot
    }
}

__global__ void contend(unsigned* t, unsigned* fresh_out, unsigned* tries_out, unsigned X, unsigned G, unsigned R,
                        unsigned mode, unsigned spread, unsigned T) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T) return;
    unsigned g = spread ? tid % G : tid / X;
    unsigned r = spread ? tid / G : tid % X;
    unsigned fresh = 0, tries = 0;
    unsigned cell = g;
    if (mode == 2) {
        unsigned bit = 1u << (r & 31);
        unsigned old = atomicOr(&t[cell], bit);
        fresh += (old & bit) == 0;
        ++tries;
    } else {
        unsigned key = mode == 0 ? cell : cell * X + r;
        fresh += cas_insert(t, cell * R, R, key, tries);
    }
    fresh_out[tid] = fresh;
    tries_out[tid] = tries;
}

static unsigned pow2(unsigned n) { unsigned p = 1; while (p < n) p <<= 1; return p; }

int main(int argc, char** argv) {
    const unsigned T = 1u << 24;
    const int REPS = 7;
    const char* modes[] = {"dup", "collide", "bitmap"};
    const char* places[] = {"adjacent", "spread"};
    std::vector<unsigned> Xs = {1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096, 65536, T};
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    fprintf(stderr, "GPU: %s\n", p.name);
    unsigned *t, *fresh, *tries;
    size_t max_slots = (size_t)T * 2;              // collide with X <= 1024: G * pow2(2X) = 2T slots
    CK(cudaMalloc(&t, max_slots * 4)); CK(cudaMalloc(&fresh, (size_t)T * 4)); CK(cudaMalloc(&tries, (size_t)T * 4));
    std::vector<unsigned> hf(T), ht(T);
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    printf("experiment,mode,placement,X,threads,cells,rounds,time_ms,us_per_round,ns_per_test,tries_per_test,fresh,expected_fresh\n");
    int bad = 0;
    for (unsigned m = 0; m < 3; ++m)
        for (unsigned s = 0; s < 2; ++s)
            for (unsigned X : Xs) {
                if (m == 1 && X > 1024) continue;
                unsigned G = T / X;
                unsigned R = m == 1 ? pow2(2 * X) : 1;
                size_t slots = (size_t)G * R;
                std::vector<float> ms;
                for (int k = 0; k < REPS + 1; ++k) {   // the first launch is a warm-up
                    CK(cudaMemset(t, m == 2 ? 0x00 : 0xFF, slots * 4));
                    CK(cudaDeviceSynchronize());
                    CK(cudaEventRecord(a));
                    contend<<<(T + 255) / 256, 256>>>(t, fresh, tries, X, G, R, m, s, T);
                    CK(cudaEventRecord(b));
                    CK(cudaEventSynchronize(b));
                    CK(cudaGetLastError());
                    float x; CK(cudaEventElapsedTime(&x, a, b));
                    if (k) ms.push_back(x);
                }
                std::sort(ms.begin(), ms.end());
                float med = ms[ms.size() / 2];
                CK(cudaMemcpy(hf.data(), fresh, (size_t)T * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(ht.data(), tries, (size_t)T * 4, cudaMemcpyDeviceToHost));
                unsigned long long f = 0, k = 0;
                for (unsigned i = 0; i < T; ++i) { f += hf[i]; k += ht[i]; }
                unsigned long long want = m == 0 ? G : m == 1 ? (unsigned long long)G * X : (unsigned long long)G * std::min(X, 32u);
                bad += f != want;
                printf("throughput,%s,%s,%u,%u,%u,1,%.3f,,%.3f,%.2f,%llu,%llu\n", modes[m], places[s], X, T, G, med,
                       med * 1e6 / T, (double)k / T, f, want);
                fflush(stdout);
            }
    fprintf(stderr, "wrong counts: %d\n", bad);
    return bad ? 1 : 0;
}
