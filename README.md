# FusedHash: Datalog Evaluation on GPUs with Fused Hash Kernels

This repository contains the code, the experiment scripts and the measured results of the paper

> L. Bellomarini, M. Brandetti, E. Sallinger.
> *Accelerating Datalog Evaluation on GPUs with Fused Hash Kernels.*

The appendix of the paper (CSPA, CSDA, threshold of the looping kernel) is available
[here](https://drive.google.com/file/d/1F8RRa8t0YIL6Yyq88iiH6p0N0yrBWU4R/view?usp=sharing).

**FusedHash** is a GPU algorithm for semi-naive Datalog evaluation that runs rule application and set
maintenance in a single kernel. Each candidate fact stays in the registers of the thread that produced
it and is inserted into an open-addressed hash table with one lock-free compare-and-swap. A successful
insertion identifies a new fact, which the thread appends to the accumulated relation; the appended
facts form the next delta. No candidate relation is materialized, sorted or merged, so the work of an
iteration depends on the candidates it generates rather than on the size of the accumulated relation.

The same principle covers recursion through monotonic aggregation (atomic minimum for CC and SSSP,
atomic addition with a threshold for company control). Adaptive optimizations switch to a bitmap for
dense relations, evaluate dense deltas with joins on 32-bit words, keep deep recursions on the GPU
with a looping kernel, and evaluate homomorphically decomposable programs one partition at a time.

## Main results

All systems run on the same machine: an NVIDIA A100 80 GB (PCIe), an AMD EPYC 7V13 CPU with 24 cores
and 216 GiB of host memory (Ubuntu 24.04, CUDA 12.9, driver 580).

| Comparison | Result |
|---|---|
| vs. [FVlog](https://github.com/harp-lab/vflog) (commit `fe1dcdc`), TC and SG | 27–35× faster on average (geometric mean per group), up to 206×; faster on every workload FVlog completes; completes 8 workloads on which FVlog runs out of memory or fails |
| vs. [GDlog](https://github.com/harp-lab/gdlog) (GPUlog, ASPLOS 2025), TC and SG | 17–36× faster on average, up to 444×; faster on every workload but one (TC on Tree<sub>17</sub>); completes 9 workloads on which GDlog runs out of memory or fails |
| vs. Soufflé 2.5 (24 threads), TC and SG | 305–518× faster on average (geometric mean per group) |
| vs. SortMerge (our sort-based engine) | 3.9–7.5× faster on average on TC and SG; up to 12.7× on SSSP |
| Largest input | transitive closure of 6.4 billion facts (G<sub>80K</sub>) in 3.7 s with 4 partitions and a peak of 38.8 GiB of GPU memory |
| CSPA (non-linear recursion) | 2.2–2.8× faster than FVlog, 2.7–4.8× faster than GDlog, 65–123× faster than Soufflé |
| CSDA (dataflow analysis, deep linear recursion) | 1.7–3.5× faster than SortMerge, 3.9–6.8× than GDlog, 46–48× than FVlog, 71–92× than Soufflé |

The measured times are in [`results/a100/`](results/a100); see [Results](#results) for the file of
each table and figure of the paper.

## Repository layout

```
setup.sh            one-shot setup: packages, Soufflé, FVlog, GDlog, our engines, data, tests
engines/            CUDA engines (make -> bin/)
  common.cuh          memory pool, CUB scans, CSR indexes, I/O, fingerprints, RESULT line
  fusedhash_tcsg.cu   FusedHash for TC and SG
  fusedhash_agg.cu    FusedHash for CC, SSSP (mmin) and company control (thresholded msum)
  fusedhash_cspa.cu   FusedHash for CSPA (non-linear recursion, appendix of the paper)
  cas_contention.cu   micro-benchmark: cost of many threads CASing the same key into one slot (Section 2)
  sortmerge_tcsg.cu   SortMerge: our implementation of the sort-based design of FVlog
  sortmerge_agg.cu    SortMerge extended to monotonic aggregation
souffle/            Datalog programs for Soufflé and their build script
fvlog/              build of the original FVlog, with a patch of its test drivers (result size) and
                    our CSDA driver (csda_test.cu, written as its TC driver)
gdlog/              build of the original GDlog (TC from its main branch, SG and CSPA from hash_diff),
                    with a patch of its drivers (result sizes, loading time) and our CSDA driver (csda.cu)
data/               download, generation and conversion of all the inputs
bench/              measurement harness and one script per experiment -> results/*.csv
tests/              correctness tests of every engine against CPU oracles
results/a100/       the CSV files measured for the paper
```

## Requirements

* An NVIDIA GPU with compute capability 8.0 or higher (A100, H100; the word-level join uses
  `__reduce_add_sync`), the NVIDIA driver and the CUDA toolkit (`nvcc`, tested with CUDA 12.9).
* Ubuntu 20.04, 22.04 or 24.04 with `sudo` (for `apt-get`), Python 3 with NumPy (`requirements.txt`),
  internet access.
* GDlog (branch `hash_diff`) downloads RMM and cuCollections when it is configured; its sample data are
  in Git LFS and are not needed (`gdlog/build_gdlog.sh` clones without them).
* FVlog downloads RAPIDS, RMM and cuCollections when it is configured; its README asks for the NVIDIA
  HPC SDK ≥ 24.1. If its build fails with the CUDA toolkit alone, install the HPC SDK and rerun
  `STEPS=fvlog ./setup.sh`; the other steps do not depend on it.
* About 30 GB of disk for the inputs in all formats (the largest aggregate inputs take several GB each),
  and 80 GB of GPU memory to reproduce the largest experiments.

## Quick start

```bash
./setup.sh                              # everything; the log goes to results/setup.log
bench/run_all.sh                        # all experiments, about a day -> results/*.csv
```

`setup.sh` runs the steps `check deps souffle fvlog gdlog engines data tests`, skipping each one whose output
already exists; a subset can be selected with `STEPS="engines tests" ./setup.sh`. The GPU architecture
is taken from `nvidia-smi` (or `CUDA_ARCH=90 ./setup.sh`).

Single experiments, and single inputs, can be run directly:

```bash
python3 bench/real.py --only tc --datasets fe_sphere usroads
python3 bench/aggregation.py --only sssp
python3 bench/fusedhash_final.py
EXPS="real synthetic" bench/run_all.sh
python3 bench/summarize.py                  # main numbers of the paper from results/a100
python3 bench/summarize.py --results results    # the same for a new run
```

Every script accepts `--resume`, which skips the rows already in its CSV file.

## Data

No data is stored in the repository: `setup.sh` downloads and generates every input.

| Input | Source | Script |
|---|---|---|
| Real graphs for TC and SG: CA-HepTh, com-dblp, Gnutella31, usroads, fe_sphere, fe_body, fe_ocean, vsp_finan (the datasets of FVlog and GDlog) | [SuiteSparse](https://sparse.tamu.edu) and [SNAP](https://snap.stanford.edu) | `data/download.py` |
| Road networks for CC and SSSP: road_usa, europe_osm | SuiteSparse (DIMACS10) | `data/download.py` |
| Synthetic BigDatalog families: Grid<sub>k</sub>, Tree<sub>h</sub>, G<sub>n</sub> | generated, fixed seeds | `data/download.py`, `data/prepare.py` |
| R-MAT, grids, small-world graphs, ownership graphs Own-n, uniform random graphs, random forests, chains | generated, fixed seeds | `data/prepare.py` |
| CSPA inputs httpd, linux, postgresql (from Graspan) | [FlowLog benchmark](https://pages.cs.wisc.edu/~m0riarty/dataset/csv/) | `data/download.py` |
| CSDA inputs httpd, linux, postgresql (from Graspan) | FlowLog benchmark on [Hugging Face](https://huggingface.co/datasets/NemoYuu/flowlog_benchmark) | `data/download.py` |
| UK register of persons with significant control (PSC), optional | [Companies House](http://download.companieshouse.gov.uk/en_pscdata.html) | `data/psc.py` |

`data/download.py` writes the graphs as NumPy arrays in `data/npy/`, and `data/prepare.py` writes the
inputs in the formats of the three systems: binary files of 32-bit rows for our engines (`data/bin`,
`data/agg`), tab-separated facts for Soufflé, FVlog and GDlog (`data/facts`, `data/cspa_facts`, `data/csda_facts`).
All generators are deterministic, so the generated inputs are identical to those of the paper.

The PSC input is optional, since the register is a 2 GB snapshot that Companies House replaces every
day. `data/psc.py` reads only the records of companies that control companies; the records of natural
persons are discarded without being decoded. The paper uses the snapshot of 2026-09-28; later
snapshots give slightly different counts. Without it, company control runs on Own-n only.

## Measurement rules

The rules are implemented once in `bench/harness.py` and applied to every system:

* each run is a separate process with a 900 s timeout, and GPU jobs run one at a time;
* **FusedHash and SortMerge**: a warm-up run on a small graph, then the best of two runs in the same
  process; the time covers index construction and the fixpoint, with the input already in host
  memory, and the result is counted and stays in GPU memory;
* **FVlog**: its own timer around the execution of its program, best of two processes;
* **GDlog**: the timers of its drivers (TC: the fixpoint; SG: base rule and fixpoint; CSPA: initial and
  analysis SCC), which leave out the construction of the indexes of the input; the harness also records
  that construction (`gdlog_load_s`) and the wall time of the process; best of two processes;
* **Soufflé**: compiled to C++ with OpenMP, one thread per core (and one thread where noted), best of
  two runs, wall time of the process;
* **plans**: for decomposable programs (TC), our engines use the single plan and retry with 4, 8 and
  16 partitions only if GPU memory is exhausted; FVlog has no partitioned plan;
* every result is compared with those of the other systems (same number of facts; column `agree`);
  failures are recorded as `OOM` (out of GPU or host memory), `TO` (timeout) or `ERR` (an error, such
  as the CUDA illegal memory accesses of FVlog and GDlog on some of the largest inputs).

**SG formulation.** The test of the FVlog code checks `x != y` only in the base rule, while the FVlog
paper writes it in both rules; all systems use the formulation of the code (`souffle/sg_bigdatalog.dl`,
`--neq 0`), so that the counts are comparable with those of FVlog.

## Engines

The engines read binary inputs and print one line `RESULT {json}` with the result size, the times,
the iterations and the peak GPU memory (high watermark of the memory pool). For example:

```
bin/fusedhash_tcsg --prog tc|sg --input g.bin [--base b.bin] [--plan 1] [--bitmap auto|off|force]
                   [--append atomic|warp] [--split delta|output] [--neq 0|1] [--reps 2]
                   [--loop 262144] [--words 1] [--fingerprint 0|1] [--verbose 0]
```

`--base` (TC only) takes the seed facts from another file, which evaluates CSDA:
NullNode(x,y) :- NullEdge(x,y). NullNode(x,y) :- NullNode(x,w), Edge(w,y).

The defaults of the engine differ from the configuration used in the paper, which
`bench/fusedhash_final.py` passes explicitly and which was chosen with `bench/ablation.py`,
`bench/configs.py` and `bench/loop_threshold.py`:

| Option | Default | Paper (A100) |
|---|---|---|
| `--split` (work distribution) | `delta` for TC, `output` for SG | `output` (one thread per candidate) for both |
| `--loop` (looping kernel up to this many candidates) | 262144 | 16384 |
| `--words` (join on words for dense deltas) | 1 | 1 |
| `--bitmap` (adaptive switch to a bitmap) | `auto` | `auto` |

## Correctness tests

```bash
python3 tests/run_tests.py
```

The tests run every engine, with every variant (hash table, bitmap, both work distributions, looping
kernel, join on words, 32-bit and 64-bit keys, partitioned plans), on small inputs with real recursion,
and compare the number of facts and an order-independent fingerprint of the set of facts with those of
CPU oracles (`tests/reference.py`), CSDA included (TC with `--base`). If FVlog, GDlog and Soufflé are
built, their counts are checked too.

## Results

The files of `results/a100/` are those measured for the paper; [`results/a100/README.md`](results/a100/README.md)
describes their columns, and `bench/summarize.py` computes the main numbers of the paper from them. The
FusedHash times in the tables and figures come from `fusedhash_final.csv` (the configuration above); those
of the other systems from the file of each experiment.

| Paper | Files | Script |
|---|---|---|
| Speedup over FVlog, GDlog and Soufflé (Figure 1) | `real.csv`, `synthetic.csv`, `large_tc.csv`, `gdlog.csv`, `fusedhash_final.csv` | `real.py`, `synthetic.py`, `large.py`, `gdlog.py`, `fusedhash_final.py` |
| Real-world datasets (TC, SG) | `real.csv`, `gdlog.csv`, `fusedhash_final.csv` | `real.py`, `gdlog.py` |
| Synthetic families (TC, SG) | `synthetic.csv`, `large_tc.csv`, `gdlog.csv`, `fusedhash_final.csv` | `synthetic.py`, `large.py`, `gdlog.py` |
| Monotonic aggregation (CC, SSSP, company control) | `aggregation.csv`, `large_agg.csv`, `memory.csv` | `aggregation.py`, `large.py`, `memory.py` |
| Rewrite amplification | `synthetic.csv`, `rewrite_amplification.csv` | `synthetic.py`, `amplification.py` |
| Chains (deep recursion) | `chain.csv`, `chain_gpu.csv`, `fusedhash_final.csv` | `secondary.py --only chain`, `chain_gpu.py` |
| Bitmap representation, work distribution, joins on words | `ablation.csv`, `configs.csv`, `fusedhash_final.csv` | `ablation.py`, `configs.py` |
| Uniform random graphs and random forests | `table5.csv` | `secondary.py --only table5` |
| Threshold of the looping kernel | `loop_threshold.csv` | `loop_threshold.py` |
| Partitioning on uniform random graphs | `partitions.csv` | `secondary.py --only partitions` |
| Larger inputs and peak GPU memory | `partitions_large.csv`, `memory.csv`, `large_tc.csv` | `partitions_large.py`, `memory.py`, `large.py` |
| CSPA (appendix) | `cspa.csv`, `cspa_first.csv` | `cspa.py` |
| CSDA (appendix) | `csda.csv` | `csda.py` |
| CAS contention on one slot (Section 2) | `cas_contention.csv`, `cas_contention_run2.csv` | `bin/cas_contention > results/cas_contention.csv` |
| SRDatalog, not in the comparison (Related Work) | `srdatalog.csv`, `srdatalog_commit.txt` | not included (see `results/a100/README.md`) |

`environment.txt` describes the machine and `fvlog_commit.txt` the FVlog commit. GDlog was measured on
the same machine in a later session (`gdlog.csv`, `chain_gpu.csv`, `cspa.csv`, `csda.csv`); `cspa.csv` and
`csda.csv` hold all the systems of that session.
`cspa_first.csv` holds the times of the first version of the CSPA engine, reported in the appendix
of the paper. `rewrite_amplification.csv` lists, for the synthetic workloads, the number of
iterations and the rewrite amplification A = Σ<sub>i</sub> |full<sub>i</sub>| / |result|, where
|full<sub>i</sub>| is the number of facts accumulated after iteration i: a property of the workload,
the same for every engine, which `bench/amplification.py` reads from the per-iteration log of SortMerge.

## Citation

If you use this code, please cite the paper above (see `CITATION.cff`; bibliographic details will be
added on publication).
