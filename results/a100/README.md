# Results on the NVIDIA A100

The CSV files measured for the paper, on an NVIDIA A100 80 GB (PCIe) with an AMD EPYC 7V13 (24 cores)
and 216 GiB of host memory (`environment.txt`); FVlog at the commit in `fvlog_commit.txt`. GDlog
(`gdlog.csv`, `chain_gpu.csv`, `cspa.csv`, `csda.csv`; the commits of its branches `main` and `hash_diff` in
`gdlog_commit.txt`) was measured on the same machine in a later session, with the other systems of `cspa.csv`
and `csda.csv` measured again in that session.
`python3 bench/summarize.py` computes the main numbers of the paper from these files.

## Conventions

* Times are in seconds (`*_s`), the best of two runs; memory is in GiB (`*_gib`).
* A time cell that is not a number records a failure: `OOM` (out of GPU or host memory), `timeout`
  (more than 900 s), `error` (the process terminated with an error, with its message in `notes`; for
  Soufflé this is the process killed when it exhausts the 216 GiB of host memory, reported as OOM in
  the paper), `missing` (the system does not implement the program).
* `result` is the number of derived facts (groups for CC and SSSP); `agree` is 1 when all the systems
  that complete a workload derive the same number of facts.
* `plan` / `P` is the number of partitions (1 = single plan); `set` is the structure that holds the
  accumulated relation at the end of the run (`hash32`, `hash64` or `bitmap`).
* The FusedHash times in the tables and figures of the paper come from `fusedhash_final.csv`; the
  `fusedhash_s` columns of the other files were measured with the default options of the engine.

## Files

| File | Script | Rows | Columns |
|---|---|---|---|
| `real.csv`, `synthetic.csv`, `large_tc.csv` | `real.py`, `synthetic.py`, `large.py` | TC and SG on the real-world graphs, the synthetic families, G<sub>60K</sub> and G<sub>80K</sub> | `nodes`, `edges` of the input; time, plan and set of FusedHash; time and plan of SortMerge; time and result of FVlog; time and peak RSS of Soufflé (24 threads) |
| `fusedhash_final.csv` | `fusedhash_final.py` | every TC and SG input, with the configuration of the paper; then the largest TC inputs with P = 1..16 | `suite` (real, synthetic, large, chain, table5, partitions), `P`, time, set, facts of the largest partition, peak GPU memory (`used` = allocated by the memory pool, `reserved` = taken from the device), iterations, of which in the looping kernel and with the join on words |
| `aggregation.csv`, `large_agg.csv` | `aggregation.py`, `large.py` | CC, SSSP and company control | `rows` of the input, `checksum` (sum of the values, CC and SSSP), iterations of each engine, `map_keys` (groups of the running sums, company control), times of FusedHash and SortMerge |
| `memory.csv` | `memory.py` | every input | plan and peak GPU memory of FusedHash and SortMerge |
| `ablation.csv` | `ablation.py` | TC and SG inputs, one component at a time | `variant`: `base` (hash table only), `bitmap` (adaptive bitmap), `warp` (warp-aggregated append), `split` (the other work distribution), `loop` (looping kernel), `words` (join on words), `all`; time, set and its size `set_gib` |
| `configs.csv` | `configs.py` | TC and SG inputs | `config`: `delta` / `output` (one thread per fact of Δ / per candidate), `output+loop` (with the looping kernel) |
| `loop_threshold.csv` | `loop_threshold.py` | chains, grids and trees | `loop`: largest number of candidates of an iteration run by the looping kernel (0 = off) |
| `chain.csv` | `secondary.py --only chain` | chains of 1,000, 3,000 and 10,000 vertices | FusedHash without and with the looping kernel (default threshold), time per iteration in ms, FVlog, Soufflé with 1 thread |
| `table5.csv` | `secondary.py --only table5` | uniform random graphs (TC), random forests (SG) | Soufflé with 1 and 24 threads (`soufflen_s`), FVlog, FusedHash |
| `partitions.csv`, `partitions_large.csv` | `secondary.py --only partitions`, `partitions_large.py` | uniform random graphs; the largest TC inputs | `P`, facts, facts of the largest partition, peak GPU memory, time |
| `cspa.csv`, `cspa_first.csv` | `cspa.py` | CSPA on httpd, linux, postgresql | sizes of VF, VA, MA; times of FusedHash (final and first version of the engine), FVlog, GDlog (its own timers; `gdlog_wall_s`: its process), Soufflé |
| `gdlog.csv` | `gdlog.py` | every TC and SG input of `real.csv`, `synthetic.csv`, `large_tc.csv`, GDlog alone | its result and the reference result, `gdlog_s` (its own timers: the fixpoint, without the construction of the indexes of the input), `gdlog_load_s` (that construction), their sum, the wall time of its process; the times of FusedHash (final), FVlog and Soufflé copied from the other files |
| `chain_gpu.csv` | `chain_gpu.py` | the chains of `chain.csv`, GDlog alone | as `gdlog.csv`, with the times of `chain.csv` beside |
| `cas_contention.csv`, `cas_contention_run2.csv` | `engines/cas_contention.cu` (two runs) | 2<sup>24</sup> threads in groups of X, each group on its own cell: the same key (`dup`), different keys with the same start slot (`collide`), the same bitmap word (`bitmap`); threads of a group consecutive (`adjacent`) or strided (`spread`) | kernel time in ms (median of 7 launches), ns per test, CAS attempts per test, fresh insertions and their expected number. Section 2 of the paper: `dup` with X = 1 and X = 2<sup>24</sup> |
| `srdatalog.csv`, `srdatalog_commit.txt` | SRDatalog (harp-lab/srdatalog-python, commit in `srdatalog_commit.txt`), compiled for sm_80; its scripts are not part of this repository | every TC and SG input | its result and the reference result, its own timers (`srdatalog_s`, load, total; resolution 0.1 s), the wall time of its process, the times of the other systems; `agree` = 0 on TC G<sub>20K</sub>–G<sub>60K</sub>, where its result differs from that of all the other systems, which is why the paper leaves it out of the comparison |
| `csda.csv` | `csda.py` | CSDA on httpd, linux, postgresql | input sizes (`edges`, `null_edges`), result, iterations; times of FusedHash (final configuration, TC with `--base`), SortMerge, FVlog and GDlog (our drivers `fvlog/csda_test.cu`, `gdlog/csda.cu`), Soufflé |

`rewrite_amplification.csv` (`bench/amplification.py`) lists the iterations and the rewrite
amplification A = Σ<sub>i</sub> |full<sub>i</sub>| / |result| of the synthetic workloads.
