#!/usr/bin/env bash
# Runs all the experiments in sequence (one GPU job at a time) and logs to results/logs/.
#   bench/run_all.sh                 everything (about a day: Souffle timeouts are 900 s)
#   RESUME=1 EXPS="gdlog csda" bench/run_all.sh     after an interruption: --resume to the scripts that
#                                                  take it (the rows already in results/ are kept)
#   EXPS="real synthetic" bench/run_all.sh
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root/bench"
mkdir -p "$root/results/logs"
EXPS="${EXPS:-real synthetic aggregation large ablation configs loop_threshold fusedhash_final secondary partitions_large memory cspa gdlog csda chain_gpu amplification}"
# memory.py reads the engines of bin_mem/ (same code, built separately so that a rebuild of bin/ does not
# affect a running measurement)
if [[ " $EXPS " == *" memory "* ]]; then make -C "$root/engines" BIN=../bin_mem ARCH="${ARCH:-sm_80}"; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$root/results/logs/gpu.txt"
for e in $EXPS; do
    echo "=== $(date '+%F %T') $e" | tee -a "$root/results/logs/progress.log"
    args=()
    if [ -n "${RESUME:-}" ] && grep -q -- "--resume" "$e.py"; then args=(--resume); fi
    python3 "$e.py" "${args[@]}" 2>&1 | tee -a "$root/results/logs/$e.log"
    echo "=== $(date '+%F %T') $e done (${PIPESTATUS[0]})" | tee -a "$root/results/logs/progress.log"
done
