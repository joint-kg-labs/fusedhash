#!/usr/bin/env bash
# Clones and builds GDlog (github.com/harp-lab/gdlog; published as GPUlog, "Optimizing Datalog for the
# GPU", ASPLOS 2025) as its own run scripts do: TC from the branch main (run_tc_all.sh), SG and CSPA
# from the branch hash_diff (run_sg_all.sh, run_cspa_all.sh, which downloads RMM and cuCollections at
# configure time: internet needed). The commits are recorded in results/gdlog_commit.txt.
#   gdlog/build_gdlog.sh                 -> tools/gdlog/build/{TC,CSDA}, tools/gdlog-hash_diff/build/{SG,CSPA}
# (CSDA: our driver gdlog/csda.cu, added to the branch main by gdlog/patch_gdlog.py)
# Its README asks for a patch of Thrust (NVIDIA/thrust#1832, an integer overflow in thrust::reduce);
# the Thrust of CUDA 12.x (CCCL) is checked by the correctness tests instead (tests/run_tests.py).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
src="$root/tools/gdlog"
hd="$root/tools/gdlog-hash_diff"
arch="${CUDA_ARCH:-80}"
mkdir -p "$root/tools" "$root/results"
# its sample data are in Git LFS, whose downloads GitHub refuses for that repository: we do not need them
export GIT_LFS_SKIP_SMUDGE=1
[ -d "$src/.git" ] || git clone https://github.com/harp-lab/gdlog.git "$src"
cd "$src"
git fetch -q origin
git checkout -q main
[ -d "$hd" ] || git worktree add -q "$hd" origin/hash_diff
{ echo "main $(git rev-parse HEAD)"; echo "hash_diff $(git -C "$hd" rev-parse HEAD)"; } > "$root/results/gdlog_commit.txt"
python3 "$here/patch_gdlog.py" "$src" "$hd"
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
# main: TC (the architecture from CMAKE_CUDA_ARCHITECTURES)
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$arch" -S "$src" -B "$src/build" 2>&1 | tail -5
cmake --build "$src/build" -j"$(nproc)" --target TC CSDA 2>&1 | tail -5
# hash_diff: SG and CSPA (its CMakeLists fixes the architectures to 80;89;90)
cmake -DCMAKE_BUILD_TYPE=Release -S "$hd" -B "$hd/build" 2>&1 | tail -5
cmake --build "$hd/build" -j"$(nproc)" --target SG CSPA 2>&1 | tail -5
ls -l "$src/build/TC" "$src/build/CSDA" "$hd/build/SG" "$hd/build/CSPA"
cat "$root/results/gdlog_commit.txt"
