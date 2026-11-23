#!/usr/bin/env bash
# Clones and builds FVlog (github.com/harp-lab/vflog) into tools/vflog; binaries in
# tools/vflog/build/test (TC_TEST, SG_TEST, CSPA_TEST, CSDA_TEST from fvlog/csda_test.cu, ...).
# FVlog downloads rapids-cmake, RMM and cuCollections at configure time (internet needed).
#   fvlog/build_fvlog.sh [commit]      (default: fe1dcdc, the commit used in the paper; recorded)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
dst="$root/tools/vflog"
mkdir -p "$root/tools" "$root/results"
if [ ! -d "$dst/.git" ]; then
    git clone https://github.com/harp-lab/vflog.git "$dst"
fi
cd "$dst"
git checkout "${1:-fe1dcdc84b83ff2dd7a32015614c43a423c18504}"
git rev-parse HEAD > "$root/results/fvlog_commit.txt"
python3 "$here/patch_fvlog.py" "$dst/test/src"
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-80} -Bbuild . 2>&1 | tail -20
cmake --build build -j"$(nproc)" 2>&1 | tail -20
ls -l build/test/TC_TEST build/test/SG_TEST build/test/CSPA_TEST build/test/CSDA_TEST
echo "FVlog commit $(cat "$root/results/fvlog_commit.txt")"
