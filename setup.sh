#!/usr/bin/env bash
# One-shot setup on the GPU VM (Ubuntu 20.04/22.04/24.04 with an NVIDIA driver and the CUDA
# toolkit; internet access). Installs the build dependencies, builds Souffle 2.5 with OpenMP,
# clones and builds FVlog, builds our CUDA engines, prepares the data and runs the correctness
# tests, and GDlog. Every step is skipped if its output already exists; the log goes to results/setup.log.
#
#   ./setup.sh                 everything
#   STEPS="engines tests" ./setup.sh
#   CUDA_ARCH=80 (A100) or 90 (H100); default: the compute capability of GPU 0 (nvidia-smi), else 80
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"
mkdir -p results tools
exec > >(tee -a results/setup.log) 2>&1
STEPS="${STEPS:-check deps souffle fvlog gdlog engines data tests}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
CUDA_ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')}"
CUDA_ARCH="${CUDA_ARCH:-80}"
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo; echo "=== $(date '+%H:%M:%S') $*"; }

if has check; then
    say "environment"
    nvidia-smi || { echo "nvidia-smi not found: NVIDIA driver missing"; exit 1; }
    nvcc --version || { echo "nvcc not found: install the CUDA toolkit or set CUDA_HOME"; exit 1; }
    lsb_release -a 2>/dev/null || cat /etc/os-release
    echo "CPU threads: $(nproc)"; free -g
    python3 --version
    { nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nvcc --version | tail -2;
      echo "cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//') x $(nproc) threads";
      free -g | head -2; } > results/environment.txt
fi

if has deps; then
    say "system packages"
    sudo apt-get update -y
    sudo apt-get install -y build-essential git cmake bison flex libffi-dev libncurses5-dev \
        libsqlite3-dev zlib1g-dev mcpp python3 python3-pip python3-numpy libtbb-dev time lsb-release
    python3 -c "import numpy" || pip3 install --user numpy
    cmake_ver=$(cmake --version | head -1 | awk '{print $3}')
    if [ "$(printf '%s\n3.20\n' "$cmake_ver" | sort -V | head -1)" != "3.20" ]; then
        echo "cmake $cmake_ver < 3.20: installing a recent cmake with pip"
        pip3 install --user "cmake>=3.26"
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
export PATH="$HOME/.local/bin:$PATH"

if has souffle; then
    say "Souffle 2.5 (OpenMP, 64-bit numbers)"
    if [ ! -x tools/souffle/bin/souffle ]; then
        [ -d tools/souffle-src ] || git clone --depth 1 --branch 2.5 https://github.com/souffle-lang/souffle.git tools/souffle-src
        cmake -S tools/souffle-src -B tools/souffle-src/build -DCMAKE_BUILD_TYPE=Release \
            -DSOUFFLE_DOMAIN_64BIT=ON -DSOUFFLE_USE_OPENMP=ON -DSOUFFLE_ENABLE_TESTING=OFF \
            -DCMAKE_INSTALL_PREFIX="$root/tools/souffle"
        cmake --build tools/souffle-src/build -j"$(nproc)"
        cmake --install tools/souffle-src/build
    fi
    tools/souffle/bin/souffle --version
    souffle/build_programs.sh
fi

if has fvlog; then
    say "FVlog"
    CUDA_ARCH=$CUDA_ARCH fvlog/build_fvlog.sh || {
        echo "FVlog build failed: see the messages above (its README asks for the NVIDIA HPC SDK >= 24.1;"
        echo "on the VM: https://developer.nvidia.com/hpc-sdk-downloads). The other steps continue."; }
fi

if has gdlog; then
    say "GDlog"
    CUDA_ARCH=$CUDA_ARCH gdlog/build_gdlog.sh || {
        echo "GDlog build failed: see the messages above (branch hash_diff needs internet for RMM and"
        echo "cuCollections). The other steps continue."; }
fi

if has engines; then
    say "FusedHash and SortMerge (CUDA)"
    make -C engines ARCH=sm_$CUDA_ARCH
fi

if has data; then
    say "data"
    python3 data/download.py            # real graphs (SuiteSparse, SNAP), synthetic families, CSPA, CSDA
    python3 data/prepare.py             # inputs of the three systems; aggregate inputs generated
    python3 data/prepare.py --only large
fi

if has tests; then
    say "correctness tests"
    python3 tests/run_tests.py
fi
say "setup done; next: bench/run_all.sh"
