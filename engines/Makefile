# CUDA engines: FusedHash and SortMerge for TC/SG and for monotonic aggregation.
#   make            (ARCH=sm_80 for A100; ARCH=sm_90 for H100)
NVCC  ?= nvcc
ARCH  ?= sm_80
FLAGS  = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -O3
BIN    = ../bin
PROGS  = $(BIN)/fusedhash_tcsg $(BIN)/sortmerge_tcsg $(BIN)/fusedhash_agg $(BIN)/sortmerge_agg \
         $(BIN)/fusedhash_cspa $(BIN)/cas_contention

all: $(PROGS)

$(BIN)/%: %.cu common.cuh
	@mkdir -p $(BIN)
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f $(PROGS)

.PHONY: all clean
