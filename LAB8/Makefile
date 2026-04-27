# ============================================================
# Makefile — CUDA GPU-ML Exercises (Parts 1-5)
# ============================================================
# Usage:
#   make all        — build parts 1-4 (part5 needs cuDNN)
#   make part1      — build only Part 1
#   make clean      — remove all binaries
#   make run_part1  — build and run Part 1
#
# Set SM to match your GPU's compute capability:
#   RTX 2060/2070/2080 : sm_75
#   RTX 3060/3070/3080 : sm_86
#   RTX 4080/4090      : sm_89
#   A100               : sm_80
#   H100               : sm_90
# ============================================================

SM        := sm_86
NVCC      := nvcc
NVCCFLAGS := -O2 -arch=$(SM) -lineinfo

# Libraries
LIBS_BASE  := -lm
LIBS_BLAS  := -lcublas $(LIBS_BASE)
LIBS_FULL  := -lcudnn -lcublas $(LIBS_BASE)

.PHONY: all clean part1 part2 part3 part4 part5 \
        run_part1 run_part2 run_part3 run_part4

all: part1 part2 part3 part4

part1: part1_gpu_fundamentals.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

part2: part2_shared_mem_ops.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

part3: part3_nn_building_blocks.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

part4: part4_matmul_and_convnets.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BLAS)
	@echo "[✓] Built $@"

part5: part5_digit_classifier.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_FULL)
	@echo "[✓] Built $@"

run_part1: part1
	./part1

run_part2: part2
	./part2

run_part3: part3
	./part3

run_part4: part4
	./part4

clean:
	rm -f part1 part2 part3 part4 part5
	@echo "[✓] Cleaned"
