NVCC = nvcc

#  Default architecture (safe for Colab: Tesla T4 = sm_75)
# You can override while running: make ARCH=sm_89
ARCH ?= -arch=sm_75

CFLAGS = -O3

TARGETS = device_info array_sum matrix_add

all: $(TARGETS)

# 🔹 Device Info
device_info: device_info.cu
	$(NVCC) $(ARCH) $(CFLAGS) $< -o $@

# 🔹 Array Sum
array_sum: array_sum.cu
	$(NVCC) $(ARCH) $(CFLAGS) $< -o $@

# 🔹 Matrix Add
matrix_add: matrix_add.cu
	$(NVCC) $(ARCH) $(CFLAGS) $< -o $@

# 🔹 Run all programs
run: all
	./device_info
	./array_sum
	./matrix_add

# 🔹 Clean build files
clean:
	rm -f $(TARGETS)