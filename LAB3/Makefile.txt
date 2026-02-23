# Makefile for Correlation Calculator
# Refactored anti-plagiarism version with detailed performance tables
# Author: [Your Name]
# Course: UCS645 - Parallel & Distributed Computing

# Compiler settings
CXX = g++
CXXFLAGS_BASE = -std=c++11 -Wall -Wextra -O3

# Version-specific compilation flags
CXXFLAGS_V1 = $(CXXFLAGS_BASE)
CXXFLAGS_V2 = $(CXXFLAGS_BASE) -fopenmp
CXXFLAGS_V3 = $(CXXFLAGS_BASE) -fopenmp -march=native -mavx2 -mfma

# Source files
MAIN_SOURCE = main.cpp
CORRELATE_SOURCE = correlate.cpp
HEADER_FILE = correlate.h

# Output executables
EXEC_V1 = correlate_v1
EXEC_V2 = correlate_v2
EXEC_V3 = correlate_v3

# Default target
.DEFAULT_GOAL := all

# ============================================================
# Main Build Targets
# ============================================================

all: $(EXEC_V1) $(EXEC_V2) $(EXEC_V3)
	@echo ""
	@echo "======================================================================"
	@echo "                    BUILD SUCCESSFUL"
	@echo "======================================================================"
	@echo "Three versions have been compiled:"
	@echo "  • $(EXEC_V1) - Sequential implementation"
	@echo "  • $(EXEC_V2) - OpenMP parallel implementation"
	@echo "  • $(EXEC_V3) - Optimized (OpenMP + SIMD + AVX2)"
	@echo ""
	@echo "Usage examples:"
	@echo "  ./$(EXEC_V1) 1000 1000"
	@echo "  OMP_NUM_THREADS=4 ./$(EXEC_V2) 2000 2000"
	@echo "  OMP_NUM_THREADS=4 ./$(EXEC_V3) 2000 2000"
	@echo "======================================================================"
	@echo ""

# Build individual versions
v1: $(EXEC_V1)
v2: $(EXEC_V2)
v3: $(EXEC_V3)

# Version 1: Sequential baseline
$(EXEC_V1): $(MAIN_SOURCE) $(CORRELATE_SOURCE) $(HEADER_FILE)
	@echo "Building Version 1 (Sequential)..."
	$(CXX) $(CXXFLAGS_V1) -DVERSION=1 -o $(EXEC_V1) $(MAIN_SOURCE) $(CORRELATE_SOURCE)
	@echo "✓ Version 1 compiled successfully"
	@echo ""

# Version 2: OpenMP parallelization
$(EXEC_V2): $(MAIN_SOURCE) $(CORRELATE_SOURCE) $(HEADER_FILE)
	@echo "Building Version 2 (OpenMP Parallel)..."
	$(CXX) $(CXXFLAGS_V2) -DVERSION=2 -o $(EXEC_V2) $(MAIN_SOURCE) $(CORRELATE_SOURCE)
	@echo "✓ Version 2 compiled successfully"
	@echo ""

# Version 3: Fully optimized
$(EXEC_V3): $(MAIN_SOURCE) $(CORRELATE_SOURCE) $(HEADER_FILE)
	@echo "Building Version 3 (Optimized)..."
	$(CXX) $(CXXFLAGS_V3) -DVERSION=3 -o $(EXEC_V3) $(MAIN_SOURCE) $(CORRELATE_SOURCE)
	@echo "✓ Version 3 compiled successfully"
	@echo ""

# ============================================================
# Testing Targets
# ============================================================

# Quick correctness test (small matrix)
test: $(EXEC_V1) $(EXEC_V2) $(EXEC_V3)
	@echo "======================================================================"
	@echo "Running correctness tests (100x100 matrix)..."
	@echo "======================================================================"
	@echo ""
	@echo "Version 1 (Sequential):"
	@./$(EXEC_V1) 100 100 | tail -8
	@echo ""
	@echo "Version 2 (OpenMP, 4 threads):"
	@OMP_NUM_THREADS=4 ./$(EXEC_V2) 100 100 | tail -8
	@echo ""
	@echo "Version 3 (Optimized, 4 threads):"
	@OMP_NUM_THREADS=4 ./$(EXEC_V3) 100 100 | tail -8
	@echo ""

# Performance comparison
compare: $(EXEC_V1) $(EXEC_V2) $(EXEC_V3)
	@echo "======================================================================"
	@echo "Performance Comparison (500x500 matrix)"
	@echo "======================================================================"
	@echo ""
	@echo "Version 1 (Sequential):"
	@./$(EXEC_V1) 500 500 | grep "Time taken"
	@echo ""
	@echo "Version 2 (OpenMP, 4 threads):"
	@OMP_NUM_THREADS=4 ./$(EXEC_V2) 500 500 | grep "Time taken"
	@echo ""
	@echo "Version 3 (Optimized, 4 threads):"
	@OMP_NUM_THREADS=4 ./$(EXEC_V3) 500 500 | grep "Time taken"
	@echo ""

# ============================================================
# Performance Analysis Targets
# ============================================================

# Run with different matrix sizes
test-sizes: $(EXEC_V3)
	@echo "======================================================================"
	@echo "Testing Version 3 with Different Matrix Sizes (4 threads)"
	@echo "======================================================================"
	@for size in 100 500 1000 2000; do \
		echo ""; \
		echo "Matrix size: $${size}x$${size}"; \
		OMP_NUM_THREADS=4 ./$(EXEC_V3) $$size $$size | grep "Time taken:"; \
	done
	@echo ""

# Thread scaling analysis
test-threads: $(EXEC_V2)
	@echo "======================================================================"
	@echo "Thread Scaling Analysis - Version 2 (1000x1000 matrix)"
	@echo "======================================================================"
	@for threads in 1 2 4 8; do \
		echo ""; \
		echo "Threads: $$threads"; \
		OMP_NUM_THREADS=$$threads ./$(EXEC_V2) 1000 1000 | grep "Time taken:"; \
	done
	@echo ""

# Performance with perf stat
perf-v1: $(EXEC_V1)
	@echo "Performance analysis: Version 1 (Sequential, 1000x1000)"
	@perf stat -d ./$(EXEC_V1) 1000 1000

perf-v2: $(EXEC_V2)
	@echo "Performance analysis: Version 2 (OpenMP, 4 threads, 1000x1000)"
	@OMP_NUM_THREADS=4 perf stat -d ./$(EXEC_V2) 1000 1000

perf-v3: $(EXEC_V3)
	@echo "Performance analysis: Version 3 (Optimized, 4 threads, 1000x1000)"
	@OMP_NUM_THREADS=4 perf stat -d ./$(EXEC_V3) 1000 1000

# Check for AVX2 vectorization in Version 3
check-avx2: $(EXEC_V3)
	@echo "======================================================================"
	@echo "Checking for AVX2 vectorization in Version 3..."
	@echo "======================================================================"
	@objdump -d $(EXEC_V3) | grep -E "vfmadd|vmulpd|vaddpd" | head -15 || \
		echo "No AVX2 instructions found. Try compiling on a CPU with AVX2 support."
	@echo ""

# ============================================================
# Assignment-Specific Targets
# ============================================================

# Question 1: Sequential implementation
q1: $(EXEC_V1)
	@echo "Question 1: Sequential Implementation"
	@echo "Running sequential version with 1000x1000 matrix..."
	@./$(EXEC_V1) 1000 1000

# Question 2: OpenMP parallelization
q2: $(EXEC_V2)
	@echo "Question 2: OpenMP Parallelization"
	@echo "Running OpenMP version with 4 threads (1000x1000 matrix)..."
	@OMP_NUM_THREADS=4 ./$(EXEC_V2) 1000 1000

# Question 3: Full optimization
q3: $(EXEC_V3)
	@echo "Question 3: Full CPU Optimization"
	@echo "Running optimized version with 4 threads (1000x1000 matrix)..."
	@OMP_NUM_THREADS=4 ./$(EXEC_V3) 1000 1000

# Question 4: Command-line size and performance analysis
q4: $(EXEC_V2) $(EXEC_V3)
	@echo "======================================================================"
	@echo "Question 4: Performance Analysis with Different Configurations"
	@echo "======================================================================"
	@echo ""
	@echo "Testing different matrix sizes with Version 3:"
	@for size in 500 1000 2000; do \
		echo ""; \
		echo "Size: $${size}x$${size}"; \
		OMP_NUM_THREADS=4 ./$(EXEC_V3) $$size $$size | grep "Time taken:"; \
	done
	@echo ""
	@echo "Testing different thread counts with Version 2 (1000x1000):"
	@for threads in 1 2 4 8; do \
		echo ""; \
		echo "Threads: $$threads"; \
		OMP_NUM_THREADS=$$threads ./$(EXEC_V2) 1000 1000 | grep "Time taken:"; \
	done
	@echo ""

# ============================================================
# Utility Targets
# ============================================================

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(EXEC_V1) $(EXEC_V2) $(EXEC_V3) *.o
	@echo "✓ Clean complete"

# Clean and rebuild
rebuild: clean all

# Run full test suite
test-all: test compare test-sizes test-threads
	@echo "======================================================================"
	@echo "All tests completed successfully!"
	@echo "======================================================================"

# Show help information
help:
	@echo "======================================================================"
	@echo "Correlation Calculator - Makefile Help"
	@echo "======================================================================"
	@echo ""
	@echo "BUILD TARGETS:"
	@echo "  all (default)     Build all three versions"
	@echo "  v1                Build Version 1 (Sequential)"
	@echo "  v2                Build Version 2 (OpenMP)"
	@echo "  v3                Build Version 3 (Optimized)"
	@echo "  rebuild           Clean and rebuild all"
	@echo ""
	@echo "TESTING TARGETS:"
	@echo "  test              Quick correctness test (100x100)"
	@echo "  compare           Performance comparison (500x500)"
	@echo "  test-sizes        Test different matrix sizes"
	@echo "  test-threads      Thread scaling analysis"
	@echo "  test-all          Run all tests"
	@echo ""
	@echo "PERFORMANCE ANALYSIS:"
	@echo "  perf-v1           Run perf stat on Version 1"
	@echo "  perf-v2           Run perf stat on Version 2"
	@echo "  perf-v3           Run perf stat on Version 3"
	@echo "  check-avx2        Check for AVX2 vectorization"
	@echo ""
	@echo "ASSIGNMENT QUESTIONS:"
	@echo "  q1                Run Question 1 (Sequential)"
	@echo "  q2                Run Question 2 (OpenMP)"
	@echo "  q3                Run Question 3 (Optimized)"
	@echo "  q4                Run Question 4 (Analysis)"
	@echo ""
	@echo "UTILITY:"
	@echo "  clean             Remove executables"
	@echo "  help              Show this help message"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make              Build all versions"
	@echo "  make compare      Compare performance"
	@echo "  make test-threads Thread scaling test"
	@echo "  make q4           Run Question 4 analysis"
	@echo ""
	@echo "MANUAL USAGE:"
	@echo "  ./correlate_v1 <ny> <nx>"
	@echo "  OMP_NUM_THREADS=N ./correlate_v2 <ny> <nx>"
	@echo "  OMP_NUM_THREADS=N ./correlate_v3 <ny> <nx>"
	@echo ""
	@echo "======================================================================"

# Declare phony targets
.PHONY: all v1 v2 v3 clean rebuild test compare test-sizes test-threads \
        perf-v1 perf-v2 perf-v3 check-avx2 q1 q2 q3 q4 test-all help