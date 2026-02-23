/*
 * correlate.cpp
 * Correlation Coefficient Calculator
 * Three implementation variants with progressive optimization
 */

#include "correlate.h"
#include <cmath>
#include <vector>

#ifndef VERSION
#define VERSION 1
#endif

#if VERSION >= 2
#include <omp.h>
#endif

#if VERSION == 3
    #if defined(__AVX2__) && defined(__FMA__)
        #include <immintrin.h>
        #define HAS_AVX2 1
    #else
        #define HAS_AVX2 0
    #endif
#endif

//==============================================================================
// IMPLEMENTATION 1: Basic Sequential Approach
//==============================================================================
#if VERSION == 1

void correlate(int ny, int nx, const float* data, float* result) {
    // Allocate workspace for standardized values
    std::vector<double> standardized_data(ny * nx);
    
    // Phase 1: Standardize each vector (z-score transformation)
    for (int vec_idx = 0; vec_idx < ny; ++vec_idx) {
        const int offset = vec_idx * nx;
        
        // Calculate average value
        double accumulator = 0.0;
        for (int elem = 0; elem < nx; ++elem) {
            accumulator += static_cast<double>(data[offset + elem]);
        }
        double avg_value = accumulator / static_cast<double>(nx);
        
        // Calculate variance
        double variance_sum = 0.0;
        for (int elem = 0; elem < nx; ++elem) {
            double deviation = static_cast<double>(data[offset + elem]) - avg_value;
            standardized_data[offset + elem] = deviation;
            variance_sum += deviation * deviation;
        }
        
        // Apply standardization (handle zero variance case)
        double scale_factor = (variance_sum > 1e-10) ? 
            std::sqrt(static_cast<double>(nx) / variance_sum) : 0.0;
        
        for (int elem = 0; elem < nx; ++elem) {
            standardized_data[offset + elem] *= scale_factor;
        }
    }
    
    // Phase 2: Calculate pairwise correlations
    for (int row1 = 0; row1 < ny; ++row1) {
        for (int row2 = 0; row2 <= row1; ++row2) {
            double product_sum = 0.0;
            
            for (int pos = 0; pos < nx; ++pos) {
                product_sum += standardized_data[row1 * nx + pos] * 
                              standardized_data[row2 * nx + pos];
            }
            
            // Normalize and store
            double corr_value = product_sum / static_cast<double>(nx);
            
            // Clamp to valid correlation range
            corr_value = (corr_value > 1.0) ? 1.0 : corr_value;
            corr_value = (corr_value < -1.0) ? -1.0 : corr_value;
            
            result[row1 + row2 * ny] = static_cast<float>(corr_value);
        }
    }
}

//==============================================================================
// IMPLEMENTATION 2: Parallel Threading with OpenMP
//==============================================================================
#elif VERSION == 2

void correlate(int ny, int nx, const float* data, float* result) {
    // Storage for centered and scaled vectors
    std::vector<double> normalized_vectors(ny * nx);
    
    // Parallel normalization of input vectors
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ny; ++i) {
        const int row_start = i * nx;
        
        // Compute mean for this row
        double mean_val = 0.0;
        for (int j = 0; j < nx; ++j) {
            mean_val += static_cast<double>(data[row_start + j]);
        }
        mean_val /= nx;
        
        // Center the data and accumulate squared deviations
        double sum_squared_dev = 0.0;
        for (int j = 0; j < nx; ++j) {
            double centered = static_cast<double>(data[row_start + j]) - mean_val;
            normalized_vectors[row_start + j] = centered;
            sum_squared_dev += centered * centered;
        }
        
        // Scale by standard deviation
        double std_recip = (sum_squared_dev > 1e-10) ? 
            (std::sqrt(nx) / std::sqrt(sum_squared_dev)) : 0.0;
        
        for (int j = 0; j < nx; ++j) {
            normalized_vectors[row_start + j] *= std_recip;
        }
    }
    
    // Compute correlation matrix in parallel
    // Use dynamic scheduling for load balancing (lower triangle has varying work)
    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < ny; ++i) {
        const double* vec_i = &normalized_vectors[i * nx];
        
        for (int j = 0; j <= i; ++j) {
            const double* vec_j = &normalized_vectors[j * nx];
            
            // Inner product of normalized vectors
            double inner_prod = 0.0;
            for (int k = 0; k < nx; ++k) {
                inner_prod += vec_i[k] * vec_j[k];
            }
            
            // Normalize by vector length
            inner_prod /= nx;
            
            // Ensure valid range
            if (inner_prod > 1.0) inner_prod = 1.0;
            if (inner_prod < -1.0) inner_prod = -1.0;
            
            result[i + j * ny] = static_cast<float>(inner_prod);
        }
    }
}

//==============================================================================
// IMPLEMENTATION 3: Highly Optimized (Threading + SIMD Vectorization)
//==============================================================================
#elif VERSION == 3

void correlate(int ny, int nx, const float* data, float* result) {
    // Pre-compute reciprocal for efficiency
    const double inv_dimension = 1.0 / static_cast<double>(nx);
    
    // Contiguous storage for processed data
    std::vector<double> processed(ny * nx);
    
    // Parallel preprocessing with SIMD hints
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < ny; ++idx) {
        const float* input_row = &data[idx * nx];
        double* output_row = &processed[idx * nx];
        
        // Phase 1: Calculate mean using SIMD reduction
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < nx; ++j) {
            sum += static_cast<double>(input_row[j]);
        }
        double mean = sum * inv_dimension;
        
        // Phase 2: Center data and compute variance
        double var_accum = 0.0;
        #pragma omp simd reduction(+:var_accum)
        for (int j = 0; j < nx; ++j) {
            double diff = static_cast<double>(input_row[j]) - mean;
            output_row[j] = diff;
            var_accum += diff * diff;
        }
        
        // Phase 3: Normalize (avoid division by zero)
        double normalization = (var_accum > 1e-10) ? 
            std::sqrt(static_cast<double>(nx) / var_accum) : 0.0;
        
        #pragma omp simd
        for (int j = 0; j < nx; ++j) {
            output_row[j] *= normalization;
        }
    }
    
    // Correlation computation with advanced optimizations
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < ny; ++i) {
        const double* __restrict__ row_a = &processed[i * nx];
        
        for (int j = 0; j <= i; ++j) {
            const double* __restrict__ row_b = &processed[j * nx];
            
            double dot_product = 0.0;
            
            #if HAS_AVX2
            // Manual AVX2 vectorization for maximum performance
            __m256d accumulator = _mm256_setzero_pd();
            int k = 0;
            
            // Process 4 doubles per iteration
            for (; k + 3 < nx; k += 4) {
                __m256d va = _mm256_loadu_pd(&row_a[k]);
                __m256d vb = _mm256_loadu_pd(&row_b[k]);
                accumulator = _mm256_fmadd_pd(va, vb, accumulator);
            }
            
            // Horizontal reduction
            __m128d upper = _mm256_extractf128_pd(accumulator, 1);
            __m128d lower = _mm256_castpd256_pd128(accumulator);
            __m128d combined = _mm_add_pd(lower, upper);
            __m128d reduced = _mm_hadd_pd(combined, combined);
            dot_product = _mm_cvtsd_f64(reduced);
            
            // Handle remaining elements
            for (; k < nx; ++k) {
                dot_product += row_a[k] * row_b[k];
            }
            #else
            // Compiler auto-vectorization fallback
            #pragma omp simd reduction(+:dot_product)
            for (int k = 0; k < nx; ++k) {
                dot_product += row_a[k] * row_b[k];
            }
            #endif
            
            // Final normalization
            double correlation = dot_product * inv_dimension;
            
            // Range clamping
            correlation = (correlation > 1.0) ? 1.0 : correlation;
            correlation = (correlation < -1.0) ? -1.0 : correlation;
            
            result[i + j * ny] = static_cast<float>(correlation);
        }
    }
}

#else
#error "VERSION must be 1, 2, or 3"
#endif