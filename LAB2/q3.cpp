// Exercise 3: Scientific Computing - 2D Heat Diffusion Simulation
// Finite Difference Method with OpenMP parallelization

#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <algorithm>

using namespace std;

// Physical parameters
const double ALPHA = 0.01;      // Thermal diffusivity
const double DX = 0.1;           // Spatial step
const double DY = 0.1;
const double DT = 0.001;         // Time step
const double STABILITY = ALPHA * DT / (DX * DX);  // Should be < 0.25

// Heat diffusion with different scheduling strategies
void heat_diffusion(int N, int time_steps, const string& schedule_type,
                   int num_threads, double& time_taken, double& final_temp) {
    
    // Create 2D temperature grids
    vector<vector<double>> temp(N, vector<double>(N, 0.0));
    vector<vector<double>> temp_new(N, vector<double>(N, 0.0));
    
    // Initialize: hot center, cool boundaries
    int center = N / 2;
    int radius = N / 10;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double dist = sqrt((i - center) * (i - center) + (j - center) * (j - center));
            if (dist < radius) {
                temp[i][j] = 100.0;  // Hot center
            }
        }
    }
    
    double start_time = omp_get_wtime();
    
    // Time evolution
    for (int t = 0; t < time_steps; t++) {
        double total_heat = 0.0;
        
        // Parallel spatial computation
        if (schedule_type == "static") {
            #pragma omp parallel for collapse(2) schedule(static) num_threads(num_threads) reduction(+:total_heat)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    temp_new[i][j] = temp[i][j] + ALPHA * DT * (
                        (temp[i+1][j] - 2*temp[i][j] + temp[i-1][j]) / (DX * DX) +
                        (temp[i][j+1] - 2*temp[i][j] + temp[i][j-1]) / (DY * DY)
                    );
                    total_heat += temp_new[i][j];
                }
            }
        } else if (schedule_type == "dynamic") {
            #pragma omp parallel for collapse(2) schedule(dynamic, 16) num_threads(num_threads) reduction(+:total_heat)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    temp_new[i][j] = temp[i][j] + ALPHA * DT * (
                        (temp[i+1][j] - 2*temp[i][j] + temp[i-1][j]) / (DX * DX) +
                        (temp[i][j+1] - 2*temp[i][j] + temp[i][j-1]) / (DY * DY)
                    );
                    total_heat += temp_new[i][j];
                }
            }
        } else if (schedule_type == "guided") {
            #pragma omp parallel for collapse(2) schedule(guided) num_threads(num_threads) reduction(+:total_heat)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    temp_new[i][j] = temp[i][j] + ALPHA * DT * (
                        (temp[i+1][j] - 2*temp[i][j] + temp[i-1][j]) / (DX * DX) +
                        (temp[i][j+1] - 2*temp[i][j] + temp[i][j-1]) / (DY * DY)
                    );
                    total_heat += temp_new[i][j];
                }
            }
        }
        
        // Swap grids (no race condition - reading from temp, writing to temp_new)
        swap(temp, temp_new);
    }
    
    double end_time = omp_get_wtime();
    time_taken = end_time - start_time;
    
    // Calculate final average temperature
    final_temp = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            final_temp += temp[i][j];
        }
    }
    final_temp /= (N * N);
}

// Cache-blocked version for better memory performance
void heat_diffusion_blocked(int N, int time_steps, int num_threads,
                           double& time_taken, double& final_temp) {
    vector<vector<double>> temp(N, vector<double>(N, 0.0));
    vector<vector<double>> temp_new(N, vector<double>(N, 0.0));
    
    // Initialize
    int center = N / 2;
    int radius = N / 10;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double dist = sqrt((i - center) * (i - center) + (j - center) * (j - center));
            if (dist < radius) {
                temp[i][j] = 100.0;
            }
        }
    }
    
    double start_time = omp_get_wtime();
    
    const int BLOCK_SIZE = 32;  // Cache-friendly block size
    
    for (int t = 0; t < time_steps; t++) {
        #pragma omp parallel for collapse(2) schedule(static) num_threads(num_threads)
        for (int bi = 1; bi < N - 1; bi += BLOCK_SIZE) {
            for (int bj = 1; bj < N - 1; bj += BLOCK_SIZE) {
                // Process block
                int i_end = min(bi + BLOCK_SIZE, N - 1);
                int j_end = min(bj + BLOCK_SIZE, N - 1);
                
                for (int i = bi; i < i_end; i++) {
                    for (int j = bj; j < j_end; j++) {
                        temp_new[i][j] = temp[i][j] + ALPHA * DT * (
                            (temp[i+1][j] - 2*temp[i][j] + temp[i-1][j]) / (DX * DX) +
                            (temp[i][j+1] - 2*temp[i][j] + temp[i][j-1]) / (DY * DY)
                        );
                    }
                }
            }
        }
        swap(temp, temp_new);
    }
    
    double end_time = omp_get_wtime();
    time_taken = end_time - start_time;
    
    final_temp = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            final_temp += temp[i][j];
        }
    }
    final_temp /= (N * N);
}

int main() {
    int N = 512;           // Grid size (512x512)
    int time_steps = 100;  // Number of time steps
    
    cout << "2D Heat Diffusion Simulation" << endl;
    cout << "Grid size: " << N << " x " << N << endl;
    cout << "Time steps: " << time_steps << endl;
    cout << "Stability criterion (α*Δt/Δx²): " << STABILITY << " (must be < 0.25)" << endl;
    cout << endl;
    
    // Test different scheduling strategies
    vector<string> schedules = {"static", "dynamic", "guided"};
    vector<int> thread_counts = {1, 2, 4, 8,10,12};
    
    for (const string& sched : schedules) {
        cout << "=== " << sched << " Scheduling ===" << endl;
        cout << left << setw(10) << "Threads" << setw(15) << "Time (s)"
             << setw(15) << "Speedup" << setw(15) << "Efficiency"
             << "Avg Temp" << endl;
        cout << string(70, '-') << endl;
        
        double t_serial = 0;
        
        for (int threads : thread_counts) {
            if (threads > omp_get_max_threads()) continue;
            
            double time, temp;
            heat_diffusion(N, time_steps, sched, threads, time, temp);
            
            if (threads == 1) t_serial = time;
            
            double speedup = t_serial / time;
            double efficiency = (speedup / threads) * 100.0;
            
            cout << left << setw(10) << threads
                 << setw(15) << fixed << setprecision(6) << time
                 << setw(15) << setprecision(2) << speedup << "x"
                 << setw(15) << setprecision(1) << efficiency << "%"
                 << setprecision(2) << temp << "°C" << endl;
        }
        cout << endl;
    }
    
    // Test cache-blocked version
    cout << "=== Cache-Blocked Version (Block size: 32) ===" << endl;
    cout << left << setw(10) << "Threads" << setw(15) << "Time (s)"
         << setw(15) << "Speedup" << "Efficiency" << endl;
    cout << string(55, '-') << endl;
    
    double t_serial_blocked = 0;
    for (int threads : thread_counts) {
        if (threads > omp_get_max_threads()) continue;
        
        double time, temp;
        heat_diffusion_blocked(N, time_steps, threads, time, temp);
        
        if (threads == 1) t_serial_blocked = time;
        
        double speedup = t_serial_blocked / time;
        double efficiency = (speedup / threads) * 100.0;
        
        cout << left << setw(10) << threads
             << setw(15) << fixed << setprecision(6) << time
             << setw(15) << setprecision(2) << speedup << "x"
             << setprecision(1) << efficiency << "%" << endl;
    }
    
    cout << "\n=== Analysis ===" << endl;
    cout << "1. No race conditions: Each grid point writes to unique location" << endl;
    cout << "2. Reduction used for total heat calculation" << endl;
    cout << "3. collapse(2) clause parallelizes both i and j loops" << endl;
    cout << "4. Static scheduling: Lowest overhead for uniform workload" << endl;
    cout << "5. Cache blocking: Improves data locality and cache hit rate" << endl;
    cout << "\nMemory Access Pattern:" << endl;
    cout << "- Each cell reads 5 values (center + 4 neighbors)" << endl;
    cout << "- Sequential access benefits from spatial locality" << endl;
    cout << "- Blocking improves temporal locality in cache" << endl;
    
    return 0;
}