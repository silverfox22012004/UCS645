// Exercise 2: Bioinformatics - DNA Sequence Alignment (Smith-Waterman)
// Parallel implementation of local sequence alignment algorithm

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>
#include <iomanip>
#include <random>

using namespace std;

// Scoring parameters
const int MATCH = 2;
const int MISMATCH = -1;
const int GAP = -2;

// Generate random DNA sequence
string generate_dna(int length, int seed) {
    string dna;
    const char bases[] = {'A', 'C', 'G', 'T'};
    mt19937 gen(seed);
    uniform_int_distribution<> dis(0, 3);
    
    for (int i = 0; i < length; i++) {
        dna += bases[dis(gen)];
    }
    return dna;
}

// Calculate similarity score
int score(char a, char b) {
    return (a == b) ? MATCH : MISMATCH;
}

// Smith-Waterman algorithm with wavefront parallelization
void smith_waterman_parallel(const string& seq1, const string& seq2, 
                             int num_threads, double& time_taken) {
    int m = seq1.length();
    int n = seq2.length();
    
    // Create scoring matrix
    vector<vector<int>> H(m + 1, vector<int>(n + 1, 0));
    
    double start_time = omp_get_wtime();
    
    // Wavefront parallelization
    // Process anti-diagonals in parallel
    for (int diag = 1; diag <= m + n - 1; diag++) {
        int start_i = max(1, diag - n + 1);
        int end_i = min(m, diag);
        
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int i = start_i; i <= end_i; i++) {
            int j = diag - i + 1;
            if (j >= 1 && j <= n) {
                int match = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
                int delete_gap = H[i-1][j] + GAP;
                int insert_gap = H[i][j-1] + GAP;
                
                H[i][j] = max({0, match, delete_gap, insert_gap});
            }
        }
    }
    
    double end_time = omp_get_wtime();
    time_taken = end_time - start_time;
    
    // Find maximum score
    int max_score = 0;
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            max_score = max(max_score, H[i][j]);
        }
    }
}

// Simple parallel version with row-wise parallelization
void smith_waterman_simple(const string& seq1, const string& seq2,
                          int num_threads, double& time_taken) {
    int m = seq1.length();
    int n = seq2.length();
    
    vector<vector<int>> H(m + 1, vector<int>(n + 1, 0));
    
    double start_time = omp_get_wtime();
    
    // Sequential column computation (dependencies prevent full parallelization)
    for (int i = 1; i <= m; i++) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int j = 1; j <= n; j++) {
            int match = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
            int delete_gap = H[i-1][j] + GAP;
            int insert_gap = H[i][j-1] + GAP;
            
            H[i][j] = max({0, match, delete_gap, insert_gap});
        }
    }
    
    double end_time = omp_get_wtime();
    time_taken = end_time - start_time;
}

int main() {
    // Generate test sequences
    int seq_length = 500;
    string seq1 = generate_dna(seq_length, 42);
    string seq2 = generate_dna(seq_length, 123);
    
    // Add some matching subsequence for realistic alignment
    int match_start = seq_length / 3;
    int match_len = 50;
    for (int i = 0; i < match_len; i++) {
        seq2[match_start + i] = seq1[match_start + i];
    }
    
    cout << "DNA Sequence Alignment (Smith-Waterman Algorithm)" << endl;
    cout << "Sequence 1 length: " << seq1.length() << endl;
    cout << "Sequence 2 length: " << seq2.length() << endl;
    cout << "Scoring: MATCH=" << MATCH << ", MISMATCH=" << MISMATCH 
         << ", GAP=" << GAP << "\n" << endl;
    
    cout << "=== Wavefront Parallelization (Anti-diagonal) ===" << endl;
    cout << left << setw(10) << "Threads" << setw(15) << "Time (s)"
         << setw(15) << "Speedup" << "Efficiency" << endl;
    cout << string(55, '-') << endl;
    
    vector<int> thread_counts = {1, 2, 4, 8,10,12};
    double t_serial_wavefront = 0;
    
    for (int threads : thread_counts) {
        if (threads > omp_get_max_threads()) continue;
        
        double time;
        smith_waterman_parallel(seq1, seq2, threads, time);
        
        if (threads == 1) t_serial_wavefront = time;
        
        double speedup = t_serial_wavefront / time;
        double efficiency = (speedup / threads) * 100.0;
        
        cout << left << setw(10) << threads 
             << setw(15) << fixed << setprecision(6) << time
             << setw(15) << setprecision(2) << speedup << "x"
             << setprecision(1) << efficiency << "%" << endl;
    }
    
    cout << "\n=== Row-wise Parallelization (Simpler, but limited) ===" << endl;
    cout << left << setw(10) << "Threads" << setw(15) << "Time (s)"
         << setw(15) << "Speedup" << "Efficiency" << endl;
    cout << string(55, '-') << endl;
    
    double t_serial_simple = 0;
    
    for (int threads : thread_counts) {
        if (threads > omp_get_max_threads()) continue;
        
        double time;
        smith_waterman_simple(seq1, seq2, threads, time);
        
        if (threads == 1) t_serial_simple = time;
        
        double speedup = t_serial_simple / time;
        double efficiency = (speedup / threads) * 100.0;
        
        cout << left << setw(10) << threads 
             << setw(15) << fixed << setprecision(6) << time
             << setw(15) << setprecision(2) << speedup << "x"
             << setprecision(1) << efficiency << "%" << endl;
    }
    
    cout << "\nAlgorithm Analysis:" << endl;
    cout << "1. Dynamic Programming with O(mn) time complexity" << endl;
    cout << "2. Anti-dependencies limit parallelization" << endl;
    cout << "3. Wavefront method: process anti-diagonals in parallel" << endl;
    cout << "4. Each cell depends only on: H[i-1][j-1], H[i-1][j], H[i][j-1]" << endl;
    
    cout << "\nScheduling Strategy:" << endl;
    cout << "- Dynamic scheduling handles varying diagonal lengths" << endl;
    cout << "- Load balancing important as diagonal sizes change" << endl;
    
    return 0;
}