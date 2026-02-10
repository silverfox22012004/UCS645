// Exercise 1: Molecular Dynamics - Lennard-Jones Force Calculation
// Correct OpenMP implementation with proper physics and synchronization

#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <random>

using namespace std;

struct Particle {
    double x, y, z;
    double fx, fy, fz;
};

// Lennard-Jones parameters
const double EPSILON = 1.0;
const double SIGMA   = 1.0;
const double CUTOFF  = 2.5 * SIGMA;
const double CUTOFF2 = CUTOFF * CUTOFF;

// Lennard-Jones interaction (pairwise)
inline void lj_interaction(double dx, double dy, double dz,
                           double &fx, double &fy, double &fz,
                           double &potential) {

    double r2 = dx*dx + dy*dy + dz*dz;

    if (r2 > CUTOFF2 || r2 < 1e-12) {
        fx = fy = fz = potential = 0.0;
        return;
    }

    double inv_r2 = 1.0 / r2;
    double r6  = inv_r2 * inv_r2 * inv_r2;
    double r12 = r6 * r6;

    potential = 4.0 * EPSILON * (r12 - r6);

    double fmag = 24.0 * EPSILON * inv_r2 * (2.0 * r12 - r6);

    fx = fmag * dx;
    fy = fmag * dy;
    fz = fmag * dz;
}

int main() {

    int N = 1000;
    vector<Particle> particles(N);

    // Initialize particle positions
    mt19937 gen(42);
    uniform_real_distribution<> dis(0.0, 100.0);

    for (int i = 0; i < N; i++) {
        particles[i] = {dis(gen), dis(gen), dis(gen), 0.0, 0.0, 0.0};
    }

    cout << "Molecular Dynamics: Lennard-Jones Force Calculation\n";
    cout << "Number of particles: " << N << endl;
    cout << "Cutoff distance: " << CUTOFF << "\n\n";

    vector<int> thread_counts = {1, 2, 4, 8, 10, 12};

    cout << left << setw(10) << "Threads"
         << setw(15) << "Time (s)"
         << setw(15) << "Speedup"
         << setw(15) << "Efficiency"
         << "Total Energy\n";
    cout << string(80, '=') << endl;

    double t_serial = 0.0;

    for (int threads : thread_counts) {

        if (threads > omp_get_max_threads()) continue;

        // Reset forces
        for (auto &p : particles)
            p.fx = p.fy = p.fz = 0.0;

        double total_potential = 0.0;
        double start = omp_get_wtime();

        // Parallel pairwise interaction
        #pragma omp parallel num_threads(threads) reduction(+:total_potential)
        {
            #pragma omp for schedule(dynamic,16)
            for (int i = 0; i < N; i++) {
                for (int j = i + 1; j < N; j++) {

                    double fx, fy, fz, pot;
                    double dx = particles[i].x - particles[j].x;
                    double dy = particles[i].y - particles[j].y;
                    double dz = particles[i].z - particles[j].z;

                    lj_interaction(dx, dy, dz, fx, fy, fz, pot);

                    // Newton's Third Law
                    #pragma omp atomic
                    particles[i].fx += fx;
                    #pragma omp atomic
                    particles[i].fy += fy;
                    #pragma omp atomic
                    particles[i].fz += fz;

                    #pragma omp atomic
                    particles[j].fx -= fx;
                    #pragma omp atomic
                    particles[j].fy -= fy;
                    #pragma omp atomic
                    particles[j].fz -= fz;

                    total_potential += pot;
                }
            }
        }

        double elapsed = omp_get_wtime() - start;

        if (threads == 1)
            t_serial = elapsed;

        double speedup = t_serial / elapsed;
        double efficiency = (speedup / threads) * 100.0;

        cout << left << setw(10) << threads
             << setw(15) << fixed << setprecision(6) << elapsed
             << setw(15) << setprecision(2) << speedup << "x"
             << setw(15) << setprecision(1) << efficiency << "%"
             << setprecision(2) << total_potential << endl;
    }

    cout << string(80, '=') << endl;

    cout << "\nOptimization Techniques Applied:\n";
    cout << "1. OpenMP parallelization of nested loops\n";
    cout << "2. Newton's third law to avoid double counting\n";
    cout << "3. Reduction for total potential energy\n";
    cout << "4. Dynamic scheduling for load balancing\n";
    cout << "5. Cutoff distance to reduce computations\n";

    return 0;
}
