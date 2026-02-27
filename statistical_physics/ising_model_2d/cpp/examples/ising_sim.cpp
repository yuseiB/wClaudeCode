/**
 * 2D Ising Model — Temperature Sweep
 *
 * Generates ising2d_sweep.csv with columns:
 *   T, E_mean, M_mean, Cv, chi
 *
 * Usage (from cpp/build/):
 *   cmake .. && cmake --build . && ./ising_sim
 */

#include "ising_model.hpp"

#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    const int    N       = 32;
    const int    N_THERM = 5'000;
    const int    N_MEAS  = 10'000;
    const double T_MIN   = 1.0;
    const double T_MAX   = 4.0;
    const int    N_T     = 40;

    std::printf("2D Ising Model sweep  N=%d  T=[%.1f, %.1f]  %d points\n",
                N, T_MIN, T_MAX, N_T);
    std::printf("Onsager T_c = %.4f\n\n", T_CRITICAL);

    std::vector<double> temps(N_T);
    for (int k = 0; k < N_T; ++k)
        temps[k] = T_MIN + k * (T_MAX - T_MIN) / (N_T - 1);

    // Open CSV
    std::FILE* fp = std::fopen("ising2d_sweep.csv", "w");
    std::fprintf(fp, "T,E_mean,M_mean,Cv,chi\n");

    for (int k = 0; k < N_T; ++k) {
        double T = temps[k];
        IsingModel2D model(N, 1.0, static_cast<uint64_t>(k));
        // Start ordered below Tc
        if (T < T_CRITICAL)
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    model.lattice[i][j] = 1;

        SimResult r = model.simulate(T, N_THERM, N_MEAS);
        std::fprintf(fp, "%.4f,%.6f,%.6f,%.6f,%.6f\n",
                     T, r.E_mean, r.M_mean, r.Cv, r.chi);

        if ((k + 1) % 10 == 0)
            std::printf("  %d/%d  T=%.2f  |M|=%.3f  Cv=%.3f\n",
                        k + 1, N_T, T, r.M_mean, r.Cv);
    }
    std::fclose(fp);
    std::printf("\nSaved → ising2d_sweep.csv\n");
    return 0;
}
