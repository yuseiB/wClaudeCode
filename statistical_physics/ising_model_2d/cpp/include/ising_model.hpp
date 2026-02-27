#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

// Onsager's exact critical temperature (J = k_B = 1)
inline constexpr double T_CRITICAL = 2.0 / 0.88137358702751547851434; // 2/ln(1+√2)

struct SimResult {
    double T;
    double E_mean;
    double E2_mean;
    double M_mean;
    double M2_mean;
    double Cv;
    double chi;
};

/**
 * 2D Ising model on an N×N square lattice with periodic boundary conditions.
 *
 * Hamiltonian: H = -J Σ_{<i,j>} s_i s_j  (nearest neighbours)
 * Spins: s_i ∈ {-1, +1}
 */
class IsingModel2D {
public:
    /**
     * @param n     lattice linear size (N×N sites)
     * @param j     coupling constant (> 0 for ferromagnet)
     * @param seed  RNG seed
     */
    IsingModel2D(int n, double j = 1.0, uint64_t seed = 42);

    // --- Observables ---

    /** Total energy E = -J Σ_{<i,j>} s_i s_j */
    double energy() const;

    /** Total magnetisation M = Σ_i s_i */
    double magnetization() const;

    // --- Dynamics ---

    /** One MC sweep: N² single-spin Metropolis-Hastings attempts. */
    void metropolis_step(double temperature);

    /** Thermalise then measure observables over n_measure sweeps. */
    SimResult simulate(double temperature,
                       int    n_therm   = 5'000,
                       int    n_measure = 10'000);

    // Direct lattice access for snapshots / tests
    std::vector<std::vector<int8_t>> lattice;  // lattice[row][col]
    int n;
    double j;

private:
    std::mt19937_64 rng_;
    std::uniform_int_distribution<int> row_dist_;
    std::uniform_int_distribution<int> col_dist_;
    std::uniform_real_distribution<double> uniform01_;

    int8_t& spin(int row, int col) {
        return lattice[((row % n) + n) % n][((col % n) + n) % n];
    }
    int8_t spin(int row, int col) const {
        return lattice[((row % n) + n) % n][((col % n) + n) % n];
    }
};
