#include "ising_model.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

IsingModel2D::IsingModel2D(int n, double j, uint64_t seed)
    : n(n), j(j),
      rng_(seed),
      row_dist_(0, n - 1),
      col_dist_(0, n - 1),
      uniform01_(0.0, 1.0)
{
    if (n <= 0) throw std::invalid_argument("Lattice size n must be positive");
    // Random ±1 initialisation
    std::uniform_int_distribution<int> spin_dist(0, 1);
    lattice.assign(n, std::vector<int8_t>(n));
    for (int i = 0; i < n; ++i)
        for (int jj = 0; jj < n; ++jj)
            lattice[i][jj] = spin_dist(rng_) == 0 ? -1 : 1;
}

// ---------------------------------------------------------------------------
// Observables
// ---------------------------------------------------------------------------

double IsingModel2D::energy() const {
    double e = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int jj = 0; jj < n; ++jj) {
            int s = lattice[i][jj];
            // right and down neighbours (PBC)
            int right = lattice[i][(jj + 1) % n];
            int down  = lattice[(i + 1) % n][jj];
            e += s * (right + down);
        }
    }
    return -j * e;
}

double IsingModel2D::magnetization() const {
    double m = 0.0;
    for (int i = 0; i < n; ++i)
        for (int jj = 0; jj < n; ++jj)
            m += lattice[i][jj];
    return m;
}

// ---------------------------------------------------------------------------
// Monte Carlo
// ---------------------------------------------------------------------------

void IsingModel2D::metropolis_step(double temperature) {
    const double beta = 1.0 / temperature;
    const int n2 = n * n;

    // Pre-compute acceptance probabilities for ΔE/J ∈ {4, 8}
    double exp4 = std::exp(-beta * j * 4.0);
    double exp8 = std::exp(-beta * j * 8.0);

    for (int k = 0; k < n2; ++k) {
        int i  = row_dist_(rng_);
        int jj = col_dist_(rng_);
        int s  = lattice[i][jj];

        int nb = lattice[(i - 1 + n) % n][jj]
               + lattice[(i + 1)     % n][jj]
               + lattice[i][(jj - 1 + n) % n]
               + lattice[i][(jj + 1)     % n];

        int dE_over_J = 2 * s * nb;  // in units of J

        double prob = 1.0;
        if      (dE_over_J == 4) prob = exp4;
        else if (dE_over_J == 8) prob = exp8;

        if (dE_over_J <= 0 || uniform01_(rng_) < prob)
            lattice[i][jj] = static_cast<int8_t>(-s);
    }
}

SimResult IsingModel2D::simulate(double temperature, int n_therm, int n_measure) {
    const double n2 = static_cast<double>(n * n);

    for (int k = 0; k < n_therm; ++k)
        metropolis_step(temperature);

    double e_sum = 0, e2_sum = 0, m_sum = 0, m2_sum = 0;
    for (int k = 0; k < n_measure; ++k) {
        metropolis_step(temperature);
        double e = energy() / n2;
        double m = std::abs(magnetization()) / n2;
        e_sum  += e;
        e2_sum += e * e;
        m_sum  += m;
        m2_sum += m * m;
    }

    double e_mean  = e_sum  / n_measure;
    double e2_mean = e2_sum / n_measure;
    double m_mean  = m_sum  / n_measure;
    double m2_mean = m2_sum / n_measure;

    SimResult res;
    res.T      = temperature;
    res.E_mean = e_mean;
    res.E2_mean = e2_mean;
    res.M_mean = m_mean;
    res.M2_mean = m2_mean;
    res.Cv     = (e2_mean - e_mean * e_mean) / (temperature * temperature);
    res.chi    = (m2_mean - m_mean * m_mean) / temperature;
    return res;
}
