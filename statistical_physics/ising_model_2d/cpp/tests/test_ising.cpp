#include "ising_model.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

static int failures = 0;

#define CHECK_CLOSE(actual, expected, tol)                                         \
    do {                                                                           \
        double _a = (actual), _e = (expected), _t = (tol);                        \
        if (std::abs(_a - _e) > _t) {                                             \
            std::printf("FAIL [%s:%d]  got %.10g  expected %.10g  (tol %.2e)\n",  \
                        __FILE__, __LINE__, _a, _e, _t);                           \
            ++failures;                                                            \
        } else {                                                                   \
            std::printf("PASS  got %.10g  expected %.10g\n", _a, _e);             \
        }                                                                          \
    } while (0)

#define CHECK_TRUE(cond, msg)                                      \
    do {                                                           \
        if (!(cond)) {                                             \
            std::printf("FAIL [%s:%d]  %s\n", __FILE__, __LINE__, msg); \
            ++failures;                                            \
        } else {                                                   \
            std::printf("PASS  %s\n", msg);                       \
        }                                                          \
    } while (0)

// ---------------------------------------------------------------------------

void test_energy_all_up() {
    std::printf("\n--- test_energy_all_up ---\n");
    int n = 8;
    IsingModel2D model(n, 1.0, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            model.lattice[i][j] = 1;
    double expected = -2.0 * n * n;
    CHECK_CLOSE(model.energy(), expected, 1e-10);
}

void test_energy_all_down() {
    std::printf("\n--- test_energy_all_down ---\n");
    int n = 8;
    IsingModel2D model(n, 1.0, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            model.lattice[i][j] = -1;
    double expected = -2.0 * n * n;
    CHECK_CLOSE(model.energy(), expected, 1e-10);
}

void test_energy_checkerboard() {
    std::printf("\n--- test_energy_checkerboard ---\n");
    int n = 8;
    IsingModel2D model(n, 1.0, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            model.lattice[i][j] = ((i + j) % 2 == 0) ? 1 : -1;
    double expected = +2.0 * n * n;
    CHECK_CLOSE(model.energy(), expected, 1e-10);
}

void test_magnetization_all_up() {
    std::printf("\n--- test_magnetization_all_up ---\n");
    int n = 10;
    IsingModel2D model(n, 1.0, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            model.lattice[i][j] = 1;
    CHECK_CLOSE(model.magnetization(), static_cast<double>(n * n), 1e-10);
}

void test_low_temperature_ordering() {
    std::printf("\n--- test_low_temperature_ordering ---\n");
    int n = 16;
    IsingModel2D model(n, 1.0, 2);
    for (int k = 0; k < 2000; ++k)
        model.metropolis_step(0.5);
    double m = std::abs(model.magnetization()) / (n * n);
    std::printf("  |M|/N^2 = %.4f\n", m);
    CHECK_TRUE(m > 0.9, "|M|/N^2 > 0.9 at low T");
}

void test_phase_transition() {
    std::printf("\n--- test_phase_transition (simulate) ---\n");
    // Below Tc
    {
        IsingModel2D model(24, 1.0, 3);
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j)
                model.lattice[i][j] = 1;  // start ordered
        SimResult r = model.simulate(1.5, 3000, 5000);
        std::printf("  T=1.5  |M|/N^2 = %.3f\n", r.M_mean);
        CHECK_TRUE(r.M_mean > 0.5, "|M|>0.5 below Tc");
    }
    // Above Tc
    {
        IsingModel2D model(24, 1.0, 4);
        SimResult r = model.simulate(3.5, 3000, 5000);
        std::printf("  T=3.5  |M|/N^2 = %.3f\n", r.M_mean);
        CHECK_TRUE(r.M_mean < 0.3, "|M|<0.3 above Tc");
    }
}

// ---------------------------------------------------------------------------

int main() {
    std::printf("=== 2D Ising Model Tests (C++) ===\n");

    test_energy_all_up();
    test_energy_all_down();
    test_energy_checkerboard();
    test_magnetization_all_up();
    test_low_temperature_ordering();
    test_phase_transition();

    std::printf("\n=== %s  (%d failure(s)) ===\n",
                failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures;
}
