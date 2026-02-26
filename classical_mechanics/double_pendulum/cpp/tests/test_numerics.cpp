#include "numerics.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// Minimal test harness (no external deps)
static int failures = 0;

#define CHECK_CLOSE(actual, expected, tol)                                          \
    do {                                                                            \
        double _a = (actual), _e = (expected), _t = (tol);                         \
        if (std::abs(_a - _e) > _t) {                                               \
            std::printf("FAIL [%s:%d]  got %.10g  expected %.10g  (tol %.2e)\n",   \
                        __FILE__, __LINE__, _a, _e, _t);                            \
            ++failures;                                                             \
        } else {                                                                    \
            std::printf("PASS  got %.10g\n", _a);                                  \
        }                                                                           \
    } while (0)

int main() {
    // --- integrate_trapezoid ---
    {
        // integral of 1 over [0,1] == 1
        const int N = 1000;
        std::vector<double> x(N), f(N, 1.0);
        for (int i = 0; i < N; ++i) x[i] = static_cast<double>(i) / (N - 1);
        CHECK_CLOSE(mathphys::integrate_trapezoid(f, x), 1.0, 1e-9);
    }
    {
        // integral of x^2 over [0,1] == 1/3
        const int N = 10000;
        std::vector<double> x(N), f(N);
        for (int i = 0; i < N; ++i) {
            x[i] = static_cast<double>(i) / (N - 1);
            f[i] = x[i] * x[i];
        }
        CHECK_CLOSE(mathphys::integrate_trapezoid(f, x), 1.0 / 3.0, 1e-5);
    }

    // --- finite_difference (order 1) ---
    {
        // derivative of 3.7*x + 1.5 == 3.7
        const int N = 100;
        std::vector<double> x(N), f(N);
        for (int i = 0; i < N; ++i) {
            x[i] = static_cast<double>(i) / (N - 1);
            f[i] = 3.7 * x[i] + 1.5;
        }
        auto df = mathphys::finite_difference(f, x, 1);
        CHECK_CLOSE(df[N / 2], 3.7, 1e-9);
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) FAILED.\n", failures);
    return 1;
}
