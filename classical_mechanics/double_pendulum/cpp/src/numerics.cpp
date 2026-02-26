#include "numerics.hpp"

#include <cmath>
#include <stdexcept>

namespace mathphys {

double integrate_trapezoid(std::span<const double> f, std::span<const double> x) {
    if (f.size() != x.size() || f.size() < 2) {
        throw std::invalid_argument("f and x must have the same size >= 2");
    }
    double sum = 0.0;
    for (std::size_t i = 1; i < f.size(); ++i) {
        sum += 0.5 * (f[i] + f[i - 1]) * (x[i] - x[i - 1]);
    }
    return sum;
}

std::vector<double> finite_difference(std::span<const double> f,
                                      std::span<const double> x,
                                      int order) {
    const std::size_t n = f.size();
    if (n < 3) throw std::invalid_argument("Need at least 3 points");
    if (order != 1 && order != 2) throw std::invalid_argument("Order must be 1 or 2");

    std::vector<double> df(n, 0.0);

    if (order == 1) {
        // Forward difference at left boundary
        df[0] = (f[1] - f[0]) / (x[1] - x[0]);
        // Central differences in interior
        for (std::size_t i = 1; i + 1 < n; ++i) {
            df[i] = (f[i + 1] - f[i - 1]) / (x[i + 1] - x[i - 1]);
        }
        // Backward difference at right boundary
        df[n - 1] = (f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2]);
    } else {
        // Second derivative: central difference in interior
        df[0] = (f[2] - 2 * f[1] + f[0]) / std::pow(x[1] - x[0], 2);
        for (std::size_t i = 1; i + 1 < n; ++i) {
            double h = (x[i + 1] - x[i - 1]) / 2.0;
            df[i] = (f[i + 1] - 2 * f[i] + f[i - 1]) / (h * h);
        }
        df[n - 1] = df[n - 2];  // extrapolate at boundary
    }
    return df;
}

}  // namespace mathphys
