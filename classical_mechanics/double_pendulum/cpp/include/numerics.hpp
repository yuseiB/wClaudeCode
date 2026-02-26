#pragma once
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace mathphys {

/// Trapezoidal-rule integration of f over x.
double integrate_trapezoid(std::span<const double> f, std::span<const double> x);

/// Central finite-difference derivative (order 1 or 2).
std::vector<double> finite_difference(std::span<const double> f,
                                      std::span<const double> x,
                                      int order = 1);

}  // namespace mathphys
