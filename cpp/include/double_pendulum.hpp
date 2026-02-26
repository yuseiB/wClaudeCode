#pragma once
#include <array>
#include <string>
#include <vector>

namespace mathphys {

/// Time-series results from a double-pendulum simulation.
struct SimResult {
    std::vector<double> t;
    std::vector<double> theta1, omega1;
    std::vector<double> theta2, omega2;
    std::vector<double> x2, y2;   ///< Cartesian position of bob-2
    std::vector<double> energy;   ///< Total mechanical energy
};

/**
 * Exact nonlinear double pendulum (Lagrangian formulation).
 *
 * Angles measured from the downward vertical.
 * State vector: [θ₁, ω₁, θ₂, ω₂].
 * Integrated with a fixed-step 4th-order Runge-Kutta scheme.
 */
class DoublePendulum {
public:
    double m1, m2;   ///< Bob masses [kg]
    double L1, L2;   ///< Rod lengths [m]
    double g;        ///< Gravitational acceleration [m s⁻²]

    explicit DoublePendulum(double m1 = 1.0, double m2 = 1.0,
                            double L1 = 1.0, double L2 = 1.0,
                            double g  = 9.81)
        : m1(m1), m2(m2), L1(L1), L2(L2), g(g) {}

    /// RHS of the first-order ODE system.
    std::array<double, 4> eom(const std::array<double, 4>& state) const;

    /// Total mechanical energy E = T + V.
    double energy(const std::array<double, 4>& state) const;

    /**
     * Simulate from given initial angles (radians) and angular velocities.
     * @param t_end  integration end time [s]
     * @param dt     fixed time step [s]
     */
    SimResult simulate(double theta1_0, double theta2_0,
                       double omega1_0 = 0.0, double omega2_0 = 0.0,
                       double t_end = 20.0,  double dt = 0.005) const;
};

} // namespace mathphys
