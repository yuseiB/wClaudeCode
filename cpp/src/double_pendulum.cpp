#include "double_pendulum.hpp"

#include <cmath>

namespace mathphys {

// ─────────────────────────────────────────────────────────────────────────────
// Equations of motion
// ─────────────────────────────────────────────────────────────────────────────
std::array<double, 4>
DoublePendulum::eom(const std::array<double, 4>& s) const
{
    const double th1 = s[0], w1 = s[1], th2 = s[2], w2 = s[3];
    const double dlt = th1 - th2;
    const double D   = 2*m1 + m2 - m2 * std::cos(2*dlt);

    const double a1 = (
        -g * (2*m1 + m2) * std::sin(th1)
        - m2 * g * std::sin(th1 - 2*th2)
        - 2 * std::sin(dlt) * m2 * (w2*w2*L2 + w1*w1*L1*std::cos(dlt))
    ) / (L1 * D);

    const double a2 = (
        2 * std::sin(dlt) * (
            w1*w1*L1*(m1 + m2)
            + g*(m1 + m2)*std::cos(th1)
            + w2*w2*L2*m2*std::cos(dlt)
        )
    ) / (L2 * D);

    return {w1, a1, w2, a2};
}

// ─────────────────────────────────────────────────────────────────────────────
// Total mechanical energy  E = T + V
// ─────────────────────────────────────────────────────────────────────────────
double DoublePendulum::energy(const std::array<double, 4>& s) const
{
    const double th1 = s[0], w1 = s[1], th2 = s[2], w2 = s[3];
    const double dlt = th1 - th2;
    const double T =
        0.5*m1*L1*L1*w1*w1 +
        0.5*m2*(L1*L1*w1*w1 + L2*L2*w2*w2 + 2*L1*L2*w1*w2*std::cos(dlt));
    const double V = -g * ((m1 + m2)*L1*std::cos(th1) + m2*L2*std::cos(th2));
    return T + V;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixed-step RK4 integrator
// ─────────────────────────────────────────────────────────────────────────────
SimResult DoublePendulum::simulate(double th1_0, double th2_0,
                                   double  w1_0, double  w2_0,
                                   double t_end, double dt) const
{
    const int n = static_cast<int>(t_end / dt) + 1;

    SimResult res;
    res.t.reserve(n);
    res.theta1.reserve(n); res.omega1.reserve(n);
    res.theta2.reserve(n); res.omega2.reserve(n);
    res.x2.reserve(n);     res.y2.reserve(n);
    res.energy.reserve(n);

    std::array<double, 4> y = {th1_0, w1_0, th2_0, w2_0};
    double t = 0.0;

    auto rk4 = [&](const std::array<double,4>& y, double h)
    {
        auto k1 = eom(y);
        std::array<double,4> y2, y3, y4;
        for (int i = 0; i < 4; ++i) y2[i] = y[i] + h/2 * k1[i];
        auto k2 = eom(y2);
        for (int i = 0; i < 4; ++i) y3[i] = y[i] + h/2 * k2[i];
        auto k3 = eom(y3);
        for (int i = 0; i < 4; ++i) y4[i] = y[i] + h * k3[i];
        auto k4 = eom(y4);
        std::array<double,4> yn;
        for (int i = 0; i < 4; ++i)
            yn[i] = y[i] + h/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
        return yn;
    };

    for (int i = 0; i < n; ++i) {
        res.t.push_back(t);
        res.theta1.push_back(y[0]); res.omega1.push_back(y[1]);
        res.theta2.push_back(y[2]); res.omega2.push_back(y[3]);
        res.x2.push_back( L1 * std::sin(y[0]) + L2 * std::sin(y[2]));
        res.y2.push_back(-L1 * std::cos(y[0]) - L2 * std::cos(y[2]));
        res.energy.push_back(energy(y));
        y  = rk4(y, dt);
        t += dt;
    }
    return res;
}

} // namespace mathphys
