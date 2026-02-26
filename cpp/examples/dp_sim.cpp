/**
 * dp_sim.cpp — run eight double-pendulum scenarios and write CSV output.
 *
 * Compile (from cpp/examples/):
 *   g++ -std=c++20 -O2 -I../include dp_sim.cpp ../src/double_pendulum.cpp -o dp_sim_exec
 * Run:
 *   ./dp_sim_exec
 */
#include "double_pendulum.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace mathphys;

static constexpr double PI = 3.14159265358979323846;

static void write_csv(const std::string& path, const SimResult& r)
{
    std::ofstream f(path);
    f << std::setprecision(10);
    f << "t,theta1,omega1,theta2,omega2,x2,y2,energy\n";
    for (std::size_t i = 0; i < r.t.size(); ++i) {
        f << r.t[i]      << ',' << r.theta1[i] << ',' << r.omega1[i] << ','
          << r.theta2[i] << ',' << r.omega2[i] << ','
          << r.x2[i]     << ',' << r.y2[i]     << ',' << r.energy[i] << '\n';
    }
}

int main()
{
    DoublePendulum dp;   // m1=m2=L1=L2=1 kg/m, g=9.81 m/s²

    // ── section 1: three dynamical regimes ───────────────────────────────────
    write_csv("case_nearlinear.csv",
              dp.simulate(10*PI/180,   10*PI/180,  0, 0, 30.0, 0.005));
    write_csv("case_intermediate.csv",
              dp.simulate(90*PI/180,   0,          0, 0, 30.0, 0.005));
    write_csv("case_chaotic.csv",
              dp.simulate(120*PI/180, -30*PI/180,  0, 0, 30.0, 0.005));

    // ── section 2: sensitivity to initial conditions ─────────────────────────
    const double delta = 0.001 * PI / 180;
    write_csv("sensitivity_a.csv",
              dp.simulate(120*PI/180,         -30*PI/180, 0, 0, 25.0, 0.005));
    write_csv("sensitivity_b.csv",
              dp.simulate(120*PI/180 + delta, -30*PI/180, 0, 0, 25.0, 0.005));

    // ── section 3: mass ratios ────────────────────────────────────────────────
    const std::vector<std::pair<double, std::string>> mass_cases = {
        {0.25, "mass_ratio_0.25.csv"},
        {1.00, "mass_ratio_1.00.csv"},
        {4.00, "mass_ratio_4.00.csv"},
    };
    for (const auto& [ratio, fname] : mass_cases) {
        DoublePendulum dp_m(1.0, ratio, 1.0, 1.0, 9.81);
        write_csv(fname, dp_m.simulate(90*PI/180, 0, 0, 0, 30.0, 0.005));
    }

    std::cout << "Wrote 8 CSV files.\n";
    return 0;
}
