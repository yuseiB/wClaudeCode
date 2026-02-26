/**
 * em_waveguide.cpp — 直方体・円形導波管 EM 伝搬の実装
 */

#include "em_waveguide.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace emwaveguide {

// ─────────────────────────────────────────────────────────────────────────────
// Bessel 零点テーブル (waveguide も cavity と同じ値を使用)
// ─────────────────────────────────────────────────────────────────────────────

static constexpr double TM_ZEROS_WG[5][3] = {
    {2.4048, 5.5201, 8.6537},
    {3.8317, 7.0156, 10.1735},
    {5.1356, 8.4172, 11.6198},
    {6.3802, 9.7610, 13.0152},
    {7.5883, 11.0647, 14.3725},
};

static constexpr double TE_ZEROS_WG[5][3] = {
    {3.8317, 7.0156, 10.1735},
    {1.8412, 5.3314, 8.5363},  // TE₁₁ が最低次 (χ'₁₁ ≈ 1.8412)
    {3.0542, 6.7061, 9.9695},
    {4.2012, 8.0152, 11.3459},
    {5.3175, 9.2824, 12.6819},
};

double CircularWaveguideMode::bessel_tm_zero(int m, int n) {
    if (m < 0 || m > 4 || n < 1 || n > 3)
        throw std::out_of_range("Bessel TM zeros: m=0..4, n=1..3 only");
    return TM_ZEROS_WG[m][n - 1];
}

double CircularWaveguideMode::bessel_te_zero(int m, int n) {
    if (m < 0 || m > 4 || n < 1 || n > 3)
        throw std::out_of_range("Bessel TE zeros: m=0..4, n=1..3 only");
    return TE_ZEROS_WG[m][n - 1];
}

// ─────────────────────────────────────────────────────────────────────────────
// 直方体導波管: バリデーション・事前計算・場の計算
// ─────────────────────────────────────────────────────────────────────────────

void RectangularWaveguideMode::validate() const {
    if (mode_type_ == "TE") {
        if (m_ == 0 && n_ == 0)
            throw std::invalid_argument("TE: m and n cannot both be 0");
        if (m_ < 0 || n_ < 0)
            throw std::invalid_argument("TE: m, n must be >= 0");
    } else if (mode_type_ == "TM") {
        if (m_ < 1 || n_ < 1)
            throw std::invalid_argument("TM: m and n must be >= 1");
    } else {
        throw std::invalid_argument("mode_type must be 'TE' or 'TM'");
    }
}

void RectangularWaveguideMode::precompute() {
    kx_  = m_ * PI / a_;
    ky_  = n_ * PI / b_;
    kc_  = std::sqrt(kx_*kx_ + ky_*ky_);
    fc_  = C_LIGHT * kc_ / (2.0 * PI);
    kc2_ = kc_ * kc_;
}

double RectangularWaveguideMode::wave_impedance(double frequency) const {
    double k    = 2.0 * PI * frequency / C_LIGHT;
    double beta = propagation_constant(frequency);
    if (beta == 0.0) return std::numeric_limits<double>::infinity();
    if (mode_type_ == "TE") {
        return ETA0 * k / beta;
    } else {
        return ETA0 * beta / k;
    }
}

EMPoint RectangularWaveguideMode::fields(double x, double y, double z,
                                          double frequency, double phase) const
{
    EMPoint f;
    const double beta = propagation_constant(frequency);
    // 進行波の位相因子: cos(ωt − βz + phase₀)
    const double psi  = std::cos(phase - beta * z);

    const double Cx = std::cos(kx_ * x);
    const double Sx = std::sin(kx_ * x);
    const double Cy = std::cos(ky_ * y);
    const double Sy = std::sin(ky_ * y);

    if (mode_type_ == "TE") {
        // TE モード: H_z = cos(kx·x) cos(ky·y) exp(j(ωt−βz))
        f.Hz = Cx * Cy * psi;
        if (kc2_ > 0.0) {
            // H_t = (β/kc²) ∇_T H_z
            f.Hx = (beta * kx_ / kc2_) * Sx * Cy * psi;
            f.Hy = (beta * ky_ / kc2_) * Cx * Sy * psi;
            // E_t = -(ωμ₀/kc²) ẑ×∇_T H_z
            f.Ex = -(2.0 * PI * frequency * MU0 * ky_ / kc2_) * Cx * Sy * psi;
            f.Ey = ( 2.0 * PI * frequency * MU0 * kx_ / kc2_) * Sx * Cy * psi;
        }

    } else {
        // TM モード: E_z = sin(kx·x) sin(ky·y) exp(j(ωt−βz))
        f.Ez = Sx * Sy * psi;
        if (kc2_ > 0.0) {
            // E_t = (-β/kc²) ∇_T E_z
            f.Ex = -(beta * kx_ / kc2_) * Cx * Sy * psi;
            f.Ey = -(beta * ky_ / kc2_) * Sx * Cy * psi;
            // H_t = (ωε₀/kc²) ẑ×∇_T E_z
            f.Hx = ( 2.0 * PI * frequency * EPS0 * ky_ / kc2_) * Sx * Cy * psi;
            f.Hy = -(2.0 * PI * frequency * EPS0 * kx_ / kc2_) * Cx * Sy * psi;
        }
    }
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// 円形導波管: バリデーション・事前計算・場の計算
// ─────────────────────────────────────────────────────────────────────────────

void CircularWaveguideMode::validate() const {
    if (n_ < 1) throw std::invalid_argument("n must be >= 1");
    if (m_ < 0) throw std::invalid_argument("m must be >= 0");
    if (mode_type_ != "TE" && mode_type_ != "TM")
        throw std::invalid_argument("mode_type must be 'TE' or 'TM'");
}

void CircularWaveguideMode::precompute() {
    if (mode_type_ == "TM") {
        chi_ = bessel_tm_zero(m_, n_);
    } else {
        chi_ = bessel_te_zero(m_, n_);
    }
    kc_  = chi_ / R_;
    fc_  = C_LIGHT * kc_ / (2.0 * PI);
    kc2_ = kc_ * kc_;
}

EMPoint CircularWaveguideMode::fields_polar(double rho, double phi, double z,
                                             double frequency, double phase) const
{
    EMPoint f;
    const double beta = propagation_constant(frequency);
    const double psi  = std::cos(phase - beta * z);
    const double omega = 2.0 * PI * frequency;

    // Bessel 関数の計算 (引数 kc·ρ)
    const double kc_rho  = kc_ * rho;
    const double jm      = Jm(m_, kc_rho);
    const double djm_dr  = dJm(m_, kc_rho) * kc_;  // dJ_m/dρ

    const double cos_mphi = std::cos(static_cast<double>(m_) * phi);
    const double sin_mphi = std::sin(static_cast<double>(m_) * phi);
    const double rho_safe = std::max(rho, 1.0e-30);

    if (mode_type_ == "TM") {
        // TM: E_z = J_m(kc·ρ) cos(mφ) exp(j(ωt−βz))
        f.Ez = jm * cos_mphi * psi;
        if (kc2_ > 0.0) {
            f.Ex = -(beta * kc_ / kc2_) * dJm(m_, kc_rho) * cos_mphi * psi;
            f.Ey = (beta * m_ / (kc2_ * rho_safe)) * jm * sin_mphi * psi;
            f.Hx = -(omega * EPS0 * m_ / (kc2_ * rho_safe)) * jm * sin_mphi * psi;
            f.Hy = (omega * EPS0 * kc_ / kc2_) * dJm(m_, kc_rho) * cos_mphi * psi;
        }
    } else {
        // TE: H_z = J_m(kc·ρ) cos(mφ) exp(j(ωt−βz))
        f.Hz = jm * cos_mphi * psi;
        if (kc2_ > 0.0) {
            f.Hx = -(beta * kc_ / kc2_) * dJm(m_, kc_rho) * cos_mphi * psi;
            f.Hy = (beta * m_ / (kc2_ * rho_safe)) * jm * sin_mphi * psi;
            f.Ex = (omega * MU0 * m_ / (kc2_ * rho_safe)) * jm * sin_mphi * psi;
            f.Ey = -(omega * MU0 * kc_ / kc2_) * dJm(m_, kc_rho) * cos_mphi * psi;
        }
    }
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// モードリスト生成
// ─────────────────────────────────────────────────────────────────────────────

std::vector<CircularWaveguideMode>
circular_waveguide_modes(double R, int n_modes)
{
    std::vector<CircularWaveguideMode> modes;
    for (int m = 0; m <= 4; ++m) {
        for (int n = 1; n <= 3; ++n) {
            for (const auto& t : {"TE", "TM"}) {
                try {
                    modes.emplace_back(R, m, n, t);
                } catch (...) {}
            }
        }
    }
    std::sort(modes.begin(), modes.end(),
        [](const auto& a, const auto& b) {
            return a.cutoff_frequency() < b.cutoff_frequency();
        });
    if (static_cast<int>(modes.size()) > n_modes)
        modes.erase(modes.begin() + n_modes, modes.end());
    return modes;
}

std::vector<RectangularWaveguideMode>
rectangular_waveguide_modes(double a, double b, int n_modes)
{
    std::vector<RectangularWaveguideMode> modes;
    for (int m = 0; m <= 5; ++m) {
        for (int n = 0; n <= 5; ++n) {
            for (const auto& t : {"TE", "TM"}) {
                try {
                    modes.emplace_back(a, b, m, n, t);
                } catch (...) {}
            }
        }
    }
    std::sort(modes.begin(), modes.end(),
        [](const auto& a, const auto& b) {
            return a.cutoff_frequency() < b.cutoff_frequency();
        });
    if (static_cast<int>(modes.size()) > n_modes)
        modes.erase(modes.begin() + n_modes, modes.end());
    return modes;
}

} // namespace emwaveguide
