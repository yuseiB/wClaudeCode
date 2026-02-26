/**
 * em_cavity.cpp — 直方体・円筒型キャビティ EM 解析の実装
 *
 * 実装の方針:
 *   - 解析解を直接評価 (数値積分なし)
 *   - Bessel 零点は近似値テーブルを使用 (m≤4, n≤3 の範囲)
 *   - 境界条件は数式の構造で自動的に満たされる
 */

#include "em_cavity.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace emcavity {

// ─────────────────────────────────────────────────────────────────────────────
// Bessel 零点テーブル
//
// J_m(χ_mn) = 0 の零点 χ_mn (TM モード用)
// J_m'(χ'_mn) = 0 の零点 χ'_mn (TE モード用)
//
// 値は Pozar "Microwave Engineering" Table 3.4, 3.5 より
// ─────────────────────────────────────────────────────────────────────────────

/// J_m の零点 χ_mn: rows = m (0..4), cols = n (1..3)
static constexpr double TM_ZEROS[5][3] = {
    {2.4048, 5.5201, 8.6537},  // m=0: χ₀₁, χ₀₂, χ₀₃
    {3.8317, 7.0156, 10.1735}, // m=1: χ₁₁, χ₁₂, χ₁₃
    {5.1356, 8.4172, 11.6198}, // m=2
    {6.3802, 9.7610, 13.0152}, // m=3
    {7.5883, 11.0647, 14.3725},// m=4
};

/// J_m' の零点 χ'_mn: rows = m (0..4), cols = n (1..3)
static constexpr double TE_ZEROS[5][3] = {
    {3.8317, 7.0156, 10.1735}, // m=0: χ'₀₁, χ'₀₂, χ'₀₃
    {1.8412, 5.3314, 8.5363},  // m=1: χ'₁₁ が最小 (TE₁₁ 最低次)
    {3.0542, 6.7061, 9.9695},  // m=2
    {4.2012, 8.0152, 11.3459}, // m=3
    {5.3175, 9.2824, 12.6819}, // m=4
};

double CylindricalCavityMode::bessel_tm_zero(int m, int n) {
    // インデックス範囲チェック
    if (m < 0 || m > 4 || n < 1 || n > 3) {
        throw std::out_of_range(
            "Bessel TM zeros available for m=0..4, n=1..3 only");
    }
    return TM_ZEROS[m][n - 1];
}

double CylindricalCavityMode::bessel_te_zero(int m, int n) {
    if (m < 0 || m > 4 || n < 1 || n > 3) {
        throw std::out_of_range(
            "Bessel TE zeros available for m=0..4, n=1..3 only");
    }
    return TE_ZEROS[m][n - 1];
}

// ─────────────────────────────────────────────────────────────────────────────
// 直方体キャビティ: バリデーションと事前計算
// ─────────────────────────────────────────────────────────────────────────────

void RectangularCavityMode::validate() const {
    if (mode_type_ == "TE") {
        if (m_ == 0 && n_ == 0)
            throw std::invalid_argument("TE: m and n cannot both be 0");
        if (p_ < 1)
            throw std::invalid_argument("TE: p must be >= 1");
    } else if (mode_type_ == "TM") {
        if (m_ < 1 || n_ < 1)
            throw std::invalid_argument("TM: m and n must be >= 1");
        if (p_ < 0)
            throw std::invalid_argument("TM: p must be >= 0");
    } else {
        throw std::invalid_argument("mode_type must be 'TE' or 'TM'");
    }
}

void RectangularCavityMode::precompute() {
    // 各方向の波数 kx=mπ/a, ky=nπ/b, kz=pπ/d
    kx_  = m_ * PI / a_;
    ky_  = n_ * PI / b_;
    kz_  = p_ * PI / d_;
    k_   = std::sqrt(kx_*kx_ + ky_*ky_ + kz_*kz_);
    kc2_ = kx_*kx_ + ky_*ky_;   // 横断面波数の 2 乗
    omega_ = C_LIGHT * k_;       // ω = c·k
}

// ─────────────────────────────────────────────────────────────────────────────
// 直方体キャビティ: EM 場の計算
// ─────────────────────────────────────────────────────────────────────────────

EMPoint RectangularCavityMode::fields(double x, double y, double z,
                                       double phase) const
{
    EMPoint f;
    const double ct = std::cos(phase);  // cos(ωt): E に掛かる時間因子
    const double st = std::sin(phase);  // sin(ωt): H に掛かる時間因子

    // 空間的な三角関数を事前計算
    const double Cx = std::cos(kx_ * x);
    const double Sx = std::sin(kx_ * x);
    const double Cy = std::cos(ky_ * y);
    const double Sy = std::sin(ky_ * y);
    const double Cz = std::cos(kz_ * z);
    const double Sz = std::sin(kz_ * z);

    if (mode_type_ == "TE") {
        // TE モード: H_z = cos(kx·x) cos(ky·y) sin(kz·z) · sin(ωt)
        // 導出: E = -(ωμ₀/kc²) ẑ×∇_T H_z,  H = (-jβ/kc²) ∇_T H_z + ẑ H_z
        if (kc2_ > 0.0) {
            f.Ex = ( omega_ * MU0 * ky_ / kc2_) * Cx * Sy * Sz * ct;
            f.Ey = (-omega_ * MU0 * kx_ / kc2_) * Sx * Cy * Sz * ct;
            f.Hx = -(kx_ * kz_ / kc2_) * Sx * Cy * Cz * st;
            f.Hy = -(ky_ * kz_ / kc2_) * Cx * Sy * Cz * st;
        }
        f.Ez = 0.0;
        f.Hz = Cx * Cy * Sz * st;   // H_z ~ sin(kx·x) cos(ky·y) ... ではなく cos·cos

    } else {
        // TM モード: E_z = sin(kx·x) sin(ky·y) cos(kz·z) · cos(ωt)
        // 導出: E = (-jβ/kc²) ∇_T E_z + ẑ E_z,  H = (ωε₀/kc²) ẑ×∇_T E_z
        if (kc2_ > 0.0) {
            f.Ex = (kx_ * kz_ / kc2_) * Cx * Sy * Sz * ct;
            f.Ey = (ky_ * kz_ / kc2_) * Sx * Cy * Sz * ct;
            f.Hx = ( omega_ * EPS0 * ky_ / kc2_) * Sx * Cy * Cz * st;
            f.Hy = (-omega_ * EPS0 * kx_ / kc2_) * Cx * Sy * Cz * st;
        }
        f.Ez = Sx * Sy * Cz * ct;
        f.Hz = 0.0;
    }
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// 直方体キャビティ: モードリスト生成
// ─────────────────────────────────────────────────────────────────────────────

std::vector<RectangularCavityMode>
rectangular_cavity_modes(double a, double b, double d, int n_modes)
{
    std::vector<RectangularCavityMode> modes;
    const int MAX_IDX = 5;

    // 全組み合わせを試みてバリデーション通過したものを収集
    for (int m = 0; m <= MAX_IDX; ++m) {
        for (int n = 0; n <= MAX_IDX; ++n) {
            for (int p = 0; p <= MAX_IDX; ++p) {
                for (const auto& t : {"TE", "TM"}) {
                    try {
                        modes.emplace_back(a, b, d, m, n, p, t);
                    } catch (const std::invalid_argument&) {
                        // 無効なモード指数の組み合わせはスキップ
                    }
                }
            }
        }
    }

    // 共振周波数昇順にソート
    std::sort(modes.begin(), modes.end(),
        [](const auto& a, const auto& b) {
            return a.resonant_frequency() < b.resonant_frequency();
        });

    if (static_cast<int>(modes.size()) > n_modes)
        modes.erase(modes.begin() + n_modes, modes.end());
    return modes;
}

// ─────────────────────────────────────────────────────────────────────────────
// 円筒型キャビティ: バリデーションと事前計算
// ─────────────────────────────────────────────────────────────────────────────

void CylindricalCavityMode::validate() const {
    if (n_ < 1) throw std::invalid_argument("n must be >= 1");
    if (m_ < 0) throw std::invalid_argument("m must be >= 0");
    if (mode_type_ == "TM") {
        if (p_ < 0) throw std::invalid_argument("TM: p must be >= 0");
    } else if (mode_type_ == "TE") {
        if (p_ < 1) throw std::invalid_argument("TE: p must be >= 1");
    } else {
        throw std::invalid_argument("mode_type must be 'TM' or 'TE'");
    }
}

void CylindricalCavityMode::precompute() {
    // Bessel 零点 χ を取得
    if (mode_type_ == "TM") {
        chi_ = bessel_tm_zero(m_, n_);
    } else {
        chi_ = bessel_te_zero(m_, n_);
    }
    kc_    = chi_ / R_;            // 横断面波数
    kz_    = (L_ > 0.0) ? p_ * PI / L_ : 0.0;  // 軸方向波数
    omega_ = C_LIGHT * std::sqrt(kc_*kc_ + kz_*kz_);
}

// ─────────────────────────────────────────────────────────────────────────────
// 円筒型キャビティ: EM 場の計算 (ρ-z 断面)
// ─────────────────────────────────────────────────────────────────────────────

EMPoint CylindricalCavityMode::fields_rz(double rho, double z,
                                           double phi, double phase) const
{
    EMPoint f;
    const double ct = std::cos(phase);
    const double st = std::sin(phase);

    // Bessel 関数の値と導関数
    const double kc_rho = kc_ * rho;
    const double jm      = Jm(m_, kc_rho);         // J_m(kc·ρ)
    const double djm     = dJm(m_, kc_rho) * kc_;  // dJ_m/dρ (chain rule)

    const double cos_mphi = std::cos(m_ * phi);
    const double sin_mphi = std::sin(m_ * phi);
    const double Cz = std::cos(kz_ * z);
    const double Sz = std::sin(kz_ * z);

    // ρ → 0 での特異点を回避する微小値
    const double rho_safe = std::max(rho, 1e-30);

    if (mode_type_ == "TM") {
        // TM モード: E_z = J_m(kc·ρ) cos(mφ) cos(kz·z) · cos(ωt)
        // 境界条件: ρ=R で J_m(χ_mn) = 0 → Ez=0 (自動的に満たされる)
        f.Ez = jm * cos_mphi * Cz * ct;
        if (kc_ > 0.0) {
            // E_ρ = -(kz/kc) J_m'(kc·ρ) cos(mφ) sin(kz·z)
            f.Ex = -(kz_ / kc_) * djm * cos_mphi * Sz * ct;
            // E_φ = (kz·m / (kc²·ρ)) J_m(kc·ρ) sin(mφ) sin(kz·z)
            f.Ey = (kz_ * m_ / (kc_*kc_ * rho_safe)) * jm * sin_mphi * Sz * ct;
            // H_ρ = (ωε₀·m / (kc²·ρ)) J_m sin(mφ) cos(kz·z)
            f.Hx = ( omega_ * EPS0 * m_ / (kc_*kc_ * rho_safe)) * jm * sin_mphi * Cz * st;
            // H_φ = -(ωε₀/kc) J_m'(kc·ρ) cos(mφ) cos(kz·z)
            f.Hy = -(omega_ * EPS0 / kc_) * djm * cos_mphi * Cz * st;
        }
        f.Hz = 0.0;
    } else {
        // TE モード: H_z = J_m(kc·ρ) cos(mφ) sin(kz·z) · sin(ωt)
        // 境界条件: ρ=R で J_m'(χ'_mn) = 0 → E_φ=0 (自動的に満たされる)
        f.Hz = jm * cos_mphi * Sz * st;
        if (kc_ > 0.0) {
            f.Hx = -(kz_ / kc_) * djm * cos_mphi * Cz * st;
            f.Hy = (kz_ * m_ / (kc_*kc_ * rho_safe)) * jm * sin_mphi * Cz * st;
            f.Ex = -(omega_ * MU0 * m_ / (kc_*kc_ * rho_safe)) * jm * sin_mphi * Sz * ct;
            f.Ey = -(omega_ * MU0 / kc_) * djm * cos_mphi * Sz * ct;
        }
        f.Ez = 0.0;
    }
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// 円筒型キャビティ: モードリスト生成
// ─────────────────────────────────────────────────────────────────────────────

std::vector<CylindricalCavityMode>
cylindrical_cavity_modes(double R, double L, int n_modes)
{
    std::vector<CylindricalCavityMode> modes;

    for (int m = 0; m <= 4; ++m) {
        for (int n = 1; n <= 3; ++n) {
            for (int p = 0; p <= 3; ++p) {
                for (const auto& t : {"TM", "TE"}) {
                    try {
                        modes.emplace_back(R, L, m, n, p, t);
                    } catch (...) {
                        // 無効なモードはスキップ
                    }
                }
            }
        }
    }

    std::sort(modes.begin(), modes.end(),
        [](const auto& a, const auto& b) {
            return a.resonant_frequency() < b.resonant_frequency();
        });

    if (static_cast<int>(modes.size()) > n_modes)
        modes.erase(modes.begin() + n_modes, modes.end());
    return modes;
}

} // namespace emcavity
