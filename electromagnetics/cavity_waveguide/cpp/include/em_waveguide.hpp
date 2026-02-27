/**
 * em_waveguide.hpp — 電磁波の導波管伝搬解析 (C++20)
 *
 * 対応導波管:
 *   RectangularWaveguideMode : 直方体 PEC 導波管 a×b の TE/TM モード
 *   CircularWaveguideMode    : 円形 PEC 導波管 (半径 R) の TE/TM モード
 *
 * 物理的な背景:
 *   - 伝搬方向: +z
 *   - カットオフ周波数 fc 以下: エバネッセント (指数減衰)
 *   - カットオフ以上: 進行波 E,H ∝ exp(j(ωt − βz))
 *   - 分散関係: β = √((ω/c)² − kc²)
 *   - 群速度 vg = dω/dβ < c, 位相速度 vp = ω/β > c
 *
 * 参考文献:
 *   Pozar, "Microwave Engineering", 4th ed., ch. 3
 */

#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "em_constants.hpp"
#include "em_cavity.hpp"   // EMPoint を再利用

namespace emwaveguide {

using namespace emconst;
using emcavity::EMPoint;

// ─────────────────────────────────────────────────────────────────────────────
// 直方体導波管
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 直方体 PEC 導波管 (0≤x≤a, 0≤y≤b, 伝搬方向: +z) の TE/TM モード。
 *
 * TE_mn (E_z=0): m,n≥0, 両方 0 は不可
 * TM_mn (H_z=0): m,n≥1
 *
 * カットオフ波数: kc = π√((m/a)² + (n/b)²)
 * カットオフ周波数: fc = c·kc / (2π)
 * 伝搬定数: β = √((ω/c)² − kc²)  [f > fc で実数]
 */
class RectangularWaveguideMode {
public:
    /**
     * @param a,b       導波管断面寸法 (m)
     * @param m,n       モード指数 (非負整数)
     * @param mode_type "TE" または "TM"
     */
    RectangularWaveguideMode(double a, double b,
                              int m, int n,
                              const std::string& mode_type)
        : a_(a), b_(b), m_(m), n_(n), mode_type_(mode_type)
    {
        validate();
        precompute();
    }

    /// カットオフ周波数 (Hz)
    [[nodiscard]] double cutoff_frequency() const { return fc_; }

    /**
     * 伝搬定数 β (rad/m)。
     * f < fc ではエバネッセントで β = 0 を返す (虚数部は省略)。
     *
     * @param frequency 動作周波数 (Hz)
     */
    [[nodiscard]] double propagation_constant(double frequency) const {
        double k  = 2.0 * PI * frequency / C_LIGHT;
        double d2 = k * k - kc_ * kc_;
        return (d2 >= 0.0) ? std::sqrt(d2) : 0.0;
    }

    /**
     * モード波インピーダンス (Ω)。
     * TE: Z = η₀·(k/β)
     * TM: Z = η₀·(β/k)
     */
    [[nodiscard]] double wave_impedance(double frequency) const;

    /**
     * 横断面 (x,y) の EM 場を計算する (軸方向位置 z での断面)。
     *
     * @param x,y       横断面座標 (m)
     * @param z         軸方向位置 (m)
     * @param frequency 動作周波数 (Hz)
     * @param phase     ωt (rad)
     */
    [[nodiscard]] EMPoint fields(double x, double y, double z,
                                  double frequency, double phase = 0.0) const;

    /// モードラベル (例: "TE_10")
    [[nodiscard]] std::string label() const {
        return mode_type_ + "_" + std::to_string(m_) + std::to_string(n_);
    }

private:
    double a_, b_;
    int    m_, n_;
    std::string mode_type_;

    double kx_, ky_;  ///< 横断面波数成分 (rad/m)
    double kc_;       ///< カットオフ波数 = √(kx²+ky²)
    double fc_;       ///< カットオフ周波数 (Hz)
    double kc2_;      ///< kc²

    void validate() const;
    void precompute();
};

// ─────────────────────────────────────────────────────────────────────────────
// 円形導波管
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 円形 PEC 導波管 (0≤ρ≤R, 伝搬方向: +z) の TE/TM モード。
 *
 * TM_mn (H_z=0): kc = χ_mn/R  (χ_mn = J_m の第 n 零点)
 * TE_mn (E_z=0): kc = χ'_mn/R (χ'_mn = J_m' の第 n 零点)
 *
 * TE₁₁ が最低次モード (χ'₁₁ ≈ 1.8412)。
 */
class CircularWaveguideMode {
public:
    /**
     * @param R         導波管半径 (m)
     * @param m,n       モード指数 (m≥0, n≥1)
     * @param mode_type "TE" または "TM"
     */
    CircularWaveguideMode(double R, int m, int n,
                           const std::string& mode_type)
        : R_(R), m_(m), n_(n), mode_type_(mode_type)
    {
        validate();
        precompute();
    }

    /// カットオフ周波数 (Hz)
    [[nodiscard]] double cutoff_frequency() const { return fc_; }

    /// 伝搬定数 β (rad/m)
    [[nodiscard]] double propagation_constant(double frequency) const {
        double k  = 2.0 * PI * frequency / C_LIGHT;
        double d2 = k * k - kc_ * kc_;
        return (d2 >= 0.0) ? std::sqrt(d2) : 0.0;
    }

    /**
     * 極座標 (rho, phi) の EM 場を計算する (軸方向位置 z での断面)。
     * 戻り値の (Ex,Ey,Ez) は (Eρ,Eφ,Ez) に対応。
     *
     * @param rho,phi   極座標 (m, rad)
     * @param z         軸方向位置 (m)
     * @param frequency 動作周波数 (Hz)
     * @param phase     ωt (rad)
     */
    [[nodiscard]] EMPoint fields_polar(double rho, double phi, double z,
                                        double frequency, double phase = 0.0) const;

    /// モードラベル (例: "TE_11")
    [[nodiscard]] std::string label() const {
        return mode_type_ + "_" + std::to_string(m_) + std::to_string(n_);
    }

private:
    double R_;
    int    m_, n_;
    std::string mode_type_;

    double chi_;   ///< Bessel 零点
    double kc_;    ///< カットオフ波数
    double fc_;    ///< カットオフ周波数 (Hz)
    double kc2_;

    void validate() const;
    void precompute();

    // Bessel 関数ユーティリティ
    static double Jm(int m, double x) {
        return std::cyl_bessel_j(static_cast<double>(m), x);
    }
    static double dJm(int m, double x) {
        if (m == 0) return -Jm(1, x);
        return 0.5 * (Jm(m - 1, x) - Jm(m + 1, x));
    }

    /// J_m 零点テーブル (TM)
    static double bessel_tm_zero(int m, int n);

    /// J_m' 零点テーブル (TE)
    static double bessel_te_zero(int m, int n);
};

/// 円形導波管の最低次 n_modes モードをカットオフ周波数昇順で返す
std::vector<CircularWaveguideMode> circular_waveguide_modes(double R, int n_modes = 8);

/// 直方体導波管の最低次 n_modes モードをカットオフ周波数昇順で返す
std::vector<RectangularWaveguideMode> rectangular_waveguide_modes(
    double a, double b, int n_modes = 8);

} // namespace emwaveguide
