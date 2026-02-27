/**
 * em_cavity.hpp — 電磁気共振キャビティの解析解 (C++20)
 *
 * 対応キャビティ:
 *   RectangularCavityMode  : 直方体 PEC キャビティ a×b×d の TE/TM モード
 *   CylindricalCavityMode  : 円筒型 PEC キャビティ (半径 R, 高さ L) の TE/TM モード
 *
 * 物理的な背景:
 *   PEC 境界条件: 導体壁面で接線成分の電場 E_tan = 0
 *   時間依存性 : E(r,t) = E(r)·cos(ωt), H(r,t) = H(r)·sin(ωt)  [90° 位相差]
 *
 * 注意:
 *   Bessel 関数 (円筒キャビティ) は C++ 標準ライブラリ <cmath> の
 *   std::cyl_bessel_j() を使用 (C++17 以降で利用可能)。
 *
 * 参考文献:
 *   Pozar, "Microwave Engineering", 4th ed., ch. 6
 */

#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "em_constants.hpp"

namespace emcavity {

using namespace emconst;

// ─────────────────────────────────────────────────────────────────────────────
// 汎用: 3D 電磁場を表す構造体
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 1 点での電磁場の 6 成分 (Ex,Ey,Ez,Hx,Hy,Hz)。
 * 直交座標または対応する曲線座標で (Eρ,Eφ,Ez,...) にも使用。
 */
struct EMPoint {
    double Ex{0}, Ey{0}, Ez{0};  ///< 電場 (V/m)
    double Hx{0}, Hy{0}, Hz{0};  ///< 磁場 (A/m)

    /// 電場強度 |E|
    [[nodiscard]] double E_mag() const {
        return std::sqrt(Ex*Ex + Ey*Ey + Ez*Ez);
    }

    /// 磁場強度 |H|
    [[nodiscard]] double H_mag() const {
        return std::sqrt(Hx*Hx + Hy*Hy + Hz*Hz);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 直方体キャビティ
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 直方体 PEC キャビティ (0≤x≤a, 0≤y≤b, 0≤z≤d) の TE/TM モード。
 *
 * TE_mnp: m,n≥0 (両方 0 は不可), p≥1  (E_z = 0)
 * TM_mnp: m,n≥1, p≥0               (H_z = 0)
 *
 * 共振周波数: f = (c/2) √((m/a)² + (n/b)² + (p/d)²)
 */
class RectangularCavityMode {
public:
    /**
     * @param a,b,d     キャビティ寸法 (m)
     * @param m,n,p     モード指数 (非負整数)
     * @param mode_type "TE" または "TM"
     */
    RectangularCavityMode(double a, double b, double d,
                          int m, int n, int p,
                          const std::string& mode_type)
        : a_(a), b_(b), d_(d), m_(m), n_(n), p_(p), mode_type_(mode_type)
    {
        validate();
        precompute();
    }

    /// 共振周波数 (Hz)
    [[nodiscard]] double resonant_frequency() const { return omega_ / (2.0 * PI); }

    /// モードラベル (例: "TE_101")
    [[nodiscard]] std::string label() const {
        return mode_type_ + "_" + std::to_string(m_)
                          + std::to_string(n_)
                          + std::to_string(p_);
    }

    /**
     * 位置 (x, y, z) での EM 場を計算する。
     *
     * @param x,y,z  空間座標 (m)
     * @param phase  ωt (ラジアン)
     * @return       EMPoint 構造体 (6 成分)
     */
    [[nodiscard]] EMPoint fields(double x, double y, double z,
                                 double phase = 0.0) const;

private:
    double a_, b_, d_;
    int    m_, n_, p_;
    std::string mode_type_;

    // 事前計算値
    double kx_, ky_, kz_;   ///< 各方向の波数 (rad/m)
    double k_;               ///< 全波数 = √(kx²+ky²+kz²)
    double kc2_;             ///< 横断面波数の 2 乗 kx²+ky²
    double omega_;           ///< 角周波数 = c·k (rad/s)

    void validate() const;
    void precompute();
};

/// 直方体キャビティの最低次 n_modes モードを周波数昇順で返す
std::vector<RectangularCavityMode> rectangular_cavity_modes(
    double a, double b, double d, int n_modes = 10);

// ─────────────────────────────────────────────────────────────────────────────
// 円筒型キャビティ
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 円筒型 PEC キャビティ (0≤ρ≤R, 0≤z≤L) の TM/TE モード。
 *
 * TM_mnp: kc = χ_mn/R  (χ_mn = J_m の第 n 零点), p≥0
 * TE_mnp: kc = χ'_mn/R (χ'_mn = J_m' の第 n 零点), p≥1
 *
 * 共振周波数: f = (c/2π) √((χ/R)² + (pπ/L)²)
 *
 * Bessel 零点: 組み込みの近似値テーブルを使用 (最大 m=4, n=3 まで)
 */
class CylindricalCavityMode {
public:
    /**
     * @param R,L       キャビティ寸法 (m)
     * @param m,n,p     モード指数
     * @param mode_type "TM" または "TE"
     */
    CylindricalCavityMode(double R, double L,
                           int m, int n, int p,
                           const std::string& mode_type)
        : R_(R), L_(L), m_(m), n_(n), p_(p), mode_type_(mode_type)
    {
        validate();
        precompute();
    }

    /// 共振周波数 (Hz)
    [[nodiscard]] double resonant_frequency() const { return omega_ / (2.0 * PI); }

    /// モードラベル (例: "TM_010")
    [[nodiscard]] std::string label() const {
        return mode_type_ + "_" + std::to_string(m_)
                          + std::to_string(n_)
                          + std::to_string(p_);
    }

    /**
     * ρ-z 断面上の位置 (rho, z) での EM 場を計算する。
     * 戻り値の (Ex,Ey,Ez) は (Eρ,Eφ,Ez) に対応。
     *
     * @param rho  動径座標 (m), 0≤rho≤R
     * @param z    軸方向座標 (m), 0≤z≤L
     * @param phi  方位角 (rad), デフォルト 0
     * @param phase ωt (rad)
     */
    [[nodiscard]] EMPoint fields_rz(double rho, double z,
                                     double phi  = 0.0,
                                     double phase = 0.0) const;

private:
    double R_, L_;
    int    m_, n_, p_;
    std::string mode_type_;

    double chi_;    ///< Bessel 零点 χ_mn または χ'_mn
    double kc_;     ///< 横断面波数 = χ/R
    double kz_;     ///< 軸方向波数 = pπ/L
    double omega_;  ///< 角周波数

    void validate() const;
    void precompute();

    /// Bessel 関数 J_m(x) の近似 (C++ 標準 std::cyl_bessel_j を使用)
    static double Jm(int m, double x) {
        return std::cyl_bessel_j(static_cast<double>(m), x);
    }

    /// J_m'(x) ≈ (J_{m-1}(x) - J_{m+1}(x)) / 2  (再帰式)
    static double dJm(int m, double x) {
        if (m == 0) return -Jm(1, x);
        return 0.5 * (Jm(m - 1, x) - Jm(m + 1, x));
    }

    /// J_m の零点 (TM モード用), m=0..4, n=1..3 の近似値
    static double bessel_tm_zero(int m, int n);

    /// J_m' の零点 (TE モード用), m=0..4, n=1..3 の近似値
    static double bessel_te_zero(int m, int n);
};

/// 円筒型キャビティの最低次 n_modes モードを周波数昇順で返す
std::vector<CylindricalCavityMode> cylindrical_cavity_modes(
    double R, double L, int n_modes = 10);

} // namespace emcavity
