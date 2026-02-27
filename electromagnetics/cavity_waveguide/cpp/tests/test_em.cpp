/**
 * test_em.cpp — 電磁気キャビティ・導波管モジュールの単体テスト (CTest)
 *
 * テスト設計:
 *   - 解析解が既知の量 (共振周波数、カットオフ周波数) を厳密に検証
 *   - 境界条件の確認 (PEC 壁での Ez=0 など)
 *   - バリデーション: 無効な入力で例外が送出されること
 *   - モードリストがソート済みであること
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "em_cavity.hpp"
#include "em_waveguide.hpp"

using namespace emcavity;
using namespace emwaveguide;
using namespace emconst;

// ── ユーティリティ ────────────────────────────────────────────────────────────

/// 相対誤差が tol 以内かチェック
static bool rel_close(double a, double b, double tol = 1e-9) {
    if (std::abs(b) < 1e-30) return std::abs(a) < tol;
    return std::abs(a - b) / std::abs(b) < tol;
}

/// 絶対値が tol 以内かチェック
static bool abs_close(double a, double tol = 1e-12) {
    return std::abs(a) < tol;
}

/// テスト結果の出力ヘルパー
static int g_passed = 0, g_failed = 0;

static void check(bool cond, const std::string& name) {
    if (cond) {
        std::cout << "[PASS] " << name << "\n";
        ++g_passed;
    } else {
        std::cerr << "[FAIL] " << name << "\n";
        ++g_failed;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 直方体キャビティのテスト
// ─────────────────────────────────────────────────────────────────────────────

static void test_rectangular_cavity() {
    constexpr double a = 0.04, b = 0.02, d = 0.03;  // 4×2×3 cm

    // ── TE₁₀₁ 共振周波数の解析解との一致 ──
    {
        RectangularCavityMode mode(a, b, d, 1, 0, 1, "TE");
        // f = (c/2) √((1/a)² + (1/d)²)
        double f_expected = (C_LIGHT / 2.0)
            * std::sqrt(1.0/(a*a) + 1.0/(d*d));
        check(rel_close(mode.resonant_frequency(), f_expected),
              "TE_101 resonant frequency");
    }

    // ── TM₁₁₀ 共振周波数 ──
    {
        RectangularCavityMode mode(a, b, d, 1, 1, 0, "TM");
        double f_expected = (C_LIGHT / 2.0)
            * std::sqrt(1.0/(a*a) + 1.0/(b*b));
        check(rel_close(mode.resonant_frequency(), f_expected),
              "TM_110 resonant frequency");
    }

    // ── TE: m=n=0 は例外 ──
    {
        bool threw = false;
        try { RectangularCavityMode(a, b, d, 0, 0, 1, "TE"); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "TE: m=n=0 throws");
    }

    // ── TM: m=0 は例外 ──
    {
        bool threw = false;
        try { RectangularCavityMode(a, b, d, 0, 1, 0, "TM"); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "TM: m=0 throws");
    }

    // ── 位相: ωt=0 で E が最大、H = 0 ──
    {
        RectangularCavityMode mode(a, b, d, 1, 0, 1, "TE");
        // y = b/2, x = a/2, z = d/2 (場が最大となる内部点)
        auto fld = mode.fields(a/2, b/2, d/2, 0.0);  // phase=0
        // H は sin(ωt=0) = 0 → Hx,Hy,Hz ≈ 0
        check(abs_close(fld.Hz, 1e-12) && abs_close(fld.Hx, 1e-12),
              "TE_101: H=0 at phase=0");
        // E は cos(0) = 1 → 非零
        check(fld.E_mag() > 0.0, "TE_101: E nonzero at phase=0");
    }

    // ── PEC 境界: x=0 で Ex=0, Ez=0 ──
    {
        RectangularCavityMode mode(a, b, d, 1, 0, 1, "TE");
        // x=0 では sin(kx·0) = 0 → Ex, Ez = 0 が期待される
        auto fld = mode.fields(0.0, b/2, d/2, 0.0);
        check(abs_close(fld.Ez), "TE_101: Ez=0 at x=0");
    }

    // ── モードリストがソート済み ──
    {
        auto modes = rectangular_cavity_modes(a, b, d, 8);
        bool sorted = true;
        for (size_t i = 1; i < modes.size(); ++i) {
            if (modes[i].resonant_frequency() < modes[i-1].resonant_frequency())
                sorted = false;
        }
        check(sorted, "rectangular_cavity_modes: sorted by frequency");
    }

    // ── label() ──
    {
        RectangularCavityMode mode(a, b, d, 1, 0, 1, "TE");
        check(mode.label() == "TE_101", "RectangularCavityMode::label()");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 円筒型キャビティのテスト
// ─────────────────────────────────────────────────────────────────────────────

static void test_cylindrical_cavity() {
    constexpr double R = 0.015, L = 0.030;  // R=1.5 cm, L=3 cm

    // ── TM₀₁₀ 共振周波数 ──
    {
        CylindricalCavityMode mode(R, L, 0, 1, 0, "TM");
        // χ₀₁ ≈ 2.4048, f = c·χ₀₁ / (2π R)
        constexpr double CHI01 = 2.4048;
        double f_expected = C_LIGHT * CHI01 / (2.0 * PI * R);
        check(rel_close(mode.resonant_frequency(), f_expected, 1e-4),
              "TM_010 resonant frequency (chi_01 approx)");
    }

    // ── TM モードで Hz = 0 ──
    {
        CylindricalCavityMode mode(R, L, 0, 1, 0, "TM");
        auto fld = mode.fields_rz(R * 0.5, L / 2, 0.0, 0.5);
        check(abs_close(fld.Hz), "TM: Hz=0");
    }

    // ── TE モードで Ez = 0 ──
    {
        CylindricalCavityMode mode(R, L, 1, 1, 1, "TE");
        auto fld = mode.fields_rz(R * 0.5, L / 2, 0.0, 0.5);
        check(abs_close(fld.Ez), "TE: Ez=0");
    }

    // ── n=0 は例外 ──
    {
        bool threw = false;
        try { CylindricalCavityMode(R, L, 0, 0, 1, "TM"); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "CylindricalCavity: n=0 throws");
    }

    // ── モードリストがソート済み ──
    {
        auto modes = cylindrical_cavity_modes(R, L, 6);
        bool sorted = true;
        for (size_t i = 1; i < modes.size(); ++i) {
            if (modes[i].resonant_frequency() < modes[i-1].resonant_frequency())
                sorted = false;
        }
        check(sorted, "cylindrical_cavity_modes: sorted");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 直方体導波管のテスト
// ─────────────────────────────────────────────────────────────────────────────

static void test_rectangular_waveguide() {
    // WR-90 標準導波管
    constexpr double a = 0.02286, b = 0.01016;

    // ── TE₁₀ カットオフ周波数: fc = c/(2a) ──
    {
        RectangularWaveguideMode mode(a, b, 1, 0, "TE");
        double fc_expected = C_LIGHT / (2.0 * a);
        check(rel_close(mode.cutoff_frequency(), fc_expected),
              "TE_10 cutoff frequency");
    }

    // ── TE₂₀ は TE₁₀ の 2 倍 ──
    {
        RectangularWaveguideMode m10(a, b, 1, 0, "TE");
        RectangularWaveguideMode m20(a, b, 2, 0, "TE");
        check(rel_close(m20.cutoff_frequency(), 2.0 * m10.cutoff_frequency()),
              "TE_20 fc = 2 x TE_10 fc");
    }

    // ── fc 以上では β > 0 ──
    {
        RectangularWaveguideMode mode(a, b, 1, 0, "TE");
        double fc   = mode.cutoff_frequency();
        double beta = mode.propagation_constant(fc * 1.5);
        check(beta > 0.0, "TE_10: beta > 0 above cutoff");
    }

    // ── fc 以下では β = 0 (エバネッセント) ──
    {
        RectangularWaveguideMode mode(a, b, 1, 0, "TE");
        double fc   = mode.cutoff_frequency();
        double beta = mode.propagation_constant(fc * 0.5);
        check(beta == 0.0, "TE_10: beta=0 below cutoff");
    }

    // ── β の解析式: β = √((ω/c)² − kc²) との一致 ──
    {
        RectangularWaveguideMode mode(a, b, 1, 0, "TE");
        double fc   = mode.cutoff_frequency();
        double f    = fc * 1.5;
        double kc   = mode.cutoff_frequency() * 2.0 * PI / C_LIGHT;  // kc = 2π fc/c
        double beta_formula = std::sqrt(
            std::pow(2.0 * PI * f / C_LIGHT, 2) - kc * kc);
        check(rel_close(mode.propagation_constant(f), beta_formula),
              "TE_10 beta formula");
    }

    // ── TE: m=n=0 は例外 ──
    {
        bool threw = false;
        try { RectangularWaveguideMode(a, b, 0, 0, "TE"); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "RectWaveguide TE: m=n=0 throws");
    }

    // ── label() ──
    {
        RectangularWaveguideMode mode(a, b, 1, 0, "TE");
        check(mode.label() == "TE_10", "RectangularWaveguideMode::label()");
    }

    // ── モードリストがソート済み ──
    {
        auto modes = rectangular_waveguide_modes(a, b, 6);
        bool sorted = true;
        for (size_t i = 1; i < modes.size(); ++i) {
            if (modes[i].cutoff_frequency() < modes[i-1].cutoff_frequency())
                sorted = false;
        }
        check(sorted, "rectangular_waveguide_modes: sorted");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 円形導波管のテスト
// ─────────────────────────────────────────────────────────────────────────────

static void test_circular_waveguide() {
    constexpr double R = 0.010;  // 半径 10 mm

    // ── TE₁₁ カットオフ: fc = c·χ'₁₁ / (2π R), χ'₁₁ ≈ 1.8412 ──
    {
        CircularWaveguideMode mode(R, 1, 1, "TE");
        constexpr double CHI_P11 = 1.8412;
        double fc_expected = C_LIGHT * CHI_P11 / (2.0 * PI * R);
        check(rel_close(mode.cutoff_frequency(), fc_expected, 1e-4),
              "TE_11 cutoff frequency");
    }

    // ── TM₀₁ カットオフ: fc = c·χ₀₁ / (2π R), χ₀₁ ≈ 2.4048 ──
    {
        CircularWaveguideMode mode(R, 0, 1, "TM");
        constexpr double CHI01 = 2.4048;
        double fc_expected = C_LIGHT * CHI01 / (2.0 * PI * R);
        check(rel_close(mode.cutoff_frequency(), fc_expected, 1e-4),
              "TM_01 cutoff frequency");
    }

    // ── TE₁₁ が最低次 ──
    {
        auto modes = circular_waveguide_modes(R, 8);
        check(!modes.empty() &&
              modes[0].label() == "TE_11",
              "TE_11 is lowest-order circular waveguide mode");
    }

    // ── TE モードで Ez = 0 ──
    {
        CircularWaveguideMode mode(R, 1, 1, "TE");
        double fc  = mode.cutoff_frequency();
        auto fld = mode.fields_polar(R * 0.5, 0.0, 0.0, fc * 1.5, 0.0);
        check(abs_close(fld.Ez), "TE circular: Ez=0");
    }

    // ── TM モードで Hz = 0 ──
    {
        CircularWaveguideMode mode(R, 0, 1, "TM");
        double fc  = mode.cutoff_frequency();
        auto fld = mode.fields_polar(R * 0.5, 0.0, 0.0, fc * 1.5, 0.0);
        check(abs_close(fld.Hz), "TM circular: Hz=0");
    }

    // ── n=0 は例外 ──
    {
        bool threw = false;
        try { CircularWaveguideMode(R, 0, 0, "TE"); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "CircularWaveguide: n=0 throws");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// メイン
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== EM Cavity & Waveguide Tests ===\n\n";

    std::cout << "-- Rectangular Cavity --\n";
    test_rectangular_cavity();

    std::cout << "\n-- Cylindrical Cavity --\n";
    test_cylindrical_cavity();

    std::cout << "\n-- Rectangular Waveguide --\n";
    test_rectangular_waveguide();

    std::cout << "\n-- Circular Waveguide --\n";
    test_circular_waveguide();

    std::cout << "\n================================\n";
    std::cout << "Passed: " << g_passed << " / " << (g_passed + g_failed) << "\n";

    return (g_failed == 0) ? 0 : 1;
}
