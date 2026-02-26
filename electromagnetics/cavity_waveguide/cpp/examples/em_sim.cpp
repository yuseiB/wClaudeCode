/**
 * em_sim.cpp — 電磁キャビティ・導波管シミュレーション サンプルプログラム
 *
 * 出力: CSV ファイル
 *   rect_cavity_modes.csv   — 直方体キャビティ: 最低次 10 モードの共振周波数
 *   rect_wg_dispersion.csv  — 直方体導波管 TE₁₀ の分散関係 β vs f
 *   circ_wg_modes.csv       — 円形導波管: 最低次 8 モードのカットオフ周波数
 *
 * コンパイル例:
 *   cmake --build build && ./build/em_sim
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "em_cavity.hpp"
#include "em_waveguide.hpp"

using namespace emcavity;
using namespace emwaveguide;
using namespace emconst;

// ─────────────────────────────────────────────────────────────────────────────
// 直方体キャビティ: 最低次モードの共振周波数 CSV 出力
// ─────────────────────────────────────────────────────────────────────────────
static void output_rect_cavity_modes() {
    constexpr double a = 0.04, b = 0.02, d = 0.03;  // 4×2×3 cm キャビティ

    auto modes = rectangular_cavity_modes(a, b, d, 10);

    std::ofstream ofs("rect_cavity_modes.csv");
    // CSV ヘッダー
    ofs << "mode,freq_GHz\n";
    for (const auto& m : modes) {
        ofs << std::setprecision(6)
            << m.label() << ","
            << m.resonant_frequency() * 1e-9 << "\n";
    }

    std::cout << "Rectangular cavity modes (a=4cm, b=2cm, d=3cm):\n";
    for (const auto& m : modes) {
        std::cout << "  " << std::setw(8) << m.label()
                  << "  f = " << std::fixed << std::setprecision(4)
                  << m.resonant_frequency() * 1e-9 << " GHz\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 直方体導波管 TE₁₀: 分散関係 β vs f の CSV 出力
// ─────────────────────────────────────────────────────────────────────────────
static void output_rect_wg_dispersion() {
    constexpr double a = 0.02286, b = 0.01016;  // WR-90

    RectangularWaveguideMode te10(a, b, 1, 0, "TE");
    RectangularWaveguideMode te20(a, b, 2, 0, "TE");
    RectangularWaveguideMode te01(a, b, 0, 1, "TE");

    double f_max = 30e9;  // 30 GHz まで計算
    int    N     = 300;

    std::ofstream ofs("rect_wg_dispersion.csv");
    ofs << "freq_GHz,beta_TE10,beta_TE20,beta_TE01,beta_lightline\n";

    for (int i = 0; i <= N; ++i) {
        double f = f_max * i / N;
        // 光の分散 β_light = 2π f / c (参照線)
        double b_light = 2.0 * PI * f / C_LIGHT;
        ofs << std::fixed << std::setprecision(4)
            << f * 1e-9 << ","
            << te10.propagation_constant(f) << ","
            << te20.propagation_constant(f) << ","
            << te01.propagation_constant(f) << ","
            << b_light << "\n";
    }

    std::cout << "\nWR-90 waveguide dispersion:\n";
    std::cout << "  TE_10 fc = " << std::fixed << std::setprecision(3)
              << te10.cutoff_frequency() * 1e-9 << " GHz\n";
    std::cout << "  TE_20 fc = "
              << te20.cutoff_frequency() * 1e-9 << " GHz\n";
    std::cout << "  TE_01 fc = "
              << te01.cutoff_frequency() * 1e-9 << " GHz\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 円形導波管: 最低次 8 モードのカットオフ周波数 CSV 出力
// ─────────────────────────────────────────────────────────────────────────────
static void output_circ_wg_modes() {
    constexpr double R = 0.010;  // 半径 10 mm

    auto modes = circular_waveguide_modes(R, 8);

    std::ofstream ofs("circ_wg_modes.csv");
    ofs << "mode,fc_GHz\n";
    for (const auto& m : modes) {
        ofs << m.label() << ","
            << std::fixed << std::setprecision(4)
            << m.cutoff_frequency() * 1e-9 << "\n";
    }

    std::cout << "\nCircular waveguide modes (R=10mm):\n";
    for (const auto& m : modes) {
        std::cout << "  " << std::setw(6) << m.label()
                  << "  fc = " << std::fixed << std::setprecision(3)
                  << m.cutoff_frequency() * 1e-9 << " GHz\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// メイン
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== EM Cavity & Waveguide Simulation ===\n\n";

    output_rect_cavity_modes();
    output_rect_wg_dispersion();
    output_circ_wg_modes();

    std::cout << "\nCSV files written:\n"
              << "  rect_cavity_modes.csv\n"
              << "  rect_wg_dispersion.csv\n"
              << "  circ_wg_modes.csv\n";

    return 0;
}
