/**
 * em_constants.hpp — 電磁気計算で共通に使う物理定数 (SI 単位)
 *
 * すべて constexpr で定義するため、コンパイル時に評価される。
 */

#pragma once

namespace emconst {

/// 真空中の光速 (m/s)
inline constexpr double C_LIGHT = 2.99792458e8;

/// 真空の透磁率 μ₀ (H/m)
inline constexpr double MU0 = 1.25663706212e-6;

/// 真空の誘電率 ε₀ (F/m)  ε₀ = 1/(μ₀c²)
inline constexpr double EPS0 = 8.8541878128e-12;

/// 真空インピーダンス η₀ = μ₀c ≈ 376.73 Ω
inline constexpr double ETA0 = MU0 * C_LIGHT;

/// π
inline constexpr double PI = 3.14159265358979323846;

} // namespace emconst
