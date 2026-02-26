/*!
 * constants.rs — 電磁気計算で使用する物理定数 (SI 単位)
 *
 * すべて定数式として定義。
 */

/// 真空中の光速 (m/s)
pub const C_LIGHT: f64 = 2.99792458e8;

/// 真空の透磁率 μ₀ (H/m)
pub const MU0: f64 = 1.25663706212e-6;

/// 真空の誘電率 ε₀ (F/m)
pub const EPS0: f64 = 8.8541878128e-12;

/// 真空インピーダンス η₀ = μ₀c ≈ 376.73 Ω
pub const ETA0: f64 = MU0 * C_LIGHT;

/// 円周率 π
pub const PI: f64 = std::f64::consts::PI;
