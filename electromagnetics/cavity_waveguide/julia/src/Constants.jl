"""定数モジュール — 電磁気計算で使用する物理定数 (SI 単位)"""
module Constants

"真空中の光速 (m/s)"
const C_LIGHT = 2.99792458e8

"真空の透磁率 μ₀ (H/m)"
const MU0 = 4e-7 * π

"真空の誘電率 ε₀ (F/m)"
const EPS0 = 1.0 / (MU0 * C_LIGHT^2)

"真空インピーダンス η₀ ≈ 376.73 Ω"
const ETA0 = MU0 * C_LIGHT

end # module Constants
