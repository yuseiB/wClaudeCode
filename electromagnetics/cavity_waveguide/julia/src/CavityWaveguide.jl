"""
CavityWaveguide — 電磁気キャビティ・導波管解析ライブラリ (Julia)

電磁気共振キャビティと導波管の解析解を提供するJuliaモジュール。

モジュール構成:
  Constants   — 物理定数 (SI 単位)
  Cavity      — 直方体・円筒型・球形キャビティ共振モード
  Waveguide   — 直方体・円形導波管伝搬モード

使用例:
```julia
using CavityWaveguide
# 直方体キャビティ TE₁₀₁ モード
mode = RectangularCavityMode(0.04, 0.02, 0.03, 1, 0, 1, :TE)
println(resonant_frequency(mode) / 1e9, " GHz")

# 直方体導波管 TE₁₀ モード
wg = RectangularWaveguideMode(0.04, 0.02, 1, 0, :TE)
println(cutoff_frequency(wg) / 1e9, " GHz")
```
"""
module CavityWaveguide

include("Constants.jl")
include("Cavity.jl")
include("Waveguide.jl")

using .Constants
using .Cavity
using .Waveguide

# 公開インターフェース
export
    # 定数
    C_LIGHT, MU0, EPS0, ETA0,
    # 構造体
    EMPoint,
    RectangularCavityMode, CylindricalCavityMode,
    RectangularWaveguideMode, CircularWaveguideMode,
    # 関数
    resonant_frequency, cutoff_frequency, propagation_constant,
    wave_impedance, fields, fields_rz, fields_polar, mode_label,
    rectangular_cavity_modes, cylindrical_cavity_modes,
    rectangular_waveguide_modes, circular_waveguide_modes

end # module CavityWaveguide
