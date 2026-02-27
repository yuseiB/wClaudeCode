"""
Waveguide.jl — 直方体・円形 PEC 導波管の伝搬モード解析 (Julia)

物理的な背景:
  PEC 境界条件: 導体壁面で接線成分 E_tan = 0
  伝搬定数: β = √((ω/c)² − kc²) [f > fc: 伝搬, f < fc: エバネッセント]
  分散関係: ω² = (βc)² + (kc·c)² — 双曲線型
  群速度 vg = c²β/ω < c
"""
module Waveguide

using SpecialFunctions: besselj
using ..Constants: C_LIGHT, MU0, EPS0, ETA0
using ..Cavity: TM_ZEROS, TE_ZEROS, djm, EMPoint

# ── 直方体導波管 ─────────────────────────────────────────────────────────────

"""
直方体 PEC 導波管 (0≤x≤a, 0≤y≤b) の TE/TM モード

TE_mn: E_z=0, m,n≥0 (両方 0 は不可)
TM_mn: H_z=0, m,n≥1

カットオフ波数: kc = π√((m/a)² + (n/b)²)
カットオフ周波数: fc = c·kc / (2π)
"""
struct RectangularWaveguideMode
    a::Float64; b::Float64
    m::Int; n::Int
    mode_type::Symbol
    kx::Float64; ky::Float64; kc2::Float64; fc::Float64
end

function RectangularWaveguideMode(a, b, m, n, mode_type::Symbol=:TE)
    if mode_type == :TE
        (m == 0 && n == 0) && error("TE: m and n cannot both be 0")
        (m < 0 || n < 0) && error("m, n must be ≥ 0")
    elseif mode_type == :TM
        (m < 1 || n < 1) && error("TM: m and n must be ≥ 1")
    else
        error("mode_type must be :TE or :TM, got $mode_type")
    end
    kx  = m * π / a
    ky  = n * π / b
    kc2 = kx^2 + ky^2
    fc  = C_LIGHT * sqrt(kc2) / (2π)
    RectangularWaveguideMode(a, b, m, n, mode_type, kx, ky, kc2, fc)
end

"カットオフ周波数 (Hz)"
cutoff_frequency(mo::RectangularWaveguideMode) = mo.fc

"モードラベル (例: \"TE_10\")"
mode_label(mo::RectangularWaveguideMode) = "$(mo.mode_type == :TE ? "TE" : "TM")_$(mo.m)$(mo.n)"

"伝搬定数 β (rad/m)。カットオフ以下なら 0.0 を返す。"
function propagation_constant(mo::RectangularWaveguideMode, frequency)
    k  = 2π * frequency / C_LIGHT
    d  = k^2 - mo.kc2
    d >= 0 ? sqrt(d) : 0.0
end

"波動インピーダンス (Ω)"
function wave_impedance(mo::RectangularWaveguideMode, frequency)
    k    = 2π * frequency / C_LIGHT
    beta = propagation_constant(mo, frequency)
    beta == 0.0 && return Inf
    mo.mode_type == :TE ? ETA0 * k / beta : ETA0 * beta / k
end

"""横断面上の EM 場を計算する (軸位置 z, 位相 ωt)。"""
function fields(mo::RectangularWaveguideMode, x, y, z, frequency, phase=0.0)
    beta  = propagation_constant(mo, frequency)
    omega = 2π * frequency
    psi   = cos(phase - beta * z)

    cx, sx = cos(mo.kx * x), sin(mo.kx * x)
    cy, sy = cos(mo.ky * y), sin(mo.ky * y)
    kc2 = mo.kc2

    if kc2 < 1e-30
        return EMPoint()
    end

    if mo.mode_type == :TE
        hz =  cx * cy * psi
        hx =  (beta * mo.kx / kc2) * sx * cy * psi
        hy =  (beta * mo.ky / kc2) * cx * sy * psi
        ex = -(omega * MU0 * mo.ky / kc2) * cx * sy * psi
        ey =  (omega * MU0 * mo.kx / kc2) * sx * cy * psi
        return EMPoint(ex, ey, 0.0, hx, hy, hz)
    else  # TM
        ez =  sx * sy * psi
        ex = -(beta * mo.kx / kc2) * cx * sy * psi
        ey = -(beta * mo.ky / kc2) * sx * cy * psi
        hx =  (omega * EPS0 * mo.ky / kc2) * sx * cy * psi
        hy = -(omega * EPS0 * mo.kx / kc2) * cx * sy * psi
        return EMPoint(ex, ey, ez, hx, hy, 0.0)
    end
end

"最低次 n_modes モードをカットオフ周波数昇順で返す。"
function rectangular_waveguide_modes(a, b; n_modes=8)
    modes = RectangularWaveguideMode[]
    for m in 0:5, n in 0:5, t in (:TE, :TM)
        try
            push!(modes, RectangularWaveguideMode(a, b, m, n, t))
        catch; end
    end
    sort!(modes; by=cutoff_frequency)
    unique!(mo -> (mo.mode_type, mo.m, mo.n), modes)
    modes[1:min(n_modes, length(modes))]
end

# ── 円形導波管 ───────────────────────────────────────────────────────────────

"""
円形 PEC 導波管 (0≤ρ≤R) の TE/TM モード

TM_mn: kc = χ_mn/R (J_m の零点)
TE_mn: kc = χ'_mn/R (J_m' の零点)
"""
struct CircularWaveguideMode
    r::Float64
    m::Int; n::Int
    mode_type::Symbol
    chi::Float64; kc::Float64; kc2::Float64; fc::Float64
end

function CircularWaveguideMode(r, m, n, mode_type::Symbol=:TE)
    n < 1 && error("n must be ≥ 1")
    (m > 4 || n > 3) && error("m/n out of Bessel table range")
    chi = mode_type == :TM ? TM_ZEROS[m+1, n] : TE_ZEROS[m+1, n]
    kc  = chi / r
    kc2 = kc^2
    fc  = C_LIGHT * kc / (2π)
    CircularWaveguideMode(r, m, n, mode_type, chi, kc, kc2, fc)
end

cutoff_frequency(mo::CircularWaveguideMode) = mo.fc
mode_label(mo::CircularWaveguideMode) = "$(mo.mode_type == :TE ? "TE" : "TM")_$(mo.m)$(mo.n)"

function propagation_constant(mo::CircularWaveguideMode, frequency)
    k = 2π * frequency / C_LIGHT
    d = k^2 - mo.kc2
    d >= 0 ? sqrt(d) : 0.0
end

"""極座標断面上の EM 場を計算する。"""
function fields_polar(mo::CircularWaveguideMode, rho, phi, z, frequency, phase=0.0)
    beta  = propagation_constant(mo, frequency)
    omega = 2π * frequency
    psi   = cos(phase - beta * z)

    kc = mo.kc
    kc2 = mo.kc2
    kc_rho = kc * rho
    jm_val  = besselj(mo.m, kc_rho)
    djm_val = djm(mo.m, kc_rho) * kc
    cos_mphi = cos(mo.m * phi)
    sin_mphi = sin(mo.m * phi)
    rho_s = max(rho, 1e-30)

    if mo.mode_type == :TM
        ez  = jm_val * cos_mphi * psi
        er  = -(beta * kc / kc2) * djm_val * cos_mphi * psi
        ep  =  (beta * mo.m / (kc2 * rho_s)) * jm_val * sin_mphi * psi
        hr  = -(omega * EPS0 * mo.m / (kc2 * rho_s)) * jm_val * sin_mphi * psi
        hm  =  (omega * EPS0 * kc / kc2) * djm_val * cos_mphi * psi
        return EMPoint(er, ep, ez, hr, hm, 0.0)
    else  # TE
        hz  = jm_val * cos_mphi * psi
        hr  = -(beta * kc / kc2) * djm_val * cos_mphi * psi
        hm  =  (beta * mo.m / (kc2 * rho_s)) * jm_val * sin_mphi * psi
        er  =  (omega * MU0 * mo.m / (kc2 * rho_s)) * jm_val * sin_mphi * psi
        ep  = -(omega * MU0 * kc / kc2) * djm_val * cos_mphi * psi
        return EMPoint(er, ep, 0.0, hr, hm, hz)
    end
end

"最低次 n_modes モードをカットオフ周波数昇順で返す。"
function circular_waveguide_modes(r; n_modes=8)
    modes = CircularWaveguideMode[]
    for m in 0:4, n in 1:3, t in (:TE, :TM)
        try
            push!(modes, CircularWaveguideMode(r, m, n, t))
        catch; end
    end
    sort!(modes; by=cutoff_frequency)
    unique!(mo -> (mo.mode_type, mo.m, mo.n), modes)
    modes[1:min(n_modes, length(modes))]
end

end # module Waveguide
