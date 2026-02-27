"""
Cavity.jl — 電磁気共振キャビティの解析解 (Julia)

対応キャビティ:
  RectangularCavityMode — 直方体 PEC キャビティ a×b×d
  CylindricalCavityMode — 円筒型 PEC キャビティ (半径 R, 高さ L)

物理的な背景:
  PEC 境界条件: 導体壁面で接線成分 E_tan = 0
  時間依存性: E(r,t) = E₀(r)cos(ωt),  H(r,t) = H₀(r)sin(ωt)
  電場と磁場が 90° 位相ずれ → エネルギーが E↔H 間で振動
"""
module Cavity

using SpecialFunctions: besselj
using ..Constants: C_LIGHT, MU0, EPS0, ETA0

# ── Bessel 零点テーブル (Pozar Table 3.4, 3.5) ──────────────────────────────

# J_m(χ_mn) = 0 の零点  (TM モード用)
const TM_ZEROS = [
    2.4048  5.5201  8.6537;   # m=0
    3.8317  7.0156  10.1735;  # m=1
    5.1356  8.4172  11.6198;  # m=2
    6.3802  9.7610  13.0152;  # m=3
    7.5883  11.0647 14.3725;  # m=4
]

# J_m'(χ'_mn) = 0 の零点  (TE モード用)
const TE_ZEROS = [
    3.8317  7.0156  10.1735;  # m=0
    1.8412  5.3314  8.5363;   # m=1  (最低次 χ'₁₁)
    3.0542  6.7061  9.9695;   # m=2
    4.2012  8.0152  11.3459;  # m=3
    5.3175  9.2824  12.6819;  # m=4
]

# ── Bessel 導関数 ────────────────────────────────────────────────────────────

"J_m'(x) = (J_{m-1}(x) − J_{m+1}(x)) / 2"
djm(m, x) = 0.5 * (besselj(m - 1, x) - besselj(m + 1, x))

# ── EMPoint 構造体 ───────────────────────────────────────────────────────────

"電磁場 6 成分 (Ex,Ey,Ez,Hx,Hy,Hz) または 曲線座標 (Eρ,Eφ,Ez,...)"
struct EMPoint
    ex::Float64; ey::Float64; ez::Float64
    hx::Float64; hy::Float64; hz::Float64
end

EMPoint() = EMPoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

e_mag(f::EMPoint) = sqrt(f.ex^2 + f.ey^2 + f.ez^2)
h_mag(f::EMPoint) = sqrt(f.hx^2 + f.hy^2 + f.hz^2)

# ── 直方体キャビティ ─────────────────────────────────────────────────────────

"""
直方体 PEC キャビティ (0≤x≤a, 0≤y≤b, 0≤z≤d) の TE/TM モード

TE_mnp: E_z=0, m,n≥0 (両方 0 は不可), p≥1
TM_mnp: H_z=0, m,n≥1, p≥0

共振周波数: f = (c/2) √((m/a)² + (n/b)² + (p/d)²)
"""
struct RectangularCavityMode
    a::Float64; b::Float64; d::Float64
    m::Int; n::Int; p::Int
    mode_type::Symbol          # :TE or :TM
    # 事前計算値
    kx::Float64; ky::Float64; kz::Float64
    kc2::Float64               # kx²+ky²
    omega::Float64
end

function RectangularCavityMode(a, b, d, m, n, p, mode_type::Symbol=:TE)
    # バリデーション
    if mode_type == :TE
        (m == 0 && n == 0) && error("TE: m and n cannot both be 0")
        p < 1 && error("TE: p must be ≥ 1")
    elseif mode_type == :TM
        (m < 1 || n < 1) && error("TM: m and n must be ≥ 1")
        p < 0 && error("TM: p must be ≥ 0")
    else
        error("mode_type must be :TE or :TM, got $mode_type")
    end

    kx = m * π / a
    ky = n * π / b
    kz = p * π / d
    k  = sqrt(kx^2 + ky^2 + kz^2)
    kc2 = kx^2 + ky^2
    omega = C_LIGHT * k
    RectangularCavityMode(a, b, d, m, n, p, mode_type, kx, ky, kz, kc2, omega)
end

"共振周波数 (Hz)"
resonant_frequency(mo::RectangularCavityMode) = mo.omega / (2π)

"モードラベル (例: \"TE_101\")"
mode_label(mo::RectangularCavityMode) = "$(mo.mode_type == :TE ? "TE" : "TM")_$(mo.m)$(mo.n)$(mo.p)"

"""
位置 (x, y, z) での EM 場を計算する。

時間依存性:
  E(r,t) = E₀(r)·cos(ωt + phase)   (phase = ωt で評価)
  H(r,t) = H₀(r)·sin(ωt + phase)
"""
function fields(mo::RectangularCavityMode, x, y, z, phase=0.0)
    ct, st = cos(phase), sin(phase)
    cx, sx = cos(mo.kx * x), sin(mo.kx * x)
    cy, sy = cos(mo.ky * y), sin(mo.ky * y)
    cz, sz = cos(mo.kz * z), sin(mo.kz * z)

    if mo.mode_type == :TE
        hz = cx * cy * sz * st
        if mo.kc2 > 0
            g2 = mo.kc2
            ex =  (mo.omega * MU0 * mo.ky / g2) * cx * sy * sz * ct
            ey = -(mo.omega * MU0 * mo.kx / g2) * sx * cy * sz * ct
            hx = -(mo.kx * mo.kz / g2) * sx * cy * cz * st
            hy = -(mo.ky * mo.kz / g2) * cx * sy * cz * st
        else
            ex = ey = hx = hy = 0.0
        end
        return EMPoint(ex, ey, 0.0, hx, hy, hz)
    else  # TM
        ez = sx * sy * cz * ct
        if mo.kc2 > 0
            g2 = mo.kc2
            ex =  (mo.kx * mo.kz / g2) * cx * sy * sz * ct
            ey =  (mo.ky * mo.kz / g2) * sx * cy * sz * ct
            hx =  (mo.omega * EPS0 * mo.ky / g2) * sx * cy * cz * st
            hy = -(mo.omega * EPS0 * mo.kx / g2) * cx * sy * cz * st
        else
            ex = ey = hx = hy = 0.0
        end
        return EMPoint(ex, ey, ez, hx, hy, 0.0)
    end
end

"最低次 n_modes モードを周波数昇順で返す。"
function rectangular_cavity_modes(a, b, d; n_modes=10)
    modes = RectangularCavityMode[]
    for m in 0:5, n in 0:5, p in 0:5, t in (:TE, :TM)
        try
            push!(modes, RectangularCavityMode(a, b, d, m, n, p, t))
        catch; end
    end
    sort!(modes; by=resonant_frequency)
    unique!(mo -> (mo.mode_type, mo.m, mo.n, mo.p), modes)
    modes[1:min(n_modes, length(modes))]
end

# ── 円筒型キャビティ ─────────────────────────────────────────────────────────

"""
円筒型 PEC キャビティ (0≤ρ≤R, 0≤z≤L) の TM/TE モード

TM_mnp: kc = χ_mn/R (J_m の零点), p≥0
TE_mnp: kc = χ'_mn/R (J_m' の零点), p≥1

共振周波数: f = (c/2π) √((χ/R)² + (pπ/L)²)
"""
struct CylindricalCavityMode
    r::Float64; l::Float64
    m::Int; n::Int; p::Int
    mode_type::Symbol
    chi::Float64
    kc::Float64; kz::Float64; omega::Float64
end

function CylindricalCavityMode(r, l, m, n, p, mode_type::Symbol=:TM)
    n < 1 && error("n must be ≥ 1")
    m > 4 || n > 3 || true  # silent range check
    (m > 4 || n > 3) && error("m/n out of Bessel table range")
    mode_type == :TE && p < 1 && error("TE: p must be ≥ 1")

    chi = mode_type == :TM ? TM_ZEROS[m+1, n] : TE_ZEROS[m+1, n]
    kc  = chi / r
    kz  = p * π / l
    omega = C_LIGHT * sqrt(kc^2 + kz^2)
    CylindricalCavityMode(r, l, m, n, p, mode_type, chi, kc, kz, omega)
end

resonant_frequency(mo::CylindricalCavityMode) = mo.omega / (2π)
mode_label(mo::CylindricalCavityMode) = "$(mo.mode_type == :TE ? "TE" : "TM")_$(mo.m)$(mo.n)$(mo.p)"

"""ρ-z 断面上の EM 場を計算する。"""
function fields_rz(mo::CylindricalCavityMode, rho, z; phi=0.0, phase=0.0)
    ct, st = cos(phase), sin(phase)
    kc = mo.kc
    kc_rho = kc * rho
    jm_val  = besselj(mo.m, kc_rho)
    djm_val = djm(mo.m, kc_rho) * kc
    cos_mphi = cos(mo.m * phi)
    sin_mphi = sin(mo.m * phi)
    cz, sz = cos(mo.kz * z), sin(mo.kz * z)
    rho_s = max(rho, 1e-30)
    kc2 = kc^2

    if mo.mode_type == :TM
        ez = jm_val * cos_mphi * cz * ct
        if kc > 0
            er = -(mo.kz / kc) * djm_val * cos_mphi * sz * ct
            ep =  (mo.kz * mo.m / (kc2 * rho_s)) * jm_val * sin_mphi * sz * ct
            hr =  (mo.omega * EPS0 * mo.m / (kc2 * rho_s)) * jm_val * sin_mphi * cz * st
            hm = -(mo.omega * EPS0 / kc) * djm_val * cos_mphi * cz * st
        else
            er = ep = hr = hm = 0.0
        end
        return EMPoint(er, ep, ez, hr, hm, 0.0)
    else  # TE
        hz = jm_val * cos_mphi * sz * st
        if kc > 0
            hr = -(mo.kz / kc) * djm_val * cos_mphi * cz * st
            hm =  (mo.kz * mo.m / (kc2 * rho_s)) * jm_val * sin_mphi * cz * st
            er = -(mo.omega * MU0 * mo.m / (kc2 * rho_s)) * jm_val * sin_mphi * sz * ct
            ep = -(mo.omega * MU0 / kc) * djm_val * cos_mphi * sz * ct
        else
            hr = hm = er = ep = 0.0
        end
        return EMPoint(er, ep, 0.0, hr, hm, hz)
    end
end

"最低次 n_modes モードを周波数昇順で返す。"
function cylindrical_cavity_modes(r, l; n_modes=10)
    modes = CylindricalCavityMode[]
    for m in 0:4, n in 1:3, p in 0:3, t in (:TM, :TE)
        try
            push!(modes, CylindricalCavityMode(r, l, m, n, p, t))
        catch; end
    end
    sort!(modes; by=resonant_frequency)
    unique!(mo -> (mo.mode_type, mo.m, mo.n, mo.p), modes)
    modes[1:min(n_modes, length(modes))]
end

end # module Cavity
