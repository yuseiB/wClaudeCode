"""
em_sim.jl — 電磁気キャビティ・導波管シミュレーション (Julia)

使用法:
    julia --project=.. examples/em_sim.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using CavityWaveguide
using CavityWaveguide: C_LIGHT

function main()
    # ── 1. 直方体キャビティ ──────────────────────────────────────────────────
    println("=== Rectangular Cavity Resonant Modes ===")
    println("  Dimensions: a=4 cm, b=2 cm, d=3 cm")
    @printf "%-12s %15s\n" "Mode" "f_res (GHz)"
    println("-" ^ 30)

    a, b, d = 0.04, 0.02, 0.03
    for mo in rectangular_cavity_modes(a, b, d; n_modes=8)
        @printf "%-12s %15.4f\n" mode_label(mo) (resonant_frequency(mo) * 1e-9)
    end

    # ── 2. 円筒型キャビティ ──────────────────────────────────────────────────
    println("\n=== Cylindrical Cavity Resonant Modes ===")
    println("  Dimensions: R=1.5 cm, L=3 cm")
    @printf "%-12s %15s\n" "Mode" "f_res (GHz)"
    println("-" ^ 30)

    r_cyl, l_cyl = 0.015, 0.03
    for mo in cylindrical_cavity_modes(r_cyl, l_cyl; n_modes=8)
        @printf "%-12s %15.4f\n" mode_label(mo) (resonant_frequency(mo) * 1e-9)
    end

    # ── 3. 直方体導波管 ──────────────────────────────────────────────────────
    println("\n=== Rectangular Waveguide Modes ===")
    println("  Dimensions: a=4 cm, b=2 cm")
    @printf "%-10s %14s %14s %14s\n" "Mode" "fc (GHz)" "f_op (GHz)" "β (rad/m)"
    println("-" ^ 55)

    for mo in rectangular_waveguide_modes(a, b; n_modes=6)
        fc  = cutoff_frequency(mo)
        fop = fc * 1.5
        beta = propagation_constant(mo, fop)
        @printf "%-10s %14.4f %14.4f %14.4f\n" mode_label(mo) (fc*1e-9) (fop*1e-9) beta
    end

    # ── 4. 円形導波管 ────────────────────────────────────────────────────────
    println("\n=== Circular Waveguide Modes ===")
    println("  Radius: R=1.5 cm")
    @printf "%-10s %14s %14s %14s\n" "Mode" "fc (GHz)" "f_op (GHz)" "β (rad/m)"
    println("-" ^ 55)

    r_wg = 0.015
    for mo in circular_waveguide_modes(r_wg; n_modes=6)
        fc  = cutoff_frequency(mo)
        fop = fc * 1.5
        beta = propagation_constant(mo, fop)
        @printf "%-10s %14.4f %14.4f %14.4f\n" mode_label(mo) (fc*1e-9) (fop*1e-9) beta
    end

    # ── 5. TE_10 分散関係 ────────────────────────────────────────────────────
    println("\n=== TE_10 Dispersion: β vs f (a=4 cm) ===")
    @printf "%10s %14s\n" "f (GHz)" "β (rad/m)"
    println("-" ^ 27)

    te10 = RectangularWaveguideMode(a, b, 1, 0, :TE)
    fc10 = cutoff_frequency(te10)
    for i in 0:10
        f = fc10 * (0.5 + 1.5 * i / 10)
        beta = propagation_constant(te10, f)
        @printf "%10.3f %14.3f\n" (f*1e-9) beta
    end
end

using Printf
main()
