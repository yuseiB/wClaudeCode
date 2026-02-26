"""
2D Ising Model — Temperature Sweep (Julia)

Generates ising2d_sweep.csv with columns: T, E_mean, M_mean, Cv, chi

Usage:
  julia --project=statistical_physics/ising_model_2d/julia \\
        statistical_physics/ising_model_2d/julia/examples/ising_sim.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using IsingModel2D

const N       = 32
const N_THERM = 5_000
const N_MEAS  = 10_000
const T_MIN   = 1.0
const T_MAX   = 4.0
const N_T     = 40

println("2D Ising Model sweep  N=$N  T=[$T_MIN, $T_MAX]  $N_T points")
println("Onsager T_c = $(round(T_CRITICAL, digits=4))")

temps = range(T_MIN, T_MAX, length=N_T)

out_file = joinpath(@__DIR__, "ising2d_sweep.csv")
open(out_file, "w") do io
    println(io, "T,E_mean,M_mean,Cv,chi")
    for (k, T) in enumerate(temps)
        m = IsingModel(N; seed=k)
        if T < T_CRITICAL
            fill!(m.lattice, Int8(1))   # start ordered below Tc
        end
        r = simulate(m, Float64(T); n_therm=N_THERM, n_measure=N_MEAS)
        println(io, "$(round(T, digits=4)),$(round(r.E_mean, digits=6))," *
                    "$(round(r.M_mean, digits=6)),$(round(r.Cv, digits=6))," *
                    "$(round(r.chi, digits=6))")
        if k % 10 == 0
            println("  $k/$N_T  T=$(round(T, digits=2))  |M|=$(round(r.M_mean, digits=3))  Cv=$(round(r.Cv, digits=3))")
        end
    end
end
println("\nSaved → $out_file")
