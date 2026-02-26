"""
Run eight double-pendulum scenarios and write CSV output.

Run from julia/examples/:
    julia --project=.. dp_sim.jl
"""

include(joinpath(@__DIR__, "..", "src", "DoublePendulum.jl"))
using .DoublePendulumModule

using Printf

const PI = π

function write_csv(path::String, r)
    open(path, "w") do io
        println(io, "t,theta1,omega1,theta2,omega2,x2,y2,energy")
        for i in eachindex(r.t)
            @printf(io, "%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                r.t[i], r.theta1[i], r.omega1[i],
                r.theta2[i], r.omega2[i],
                r.x2[i], r.y2[i], r.energy[i])
        end
    end
end

dp = DoublePendulum()   # m1=m2=L1=L2=1, g=9.81

# ── section 1: three dynamical regimes ───────────────────────────────────────
write_csv("case_nearlinear.csv",
    simulate(dp, theta1_0=10PI/180, theta2_0=10PI/180, t_end=30.0, dt=0.005))
write_csv("case_intermediate.csv",
    simulate(dp, theta1_0=90PI/180, theta2_0=0.0, t_end=30.0, dt=0.005))
write_csv("case_chaotic.csv",
    simulate(dp, theta1_0=120PI/180, theta2_0=-30PI/180, t_end=30.0, dt=0.005))

# ── section 2: sensitivity to initial conditions ──────────────────────────────
delta = 0.001 * PI / 180
write_csv("sensitivity_a.csv",
    simulate(dp, theta1_0=120PI/180, theta2_0=-30PI/180, t_end=25.0, dt=0.005))
write_csv("sensitivity_b.csv",
    simulate(dp, theta1_0=120PI/180+delta, theta2_0=-30PI/180, t_end=25.0, dt=0.005))

# ── section 3: mass ratios ────────────────────────────────────────────────────
for (ratio, fname) in [
    (0.25, "mass_ratio_0.25.csv"),
    (1.00, "mass_ratio_1.00.csv"),
    (4.00, "mass_ratio_4.00.csv"),
]
    dp_m = DoublePendulum(1.0, ratio, 1.0, 1.0, 9.81)
    write_csv(fname, simulate(dp_m, theta1_0=90PI/180, theta2_0=0.0, t_end=30.0, dt=0.005))
end

println("Wrote 8 CSV files.")
