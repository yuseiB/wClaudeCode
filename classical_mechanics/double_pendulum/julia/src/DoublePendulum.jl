"""
Exact nonlinear double pendulum (Lagrangian formulation, RK4 integrator).

Angles are measured from the downward vertical.
State vector: [θ₁, ω₁, θ₂, ω₂].
"""
module DoublePendulumModule

export DoublePendulum, simulate

struct DoublePendulum
    m1::Float64   # bob-1 mass [kg]
    m2::Float64   # bob-2 mass [kg]
    L1::Float64   # rod-1 length [m]
    L2::Float64   # rod-2 length [m]
    g::Float64    # gravitational acceleration [m s⁻²]
end

DoublePendulum(; m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81) =
    DoublePendulum(m1, m2, L1, L2, g)

"""
    eom(dp, state) -> dstate

RHS of the first-order ODE.  state = [θ₁, ω₁, θ₂, ω₂].
"""
function eom(dp::DoublePendulum, s::Vector{Float64})::Vector{Float64}
    θ1, ω1, θ2, ω2 = s
    Δ = θ1 - θ2
    D = 2dp.m1 + dp.m2 - dp.m2 * cos(2Δ)

    α1 = (
        -dp.g * (2dp.m1 + dp.m2) * sin(θ1)
        - dp.m2 * dp.g * sin(θ1 - 2θ2)
        - 2sin(Δ) * dp.m2 * (ω2^2 * dp.L2 + ω1^2 * dp.L1 * cos(Δ))
    ) / (dp.L1 * D)

    α2 = (
        2sin(Δ) * (
            ω1^2 * dp.L1 * (dp.m1 + dp.m2)
            + dp.g * (dp.m1 + dp.m2) * cos(θ1)
            + ω2^2 * dp.L2 * dp.m2 * cos(Δ)
        )
    ) / (dp.L2 * D)

    [ω1, α1, ω2, α2]
end

"""Total mechanical energy E = T + V."""
function energy(dp::DoublePendulum, s::Vector{Float64})::Float64
    θ1, ω1, θ2, ω2 = s
    Δ = θ1 - θ2
    T = 0.5dp.m1 * dp.L1^2 * ω1^2 +
        0.5dp.m2 * (dp.L1^2 * ω1^2 + dp.L2^2 * ω2^2 +
                    2dp.L1 * dp.L2 * ω1 * ω2 * cos(Δ))
    V = -dp.g * ((dp.m1 + dp.m2) * dp.L1 * cos(θ1) + dp.m2 * dp.L2 * cos(θ2))
    T + V
end

"""
    simulate(dp; theta1_0, theta2_0, omega1_0, omega2_0, t_end, dt) -> NamedTuple

Fixed-step RK4 integration. Returns columns: t, theta1, omega1, theta2, omega2, x2, y2, energy.
"""
function simulate(dp::DoublePendulum;
                  theta1_0::Float64, theta2_0::Float64,
                  omega1_0::Float64 = 0.0, omega2_0::Float64 = 0.0,
                  t_end::Float64 = 20.0, dt::Float64 = 0.005)

    n  = floor(Int, t_end / dt) + 1
    ts = Vector{Float64}(undef, n)
    θ1 = Vector{Float64}(undef, n)
    ω1 = Vector{Float64}(undef, n)
    θ2 = Vector{Float64}(undef, n)
    ω2 = Vector{Float64}(undef, n)
    x2 = Vector{Float64}(undef, n)
    y2 = Vector{Float64}(undef, n)
    en = Vector{Float64}(undef, n)

    y = [theta1_0, omega1_0, theta2_0, omega2_0]
    t = 0.0

    # Helper to record the current state
    function record!(i)
        ts[i] = t; θ1[i] = y[1]; ω1[i] = y[2]; θ2[i] = y[3]; ω2[i] = y[4]
        x2[i] =  dp.L1 * sin(y[1]) + dp.L2 * sin(y[3])
        y2[i] = -dp.L1 * cos(y[1]) - dp.L2 * cos(y[3])
        en[i] = energy(dp, y)
    end

    record!(1)
    for i in 2:n
        k1 = eom(dp, y)
        k2 = eom(dp, y .+ (dt/2) .* k1)
        k3 = eom(dp, y .+ (dt/2) .* k2)
        k4 = eom(dp, y .+ dt     .* k3)
        y .+= (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        t  += dt
        record!(i)
    end

    (t=ts, theta1=θ1, omega1=ω1, theta2=θ2, omega2=ω2, x2=x2, y2=y2, energy=en)
end

end # module DoublePendulumModule
