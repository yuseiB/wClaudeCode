"""
2D Ising Model — Metropolis-Hastings Monte Carlo

Square N×N lattice with periodic boundary conditions.
Hamiltonian:  H = -J Σ_{<i,j>} s_i s_j
Spins:        s_i ∈ {-1, +1}

Onsager's exact critical temperature (J = k_B = 1):
  T_c = 2 / ln(1 + √2) ≈ 2.2692
"""

using Random
using Statistics

# Onsager's exact critical temperature (J = k_B = 1)
const T_CRITICAL = 2.0 / log(1.0 + sqrt(2.0))

"""
Mutable struct representing a 2D Ising lattice.

Fields
------
- `lattice` : N×N matrix of Int8 (±1 spins)
- `n`       : linear lattice size
- `j`       : coupling constant (J > 0 for ferromagnet)
- `rng`     : reproducible random-number generator
"""
mutable struct IsingModel
    lattice::Matrix{Int8}
    n::Int
    j::Float64
    rng::AbstractRNG
end

"""
    IsingModel(n; j=1.0, seed=42)

Create a new N×N Ising model with random ±1 spin initialisation.
"""
function IsingModel(n::Int; j::Float64=1.0, seed::Int=42)
    rng = MersenneTwister(seed)
    lattice = rand(rng, Int8[-1, 1], n, n)
    IsingModel(lattice, n, j, rng)
end

# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

"""
    energy(model) → Float64

Total energy E = -J Σ_{<i,j>} s_i s_j (nearest neighbours, PBC).
"""
function energy(m::IsingModel)::Float64
    n = m.n
    e = 0
    for i in 1:n, j in 1:n
        s = Int(m.lattice[i, j])
        right = Int(m.lattice[i, mod1(j + 1, n)])
        down  = Int(m.lattice[mod1(i + 1, n), j])
        e += s * (right + down)
    end
    return -m.j * e
end

"""
    magnetization(model) → Float64

Total magnetisation M = Σ_i s_i.
"""
magnetization(m::IsingModel)::Float64 = Float64(sum(m.lattice))

# ---------------------------------------------------------------------------
# Monte Carlo dynamics
# ---------------------------------------------------------------------------

"""
    metropolis_step!(model, T)

One full MC sweep: N² single-spin Metropolis-Hastings attempts.
"""
function metropolis_step!(m::IsingModel, T::Float64)
    n  = m.n
    n2 = n * n
    β  = 1.0 / T

    # Pre-compute acceptance probabilities for ΔE/J ∈ {4, 8}
    exp4 = exp(-β * m.j * 4.0)
    exp8 = exp(-β * m.j * 8.0)

    for _ in 1:n2
        i  = rand(m.rng, 1:n)
        j  = rand(m.rng, 1:n)
        s  = Int(m.lattice[i, j])

        nb = Int(m.lattice[mod1(i - 1, n), j]) +
             Int(m.lattice[mod1(i + 1, n), j]) +
             Int(m.lattice[i, mod1(j - 1, n)]) +
             Int(m.lattice[i, mod1(j + 1, n)])

        dE_over_J = 2 * s * nb

        prob = dE_over_J <= 0 ? 1.0 :
               dE_over_J == 4 ? exp4 :
               dE_over_J == 8 ? exp8 : 0.0

        if rand(m.rng) < prob
            m.lattice[i, j] = Int8(-s)
        end
    end
end

"""
    simulate(model, T; n_therm=5000, n_measure=10000) → NamedTuple

Thermalise then measure observables.

Returns
-------
`(T, E_mean, E2_mean, M_mean, M2_mean, Cv, chi)`
"""
function simulate(m::IsingModel, T::Float64;
                  n_therm::Int=5_000, n_measure::Int=10_000)

    n2 = Float64(m.n^2)

    for _ in 1:n_therm
        metropolis_step!(m, T)
    end

    e_arr = Vector{Float64}(undef, n_measure)
    mag_arr = Vector{Float64}(undef, n_measure)

    for k in 1:n_measure
        metropolis_step!(m, T)
        e_arr[k]   = energy(m) / n2
        mag_arr[k] = abs(magnetization(m)) / n2
    end

    e_mean  = mean(e_arr)
    e2_mean = mean(e_arr .^ 2)
    m_mean  = mean(mag_arr)
    m2_mean = mean(mag_arr .^ 2)

    cv  = (e2_mean  - e_mean^2)  / T^2
    chi = (m2_mean  - m_mean^2)  / T

    return (T=T, E_mean=e_mean, E2_mean=e2_mean,
            M_mean=m_mean, M2_mean=m2_mean, Cv=cv, chi=chi)
end
