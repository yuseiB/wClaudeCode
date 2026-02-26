"""Unit tests for 2D Ising Model (Julia)."""

# Ensure the package is loadable from this script's location
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using IsingModel2D

using Test

T_C = T_CRITICAL  # ≈ 2.2692

# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
@testset "Energy — all spins up" begin
    n = 8
    m = IsingModel(n; seed=0)
    fill!(m.lattice, Int8(1))
    @test isapprox(energy(m), -2.0 * n^2, atol=1e-10)
end

@testset "Energy — all spins down" begin
    n = 8
    m = IsingModel(n; seed=0)
    fill!(m.lattice, Int8(-1))
    @test isapprox(energy(m), -2.0 * n^2, atol=1e-10)
end

@testset "Energy — checkerboard" begin
    n = 8
    m = IsingModel(n; seed=0)
    for i in 1:n, j in 1:n
        m.lattice[i, j] = Int8((i + j) % 2 == 0 ? 1 : -1)
    end
    @test isapprox(energy(m), +2.0 * n^2, atol=1e-10)
end

# ---------------------------------------------------------------------------
# Magnetisation
# ---------------------------------------------------------------------------
@testset "Magnetisation — all up" begin
    n = 10
    m = IsingModel(n; seed=0)
    fill!(m.lattice, Int8(1))
    @test magnetization(m) == Float64(n^2)
end

@testset "Magnetisation — all down" begin
    n = 10
    m = IsingModel(n; seed=0)
    fill!(m.lattice, Int8(-1))
    @test magnetization(m) == -Float64(n^2)
end

# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
@testset "Low-T ordering" begin
    n = 16
    m = IsingModel(n; seed=2)
    for _ in 1:2000
        metropolis_step!(m, 0.5)
    end
    mag = abs(magnetization(m)) / n^2
    @test mag > 0.9
end

@testset "High-T disordering" begin
    n = 32
    m = IsingModel(n; seed=1)
    fill!(m.lattice, Int8(1))
    init_m = abs(magnetization(m)) / n^2
    for _ in 1:500
        metropolis_step!(m, 100.0)
    end
    final_m = abs(magnetization(m)) / n^2
    @test final_m < init_m
end

# ---------------------------------------------------------------------------
# Phase transition (statistical)
# ---------------------------------------------------------------------------
@testset "simulate — below Tc" begin
    m = IsingModel(24; seed=3)
    fill!(m.lattice, Int8(1))
    r = simulate(m, 1.5; n_therm=3000, n_measure=5000)
    @test r.M_mean > 0.5
end

@testset "simulate — above Tc" begin
    m = IsingModel(24; seed=4)
    r = simulate(m, 3.5; n_therm=3000, n_measure=5000)
    @test r.M_mean < 0.3
end

@testset "Onsager T_c value" begin
    @test isapprox(T_C, 2.2692, atol=1e-4)
end

println("\nAll tests passed ✓")
