using Test
include("../src/MathPhys.jl")
using .MathPhys

@testset "integrate_trapezoid" begin
    x = range(0, 1; length=1000)
    @test abs(integrate_trapezoid(ones(1000), collect(x)) - 1.0) < 1e-9
    @test abs(integrate_trapezoid(collect(x).^2, collect(x)) - 1/3) < 1e-5
end
