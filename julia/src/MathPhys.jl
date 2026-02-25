module MathPhys

using LinearAlgebra
using Statistics

"""
    integrate_trapezoid(f, x)

Trapezoidal-rule integration of `f` over `x`.
"""
function integrate_trapezoid(f::AbstractVector{<:Real}, x::AbstractVector{<:Real})::Float64
    @assert length(f) == length(x) && length(f) >= 2
    sum(0.5 * (f[i] + f[i-1]) * (x[i] - x[i-1]) for i in 2:length(f))
end

end  # module MathPhys
