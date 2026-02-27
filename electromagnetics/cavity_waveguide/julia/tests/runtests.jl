"""
runtests.jl — 電磁気キャビティ・導波管ライブラリのテスト

使用法:
    julia --project=.. tests/runtests.jl
"""

using Test
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using CavityWaveguide
using CavityWaveguide: C_LIGHT

@testset "CavityWaveguide Tests" begin

    # ── 直方体キャビティ ──────────────────────────────────────────────────────

    @testset "RectangularCavity" begin
        a, b, d = 0.04, 0.02, 0.03

        @testset "TE_101 周波数" begin
            mo = RectangularCavityMode(a, b, d, 1, 0, 1, :TE)
            expected = C_LIGHT / 2 * sqrt((1/a)^2 + (1/d)^2)
            @test abs(resonant_frequency(mo) - expected) < 1e6
        end

        @testset "TM_110 周波数" begin
            mo = RectangularCavityMode(a, b, d, 1, 1, 0, :TM)
            expected = C_LIGHT / 2 * sqrt((1/a)^2 + (1/b)^2)
            @test abs(resonant_frequency(mo) - expected) < 1e6
        end

        @testset "TE m=n=0 エラー" begin
            @test_throws ErrorException RectangularCavityMode(a, b, d, 0, 0, 1, :TE)
        end

        @testset "TM m=0 エラー" begin
            @test_throws ErrorException RectangularCavityMode(a, b, d, 0, 1, 1, :TM)
        end

        @testset "TE_101 位相0でE≠0" begin
            mo = RectangularCavityMode(a, b, d, 1, 0, 1, :TE)
            f  = fields(mo, a/2, b/2, d/2, 0.0)
            @test e_mag(f) > 0
        end

        @testset "TE_101 位相0でH=0" begin
            mo = RectangularCavityMode(a, b, d, 1, 0, 1, :TE)
            f  = fields(mo, a/2, b/2, d/2, 0.0)
            @test h_mag(f) < 1e-20
        end

        @testset "TE: Ez=0" begin
            mo = RectangularCavityMode(a, b, d, 1, 0, 1, :TE)
            f  = fields(mo, a/3, b/3, d/3, 0.5)
            @test f.ez == 0.0
        end

        @testset "TM: Hz=0" begin
            mo = RectangularCavityMode(a, b, d, 1, 1, 1, :TM)
            f  = fields(mo, a/3, b/3, d/3, 0.5)
            @test f.hz == 0.0
        end

        @testset "モードラベル" begin
            mo = RectangularCavityMode(a, b, d, 1, 0, 1, :TE)
            @test mode_label(mo) == "TE_101"
        end

        @testset "モードリスト昇順" begin
            modes = rectangular_cavity_modes(a, b, d; n_modes=6)
            for i in 2:length(modes)
                @test resonant_frequency(modes[i]) >= resonant_frequency(modes[i-1])
            end
        end
    end

    # ── 円筒型キャビティ ──────────────────────────────────────────────────────

    @testset "CylindricalCavity" begin
        r, l = 0.015, 0.03

        @testset "TM_010 周波数" begin
            chi01 = 2.4048
            expected = C_LIGHT * chi01 / (2π * r)
            mo = CylindricalCavityMode(r, l, 0, 1, 0, :TM)
            @test abs(resonant_frequency(mo) - expected) / expected < 0.001
        end

        @testset "TM: Hz=0" begin
            mo = CylindricalCavityMode(r, l, 0, 1, 0, :TM)
            f  = fields_rz(mo, 0.005, 0.01; phase=0.5)
            @test f.hz == 0.0
        end

        @testset "TE: Ez=0" begin
            mo = CylindricalCavityMode(r, l, 1, 1, 1, :TE)
            f  = fields_rz(mo, 0.005, 0.01; phase=0.5)
            @test f.ez == 0.0
        end

        @testset "n=0 エラー" begin
            @test_throws ErrorException CylindricalCavityMode(r, l, 0, 0, 0, :TM)
        end

        @testset "モードリスト昇順" begin
            modes = cylindrical_cavity_modes(r, l; n_modes=6)
            for i in 2:length(modes)
                @test resonant_frequency(modes[i]) >= resonant_frequency(modes[i-1])
            end
        end
    end

    # ── 直方体導波管 ──────────────────────────────────────────────────────────

    @testset "RectangularWaveguide" begin
        a, b = 0.04, 0.02

        @testset "TE_10 カットオフ周波数" begin
            mo = RectangularWaveguideMode(a, b, 1, 0, :TE)
            expected = C_LIGHT / (2a)
            @test abs(cutoff_frequency(mo) - expected) < 1e6
        end

        @testset "TE_20 = 2 × TE_10" begin
            m10 = RectangularWaveguideMode(a, b, 1, 0, :TE)
            m20 = RectangularWaveguideMode(a, b, 2, 0, :TE)
            @test abs(cutoff_frequency(m20) / cutoff_frequency(m10) - 2.0) < 1e-10
        end

        @testset "fc より高い周波数で β > 0" begin
            mo = RectangularWaveguideMode(a, b, 1, 0, :TE)
            @test propagation_constant(mo, mo.fc * 1.5) > 0
        end

        @testset "fc より低い周波数で β = 0" begin
            mo = RectangularWaveguideMode(a, b, 1, 0, :TE)
            @test propagation_constant(mo, mo.fc * 0.5) == 0.0
        end

        @testset "TE m=n=0 エラー" begin
            @test_throws ErrorException RectangularWaveguideMode(a, b, 0, 0, :TE)
        end

        @testset "TM m=0 エラー" begin
            @test_throws ErrorException RectangularWaveguideMode(a, b, 0, 1, :TM)
        end

        @testset "モードラベル" begin
            mo = RectangularWaveguideMode(a, b, 1, 0, :TE)
            @test mode_label(mo) == "TE_10"
        end

        @testset "モードリスト昇順" begin
            modes = rectangular_waveguide_modes(a, b; n_modes=6)
            for i in 2:length(modes)
                @test cutoff_frequency(modes[i]) >= cutoff_frequency(modes[i-1])
            end
        end
    end

    # ── 円形導波管 ────────────────────────────────────────────────────────────

    @testset "CircularWaveguide" begin
        r = 0.015

        @testset "TE_11 が最低次モード" begin
            te11 = CircularWaveguideMode(r, 1, 1, :TE)
            tm01 = CircularWaveguideMode(r, 0, 1, :TM)
            @test cutoff_frequency(te11) < cutoff_frequency(tm01)
        end

        @testset "TE_11 カットオフ周波数" begin
            chi11p = 1.8412
            expected = C_LIGHT * chi11p / (2π * r)
            mo = CircularWaveguideMode(r, 1, 1, :TE)
            @test abs(cutoff_frequency(mo) - expected) / expected < 0.001
        end

        @testset "TM_01 カットオフ周波数" begin
            chi01 = 2.4048
            expected = C_LIGHT * chi01 / (2π * r)
            mo = CircularWaveguideMode(r, 0, 1, :TM)
            @test abs(cutoff_frequency(mo) - expected) / expected < 0.001
        end

        @testset "n=0 エラー" begin
            @test_throws ErrorException CircularWaveguideMode(r, 0, 0, :TM)
        end

        @testset "TE: Ez=0" begin
            mo = CircularWaveguideMode(r, 1, 1, :TE)
            f  = fields_polar(mo, 0.005, 0.0, 0.0, mo.fc * 1.5)
            @test f.ez == 0.0
        end

        @testset "TM: Hz=0" begin
            mo = CircularWaveguideMode(r, 0, 1, :TM)
            f  = fields_polar(mo, 0.005, 0.0, 0.0, mo.fc * 1.5)
            @test f.hz == 0.0
        end

        @testset "モードリスト昇順" begin
            modes = circular_waveguide_modes(r; n_modes=6)
            for i in 2:length(modes)
                @test cutoff_frequency(modes[i]) >= cutoff_frequency(modes[i-1])
            end
        end
    end

end # testset
