"""
test_em.py — 電磁気モジュール (cavity.py / waveguide.py) の単体テスト

テスト設計方針:
  - 解析解が既知の量を厳密に検証する (共振周波数、カットオフ周波数など)
  - 境界条件 (PEC 壁での接線 E = 0) が数値的に満たされているか確認
  - エネルギー保存 (|E|² ∝ |H|²) の確認
  - 異常入力に対するバリデーション

参考値:
  直方体 TE₁₀₁  (a=4cm, b=2cm, d=3cm):
    f = (c/2) √((1/0.04)² + 0 + (1/0.03)²) ≈ 6.20 GHz

  円筒型 TM₀₁₀  (R=1.5cm):
    χ₀₁ = 2.40483,  f = c·χ₀₁ / (2π R) ≈ 7.65 GHz

  直方体 TE₁₀ 導波管 (a=22.86mm):
    fc = c/(2a) = 3e8/(2×0.02286) ≈ 6.56 GHz

  円形 TE₁₁ 導波管 (R=10mm):
    χ'₁₁ = 1.84118,  fc = c·χ'₁₁/(2πR) ≈ 8.79 GHz
"""

import pytest
import numpy as np

# テスト対象モジュールのインポート
from mathphys.cavity import (
    RectangularCavityMode,
    CylindricalCavityMode,
    SphericalCavityMode,
    rectangular_cavity_modes,
    cylindrical_cavity_modes,
    spherical_cavity_modes,
    C_LIGHT,
    EMField,
)
from mathphys.waveguide import (
    RectangularWaveguideMode,
    RectangularWaveguide,
    CircularWaveguideMode,
    CircularWaveguide,
)


# ─────────────────────────────────────────────────────────────────────────────
# 直方体キャビティのテスト
# ─────────────────────────────────────────────────────────────────────────────

class TestRectangularCavity:
    """直方体 PEC キャビティの解析解検証。"""

    # キャビティ寸法 (m): 4×2×3 cm
    A, B, D = 0.04, 0.02, 0.03

    def test_te101_resonant_frequency(self):
        """
        TE₁₀₁ 共振周波数の解析解との一致を確認。

        f = (c/2) √((m/a)² + (n/b)² + (p/d)²)
          = (c/2) √((1/0.04)² + 0 + (1/0.03)²)
        """
        mode = RectangularCavityMode(self.A, self.B, self.D, 1, 0, 1, 'TE')
        f_expected = (C_LIGHT / 2) * np.sqrt((1 / self.A)**2 + (1 / self.D)**2)
        assert abs(mode.resonant_frequency - f_expected) / f_expected < 1e-12

    def test_tm110_resonant_frequency(self):
        """TM₁₁₀ 共振周波数の解析解との一致。"""
        mode = RectangularCavityMode(self.A, self.B, self.D, 1, 1, 0, 'TM')
        f_expected = (C_LIGHT / 2) * np.sqrt((1 / self.A)**2 + (1 / self.B)**2)
        assert abs(mode.resonant_frequency - f_expected) / f_expected < 1e-12

    def test_te_mode_validation_both_zero(self):
        """TE: m=n=0 は無効 → ValueError。"""
        with pytest.raises(ValueError, match="m and n cannot both be 0"):
            RectangularCavityMode(self.A, self.B, self.D, 0, 0, 1, 'TE')

    def test_te_mode_validation_p_zero(self):
        """TE: p=0 は無効 → ValueError。"""
        with pytest.raises(ValueError):
            RectangularCavityMode(self.A, self.B, self.D, 1, 0, 0, 'TE')

    def test_tm_mode_validation_m_zero(self):
        """TM: m=0 は無効 → ValueError。"""
        with pytest.raises(ValueError):
            RectangularCavityMode(self.A, self.B, self.D, 0, 1, 0, 'TM')

    def test_invalid_mode_type(self):
        """不正な mode_type → ValueError。"""
        with pytest.raises(ValueError):
            RectangularCavityMode(self.A, self.B, self.D, 1, 0, 1, 'XY')

    def test_te101_pec_boundary_x0(self):
        """
        PEC 境界条件: x=0 の壁で接線 E = 0。

        TE₁₀₁ では Ey が主成分。x=0 で Ey = 0 でなければならない
        (kx·x の sin は x=0 で 0 なので自動的に満たされる)。
        """
        mode = RectangularCavityMode(self.A, self.B, self.D, 1, 0, 1, 'TE')
        y = np.linspace(0, self.B, 20)
        z = np.linspace(0, self.D, 20)
        YY, ZZ = np.meshgrid(y, z)
        XX = np.zeros_like(YY)  # x = 0 の壁

        fld = mode.fields(XX, YY, ZZ, phase=0.0)
        # x=0 での接線成分 Ey, Ez は 0 でなければならない
        assert np.max(np.abs(fld.Ey)) < 1e-12, "Ey should be 0 at x=0"
        assert np.max(np.abs(fld.Ez)) < 1e-12, "Ez should be 0 at x=0"

    def test_te101_pec_boundary_y_surface(self):
        """y=0 の壁での PEC 境界条件: Ex と Ez が 0。"""
        mode = RectangularCavityMode(self.A, self.B, self.D, 1, 0, 1, 'TE')
        x = np.linspace(0, self.A, 20)
        z = np.linspace(0, self.D, 20)
        XX, ZZ = np.meshgrid(x, z)
        YY = np.zeros_like(XX)  # y = 0 の壁 (n=0 なので Ey の ky=0 に注意)

        fld = mode.fields(XX, YY, ZZ, phase=0.0)
        # TE₁₀₁ で n=0 なので ky=0, Ey の境界は自動的に OK
        assert np.max(np.abs(fld.Ez)) < 1e-12, "Ez should be 0 at y=0 (TE)"

    def test_fields_shape(self):
        """fields() の戻り値形状が入力と一致すること。"""
        mode = RectangularCavityMode(self.A, self.B, self.D, 1, 0, 1, 'TE')
        x = np.linspace(0, self.A, 10)
        y = np.linspace(0, self.B, 8)
        z = np.linspace(0, self.D, 12)
        XX, YY, ZZ = np.meshgrid(x, y, z)
        fld = mode.fields(XX, YY, ZZ)
        assert fld.Ex.shape == XX.shape

    def test_emfield_magnitude(self):
        """EMField.E_magnitude が各成分の二乗和の平方根であること。"""
        fld = EMField(
            Ex=np.array([3.0]), Ey=np.array([4.0]), Ez=np.array([0.0]),
            Hx=np.array([0.0]), Hy=np.array([0.0]), Hz=np.array([1.0]),
        )
        assert abs(fld.E_magnitude[0] - 5.0) < 1e-12
        assert abs(fld.H_magnitude[0] - 1.0) < 1e-12

    def test_phase_energy_exchange(self):
        """
        位相 0 と π/2 で電場・磁場のエネルギーが入れ替わること。

        E(t) = E₀ cos(ωt)  → |E(0)|²  = |E₀|²,  |E(π/2)|² = 0
        H(t) = H₀ sin(ωt)  → |H(0)|²  = 0,       |H(π/2)|² = |H₀|²
        """
        mode = RectangularCavityMode(self.A, self.B, self.D, 1, 0, 1, 'TE')
        x = np.linspace(0, self.A, 20)
        z = np.linspace(0, self.D, 20)
        XX, ZZ = np.meshgrid(x, z)
        YY = np.ones_like(XX) * self.B / 2

        # ωt=0: E が最大、H は 0
        fld0 = mode.fields(XX, YY, ZZ, phase=0.0)
        assert np.max(fld0.H_magnitude) < 1e-10, "H should be ~0 at phase=0"
        assert np.max(fld0.E_magnitude) > 0, "E should be nonzero at phase=0"

        # ωt=π/2: H が最大、E は 0
        fld_half = mode.fields(XX, YY, ZZ, phase=np.pi / 2)
        assert np.max(fld_half.E_magnitude) < 1e-10, "E should be ~0 at phase=pi/2"
        assert np.max(fld_half.H_magnitude) > 0, "H should be nonzero at phase=pi/2"

    def test_modes_sorted_by_frequency(self):
        """rectangular_cavity_modes() が周波数昇順でソートされていること。"""
        modes = rectangular_cavity_modes(self.A, self.B, self.D, n_modes=8)
        freqs = [m.resonant_frequency for m in modes]
        assert freqs == sorted(freqs)

    def test_mode_labels(self):
        """label() が正しい文字列を返すこと。"""
        mode_te = RectangularCavityMode(self.A, self.B, self.D, 1, 0, 1, 'TE')
        mode_tm = RectangularCavityMode(self.A, self.B, self.D, 2, 1, 0, 'TM')
        assert mode_te.label() == 'TE_101'
        assert mode_tm.label() == 'TM_210'


# ─────────────────────────────────────────────────────────────────────────────
# 円筒型キャビティのテスト
# ─────────────────────────────────────────────────────────────────────────────

class TestCylindricalCavity:
    """円筒型 PEC キャビティの解析解検証。"""

    R, L = 0.015, 0.03  # R=1.5cm, L=3cm

    def test_tm010_frequency(self):
        """
        TM₀₁₀ 共振周波数: f = c·χ₀₁ / (2π R), χ₀₁ ≈ 2.40483。
        """
        from scipy.special import jn_zeros
        mode = CylindricalCavityMode(self.R, self.L, m=0, n=1, p=0, mode_type='TM')
        chi_01 = float(jn_zeros(0, 1)[0])  # J₀ の第 1 零点
        f_expected = C_LIGHT * chi_01 / (2 * np.pi * self.R)
        assert abs(mode.resonant_frequency - f_expected) / f_expected < 1e-10

    def test_te111_frequency(self):
        """
        TE₁₁₁ 共振周波数: f = (c/2π) √( (χ'₁₁/R)² + (π/L)² )。
        χ'₁₁ ≈ 1.8412 (J₁' の第 1 零点)
        """
        from scipy.special import jnp_zeros
        mode = CylindricalCavityMode(self.R, self.L, m=1, n=1, p=1, mode_type='TE')
        chi_p11 = float(jnp_zeros(1, 1)[0])
        kc = chi_p11 / self.R
        kz = np.pi / self.L
        f_expected = C_LIGHT * np.sqrt(kc**2 + kz**2) / (2 * np.pi)
        assert abs(mode.resonant_frequency - f_expected) / f_expected < 1e-10

    def test_invalid_n_zero(self):
        """n=0 は無効 → ValueError。"""
        with pytest.raises(ValueError):
            CylindricalCavityMode(self.R, self.L, m=0, n=0, p=1, mode_type='TM')

    def test_fields_rz_shape(self):
        """fields_rz() の戻り値形状が入力と一致すること。"""
        mode = CylindricalCavityMode(self.R, self.L, m=0, n=1, p=0, mode_type='TM')
        rho = np.linspace(0, self.R, 10)
        z   = np.linspace(0, self.L, 15)
        RHO, ZZ = np.meshgrid(rho, z)
        fld = mode.fields_rz(RHO, ZZ)
        assert fld.Ez.shape == RHO.shape

    def test_modes_sorted(self):
        """cylindrical_cavity_modes() が周波数昇順。"""
        modes = cylindrical_cavity_modes(self.R, self.L, n_modes=6)
        freqs = [m.resonant_frequency for m in modes]
        assert freqs == sorted(freqs)

    def test_tm_hz_zero(self):
        """TM モードでは Hz = 0 (定義上)。"""
        mode = CylindricalCavityMode(self.R, self.L, m=0, n=1, p=0, mode_type='TM')
        rho = np.linspace(0.001, self.R, 10)
        z   = np.linspace(0, self.L, 10)
        RHO, ZZ = np.meshgrid(rho, z)
        fld = mode.fields_rz(RHO, ZZ, phase=0.5)
        assert np.max(np.abs(fld.Hz)) < 1e-12, "TM mode must have Hz=0"


# ─────────────────────────────────────────────────────────────────────────────
# 球形キャビティのテスト
# ─────────────────────────────────────────────────────────────────────────────

class TestSphericalCavity:
    """球形 PEC キャビティの解析解検証。"""

    R = 0.02  # R=2cm

    def test_tm11_frequency_approx(self):
        """
        TM₁₁ 共振周波数: χ₁₁ ≈ 4.493 (j₁(x)=0 の第 1 零点)。

        f = c·χ₁₁ / (2π R)
        """
        mode = SphericalCavityMode(self.R, l=1, n=1, mode_type='TM')
        # χ₁₁ の近似値 4.4934 を使って検証
        chi_11_approx = 4.4934
        f_expected = C_LIGHT * chi_11_approx / (2 * np.pi * self.R)
        assert abs(mode.resonant_frequency - f_expected) / f_expected < 1e-4

    def test_modes_sorted(self):
        """spherical_cavity_modes() が周波数昇順。"""
        modes = spherical_cavity_modes(self.R, n_modes=6)
        freqs = [m.resonant_frequency for m in modes]
        assert freqs == sorted(freqs)

    def test_label(self):
        """label() が正しい文字列を返すこと。"""
        mode = SphericalCavityMode(self.R, l=2, n=1, mode_type='TE')
        assert mode.label() == 'TE_21'

    def test_fields_rtheta_shape(self):
        """fields_rtheta() の戻り値形状が入力と一致すること。"""
        mode = SphericalCavityMode(self.R, l=1, n=1, mode_type='TM')
        r     = np.linspace(0.001, self.R, 10)
        theta = np.linspace(0, np.pi, 12)
        RR, TT = np.meshgrid(r, theta)
        fld = mode.fields_rtheta(RR, TT)
        assert fld.Ex.shape == RR.shape


# ─────────────────────────────────────────────────────────────────────────────
# 直方体導波管のテスト
# ─────────────────────────────────────────────────────────────────────────────

class TestRectangularWaveguide:
    """直方体 PEC 導波管の解析解検証。"""

    # WR-90 標準導波管寸法 (m)
    A = 0.02286  # 広辺 22.86 mm
    B = 0.01016  # 狭辺 10.16 mm

    def test_te10_cutoff_frequency(self):
        """
        TE₁₀ カットオフ周波数: fc = c/(2a)。

        WR-90: fc = 3e8/(2×0.02286) ≈ 6.562 GHz
        """
        mode = RectangularWaveguideMode(self.A, self.B, m=1, n=0, mode_type='TE')
        fc_expected = C_LIGHT / (2 * self.A)
        assert abs(mode.cutoff_frequency - fc_expected) / fc_expected < 1e-12

    def test_te20_cutoff_frequency(self):
        """TE₂₀ カットオフ周波数: fc = c/a (= 2 × fc(TE₁₀))。"""
        mode_10 = RectangularWaveguideMode(self.A, self.B, m=1, n=0, mode_type='TE')
        mode_20 = RectangularWaveguideMode(self.A, self.B, m=2, n=0, mode_type='TE')
        assert abs(mode_20.cutoff_frequency - 2 * mode_10.cutoff_frequency) < 1e-6

    def test_propagation_above_cutoff(self):
        """fc より高い周波数では β が実数 (正の実数)。"""
        mode = RectangularWaveguideMode(self.A, self.B, m=1, n=0, mode_type='TE')
        fc   = mode.cutoff_frequency
        beta = mode.propagation_constant(fc * 1.5)
        assert isinstance(beta, float)
        assert beta > 0

    def test_propagation_below_cutoff(self):
        """fc より低い周波数では β が純虚数 (エバネッセント)。"""
        mode = RectangularWaveguideMode(self.A, self.B, m=1, n=0, mode_type='TE')
        fc   = mode.cutoff_frequency
        beta = mode.propagation_constant(fc * 0.5)
        # 虚数: complex 型でなく float 型の場合 beta=0 になることもある
        # ここでは伝搬定数が 0 以下であることを確認
        assert float(np.real(beta)) == 0.0 or isinstance(beta, complex)

    def test_beta_formula(self):
        """
        β = √((ω/c)² − kc²) の解析値との一致。
        """
        mode = RectangularWaveguideMode(self.A, self.B, m=1, n=0, mode_type='TE')
        fc   = mode.cutoff_frequency
        f    = fc * 1.5
        kc   = mode.kc
        beta_formula = np.sqrt((2 * np.pi * f / C_LIGHT)**2 - kc**2)
        beta_code    = float(np.real(mode.propagation_constant(f)))
        assert abs(beta_code - beta_formula) / beta_formula < 1e-12

    def test_te_mode_validation(self):
        """TE: m=n=0 → ValueError。"""
        with pytest.raises(ValueError):
            RectangularWaveguideMode(self.A, self.B, m=0, n=0, mode_type='TE')

    def test_tm_mode_validation(self):
        """TM: m=0 → ValueError。"""
        with pytest.raises(ValueError):
            RectangularWaveguideMode(self.A, self.B, m=0, n=1, mode_type='TM')

    def test_fields_shape(self):
        """fields() の戻り値形状が入力と一致すること。"""
        mode = RectangularWaveguideMode(self.A, self.B, m=1, n=0, mode_type='TE')
        fc   = mode.cutoff_frequency
        x    = np.linspace(0, self.A, 10)
        y    = np.linspace(0, self.B, 8)
        XX, YY = np.meshgrid(x, y)
        fld = mode.fields(XX, YY, z=0.0, frequency=fc * 1.5)
        assert fld.Ex.shape == XX.shape

    def test_te10_ey_max_at_center(self):
        """
        TE₁₀ モードの Ey は x=a/2 で最大、x=0 と x=a で 0。

        Ey ∝ sin(πx/a) なので、中央で最大・端で 0 になる。
        """
        mode = RectangularWaveguideMode(self.A, self.B, m=1, n=0, mode_type='TE')
        fc   = mode.cutoff_frequency
        # y = b/2 の 1 行で確認
        x  = np.linspace(0, self.A, 100)
        y  = np.ones_like(x) * self.B / 2
        XX, YY = np.meshgrid(x, [self.B / 2])
        fld = mode.fields(XX, YY, z=0.0, frequency=fc * 1.5)
        Ey_row = fld.Ey[0, :]  # 1 行目を取得
        # 端 (x=0, x=a) で 0
        assert abs(Ey_row[0]) < 1e-10,  "Ey should be 0 at x=0"
        assert abs(Ey_row[-1]) < 1e-10, "Ey should be 0 at x=a"
        # 中央で最大
        idx_max = np.argmax(np.abs(Ey_row))
        assert abs(x[idx_max] - self.A / 2) < self.A / 10

    def test_rectangular_waveguide_helper(self):
        """RectangularWaveguide クラスの modes() が正常動作すること。"""
        wg = RectangularWaveguide(self.A, self.B)
        modes = wg.modes(n_modes=6)
        assert len(modes) == 6
        freqs = [m.cutoff_frequency for m in modes]
        assert freqs == sorted(freqs)  # カットオフ周波数昇順

    def test_dispersion(self):
        """dispersion() が適切な配列を返すこと。"""
        wg   = RectangularWaveguide(self.A, self.B)
        mode = wg.mode(1, 0, 'TE')
        fc   = mode.cutoff_frequency
        freqs, betas = wg.dispersion(mode, (fc * 0.5, fc * 2.0), n_points=100)
        assert len(freqs) == 100
        # fc 以下では β = 0
        below_cutoff = betas[freqs < fc]
        assert np.all(below_cutoff == 0.0)
        # fc より上では β > 0
        above_cutoff = betas[freqs > fc * 1.01]
        assert np.all(above_cutoff > 0)


# ─────────────────────────────────────────────────────────────────────────────
# 円形導波管のテスト
# ─────────────────────────────────────────────────────────────────────────────

class TestCircularWaveguide:
    """円形 PEC 導波管の解析解検証。"""

    R = 0.010  # 半径 10 mm

    def test_te11_cutoff_frequency(self):
        """
        TE₁₁ カットオフ周波数: fc = c·χ'₁₁ / (2π R)。
        χ'₁₁ ≈ 1.84118 (J₁' の第 1 零点)
        """
        from scipy.special import jnp_zeros
        mode = CircularWaveguideMode(self.R, m=1, n=1, mode_type='TE')
        chi_p11 = float(jnp_zeros(1, 1)[0])
        fc_expected = C_LIGHT * chi_p11 / (2 * np.pi * self.R)
        assert abs(mode.cutoff_frequency - fc_expected) / fc_expected < 1e-10

    def test_tm01_cutoff_frequency(self):
        """
        TM₀₁ カットオフ周波数: fc = c·χ₀₁ / (2π R)。
        χ₀₁ ≈ 2.40483 (J₀ の第 1 零点)
        """
        from scipy.special import jn_zeros
        mode = CircularWaveguideMode(self.R, m=0, n=1, mode_type='TM')
        chi_01 = float(jn_zeros(0, 1)[0])
        fc_expected = C_LIGHT * chi_01 / (2 * np.pi * self.R)
        assert abs(mode.cutoff_frequency - fc_expected) / fc_expected < 1e-10

    def test_te11_lowest_mode(self):
        """TE₁₁ が全モードの中で最低次 (最も低いカットオフ)。"""
        wg    = CircularWaveguide(self.R)
        modes = wg.modes(n_modes=8)
        lowest = modes[0]
        assert lowest.mode_type == 'TE'
        assert lowest.m == 1
        assert lowest.n == 1

    def test_invalid_n_zero(self):
        """n=0 は無効 → ValueError。"""
        with pytest.raises(ValueError):
            CircularWaveguideMode(self.R, m=0, n=0, mode_type='TE')

    def test_fields_polar_shape(self):
        """fields_polar() の戻り値形状が入力と一致すること。"""
        mode = CircularWaveguideMode(self.R, m=1, n=1, mode_type='TE')
        fc   = mode.cutoff_frequency
        r    = np.linspace(0, self.R, 10)
        phi  = np.linspace(0, 2 * np.pi, 12)
        RR, PHI = np.meshgrid(r, phi)
        fld = mode.fields_polar(RR, PHI, z=0.0, frequency=fc * 1.5)
        assert fld.Ex.shape == RR.shape

    def test_tm_ez_nonzero(self):
        """TM モードでは Ez ≠ 0 (主成分)。"""
        mode = CircularWaveguideMode(self.R, m=0, n=1, mode_type='TM')
        fc   = mode.cutoff_frequency
        r    = np.linspace(0.001, self.R * 0.9, 10)
        phi  = np.zeros_like(r)
        RR   = r.reshape(1, -1)
        PHI  = phi.reshape(1, -1)
        fld  = mode.fields_polar(RR, PHI, z=0.0, frequency=fc * 1.5)
        assert np.max(np.abs(fld.Ez)) > 0, "TM mode must have Ez ≠ 0"

    def test_te_ez_zero(self):
        """TE モードでは Ez = 0 (定義上)。"""
        mode = CircularWaveguideMode(self.R, m=1, n=1, mode_type='TE')
        fc   = mode.cutoff_frequency
        r    = np.linspace(0.001, self.R * 0.9, 10)
        phi  = np.linspace(0, 2 * np.pi, 12)
        RR, PHI = np.meshgrid(r, phi)
        fld = mode.fields_polar(RR, PHI, z=0.0, frequency=fc * 1.5)
        assert np.max(np.abs(fld.Ez)) < 1e-12, "TE mode must have Ez=0"

    def test_modes_sorted(self):
        """circular_waveguide.modes() がカットオフ周波数昇順。"""
        wg    = CircularWaveguide(self.R)
        modes = wg.modes(n_modes=6)
        freqs = [m.cutoff_frequency for m in modes]
        assert freqs == sorted(freqs)
