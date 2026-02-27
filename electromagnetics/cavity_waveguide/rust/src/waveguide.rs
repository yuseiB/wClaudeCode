/*!
 * waveguide.rs — 直方体・円形 PEC 導波管の伝搬モード解析 (Rust)
 *
 * 対応導波管:
 *   RectangularWaveguideMode — 直方体 PEC 導波管 a×b
 *   CircularWaveguideMode    — 円形 PEC 導波管 (半径 R)
 *
 * 物理的な背景:
 *   PEC 境界条件: 導体壁面で接線成分 E_tan = 0
 *   伝搬定数: β = √((ω/c)² − kc²) [f > fc: 伝搬, f < fc: エバネッセント]
 *   分散関係: ω² = (βc)² + (kc·c)² — 双曲線型, ω=βc の光線とは異なる
 *   群速度 vg = c²β/ω < c (情報伝搬速度)
 *   位相速度 vp = ω/β > c (超光速だが情報を伝えない)
 */

use crate::constants::*;
use crate::EMPoint;
use crate::cavity::{ModeType, TM_ZEROS, TE_ZEROS, jm, djm};

// ─────────────────────────────────────────────────────────────────────────────
// 直方体導波管
// ─────────────────────────────────────────────────────────────────────────────

/// 直方体 PEC 導波管 (0≤x≤a, 0≤y≤b, +z 方向伝搬) の TE/TM モード
///
/// TE_mn: E_z=0, m,n≥0 (両方 0 は不可)
/// TM_mn: H_z=0, m,n≥1
///
/// カットオフ波数: kc = π√((m/a)² + (n/b)²)
/// カットオフ周波数: fc = c·kc / (2π)
/// 伝搬定数: β = √((ω/c)² − kc²) [f > fc のとき正の実数]
#[derive(Debug, Clone)]
pub struct RectangularWaveguideMode {
    pub a: f64, pub b: f64,
    pub m: u32, pub n: u32,
    pub mode_type: ModeType,
    // 事前計算値
    kx: f64, ky: f64,
    kc2: f64,  // kc²
    pub fc: f64,   // カットオフ周波数
}

impl RectangularWaveguideMode {
    /// 新しいモードを生成する。
    pub fn new(a: f64, b: f64, m: u32, n: u32,
               mode_type: ModeType) -> Result<Self, &'static str>
    {
        match mode_type {
            ModeType::TE => {
                if m == 0 && n == 0 {
                    return Err("TE: m and n cannot both be 0");
                }
            }
            ModeType::TM => {
                if m < 1 || n < 1 {
                    return Err("TM: m and n must be >= 1");
                }
            }
        }
        let kx  = m as f64 * PI / a;
        let ky  = n as f64 * PI / b;
        let kc2 = kx*kx + ky*ky;
        let fc  = C_LIGHT * kc2.sqrt() / (2.0 * PI);
        Ok(Self { a, b, m, n, mode_type, kx, ky, kc2, fc })
    }

    /// カットオフ周波数 (Hz)
    pub fn cutoff_frequency(&self) -> f64 { self.fc }

    /// モードラベル (例: "TE_10")
    pub fn label(&self) -> String {
        let t = match self.mode_type { ModeType::TE => "TE", ModeType::TM => "TM" };
        format!("{}_{}{}", t, self.m, self.n)
    }

    /// 伝搬定数 β (rad/m)。カットオフ以下なら 0.0 を返す。
    pub fn propagation_constant(&self, frequency: f64) -> f64 {
        let k  = 2.0 * PI * frequency / C_LIGHT;
        let d  = k*k - self.kc2;
        if d >= 0.0 { d.sqrt() } else { 0.0 }
    }

    /// 横断面上の EM 場を計算する (軸位置 z, 位相 ωt)。
    ///
    /// 伝搬モード: E ~ cos(ωt − βz), H ~ cos(ωt − βz)
    ///
    /// # Arguments
    /// * `x, y`      - 横断面座標 (m)
    /// * `z`         - 軸方向座標 (m)
    /// * `frequency` - 動作周波数 (Hz)
    /// * `phase`     - ωt (rad)
    pub fn fields(&self, x: f64, y: f64, z: f64,
                  frequency: f64, phase: f64) -> EMPoint
    {
        let beta  = self.propagation_constant(frequency);
        let omega = 2.0 * PI * frequency;
        let psi   = (phase - beta * z).cos();

        let cx = (self.kx * x).cos();
        let sx = (self.kx * x).sin();
        let cy = (self.ky * y).cos();
        let sy = (self.ky * y).sin();

        let mut f = EMPoint::default();
        if self.kc2 < 1e-30 { return f; }

        match self.mode_type {
            ModeType::TE => {
                f.hz = cx * cy * psi;
                f.hx =  (beta * self.kx / self.kc2) * sx * cy * psi;
                f.hy =  (beta * self.ky / self.kc2) * cx * sy * psi;
                f.ex = -(omega * MU0 * self.ky / self.kc2) * cx * sy * psi;
                f.ey =  (omega * MU0 * self.kx / self.kc2) * sx * cy * psi;
            }
            ModeType::TM => {
                f.ez = sx * sy * psi;
                f.ex = -(beta * self.kx / self.kc2) * cx * sy * psi;
                f.ey = -(beta * self.ky / self.kc2) * sx * cy * psi;
                f.hx =  (omega * EPS0 * self.ky / self.kc2) * sx * cy * psi;
                f.hy = -(omega * EPS0 * self.kx / self.kc2) * cx * sy * psi;
            }
        }
        f
    }
}

/// 直方体導波管の最低次 `n_modes` モードをカットオフ周波数昇順で返す。
pub fn rectangular_waveguide_modes(a: f64, b: f64,
                                    n_modes: usize) -> Vec<RectangularWaveguideMode>
{
    let mut modes = Vec::new();
    for m in 0u32..=5 {
        for n in 0u32..=5 {
            for &t in &[ModeType::TE, ModeType::TM] {
                if let Ok(mo) = RectangularWaveguideMode::new(a, b, m, n, t) {
                    modes.push(mo);
                }
            }
        }
    }
    modes.sort_by(|a, b| {
        a.cutoff_frequency().partial_cmp(&b.cutoff_frequency()).unwrap()
    });
    modes.truncate(n_modes);
    modes
}


// ─────────────────────────────────────────────────────────────────────────────
// 円形導波管
// ─────────────────────────────────────────────────────────────────────────────

/// 円形 PEC 導波管 (0≤ρ≤R, +z 方向伝搬) の TE/TM モード
///
/// TM_mn: kc = χ_mn/R (J_m の零点)
/// TE_mn: kc = χ'_mn/R (J_m' の零点)
///
/// カットオフ周波数: fc = c·kc / (2π)
#[derive(Debug, Clone)]
pub struct CircularWaveguideMode {
    pub r: f64,
    pub m: u32, pub n: u32,
    pub mode_type: ModeType,
    chi: f64,
    kc: f64,
    kc2: f64,
    pub fc: f64,
}

impl CircularWaveguideMode {
    /// 新しいモードを生成する。
    pub fn new(r: f64, m: u32, n: u32,
               mode_type: ModeType) -> Result<Self, &'static str>
    {
        if n < 1 { return Err("n must be >= 1"); }
        if m > 4 || n > 3 { return Err("m/n out of Bessel table range"); }

        let chi = match mode_type {
            ModeType::TM => TM_ZEROS[m as usize][(n - 1) as usize],
            ModeType::TE => TE_ZEROS[m as usize][(n - 1) as usize],
        };
        let kc  = chi / r;
        let kc2 = kc * kc;
        let fc  = C_LIGHT * kc / (2.0 * PI);
        Ok(Self { r, m, n, mode_type, chi, kc, kc2, fc })
    }

    /// カットオフ周波数 (Hz)
    pub fn cutoff_frequency(&self) -> f64 { self.fc }

    /// モードラベル (例: "TE_11")
    pub fn label(&self) -> String {
        let t = match self.mode_type { ModeType::TE => "TE", ModeType::TM => "TM" };
        format!("{}_{}{}", t, self.m, self.n)
    }

    /// 伝搬定数 β (rad/m)。カットオフ以下なら 0.0 を返す。
    pub fn propagation_constant(&self, frequency: f64) -> f64 {
        let k = 2.0 * PI * frequency / C_LIGHT;
        let d = k*k - self.kc2;
        if d >= 0.0 { d.sqrt() } else { 0.0 }
    }

    /// 極座標断面上の EM 場を計算する。
    ///
    /// 戻り値の (ex,ey,ez) → (Eρ,Eφ,Ez), (hx,hy,hz) → (Hρ,Hφ,Hz)
    ///
    /// # Arguments
    /// * `rho, phi`  - 極座標 (m, rad)
    /// * `z`         - 軸方向座標 (m)
    /// * `frequency` - 動作周波数 (Hz)
    /// * `phase`     - ωt (rad)
    pub fn fields_polar(&self, rho: f64, phi: f64, z: f64,
                         frequency: f64, phase: f64) -> EMPoint
    {
        let beta  = self.propagation_constant(frequency);
        let omega = 2.0 * PI * frequency;
        let psi   = (phase - beta * z).cos();

        let kc_rho  = self.kc * rho;
        let jm_val  = jm(self.m as i32, kc_rho);
        let djm_val = djm(self.m as i32, kc_rho) * self.kc;

        let cos_mphi = (self.m as f64 * phi).cos();
        let sin_mphi = (self.m as f64 * phi).sin();
        let rho_safe = rho.max(1e-30);

        let mut f = EMPoint::default();
        if self.kc2 < 1e-30 { return f; }

        match self.mode_type {
            ModeType::TM => {
                f.ez  = jm_val * cos_mphi * psi;
                f.ex  = -(beta * self.kc / self.kc2) * djm_val * cos_mphi * psi;
                f.ey  =  (beta * self.m as f64 / (self.kc2 * rho_safe)) * jm_val * sin_mphi * psi;
                f.hx  = -(omega * EPS0 * self.m as f64 / (self.kc2 * rho_safe)) * jm_val * sin_mphi * psi;
                f.hy  =  (omega * EPS0 * self.kc / self.kc2) * djm_val * cos_mphi * psi;
            }
            ModeType::TE => {
                f.hz  = jm_val * cos_mphi * psi;
                f.hx  = -(beta * self.kc / self.kc2) * djm_val * cos_mphi * psi;
                f.hy  =  (beta * self.m as f64 / (self.kc2 * rho_safe)) * jm_val * sin_mphi * psi;
                f.ex  =  (omega * MU0 * self.m as f64 / (self.kc2 * rho_safe)) * jm_val * sin_mphi * psi;
                f.ey  = -(omega * MU0 * self.kc / self.kc2) * djm_val * cos_mphi * psi;
            }
        }
        f
    }
}

/// 円形導波管の最低次 `n_modes` モードをカットオフ周波数昇順で返す。
pub fn circular_waveguide_modes(r: f64,
                                  n_modes: usize) -> Vec<CircularWaveguideMode>
{
    let mut modes = Vec::new();
    for m in 0u32..=4 {
        for n in 1u32..=3 {
            for &t in &[ModeType::TE, ModeType::TM] {
                if let Ok(mo) = CircularWaveguideMode::new(r, m, n, t) {
                    modes.push(mo);
                }
            }
        }
    }
    modes.sort_by(|a, b| {
        a.cutoff_frequency().partial_cmp(&b.cutoff_frequency()).unwrap()
    });
    modes.truncate(n_modes);
    modes
}


// ─────────────────────────────────────────────────────────────────────────────
// テスト
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const A: f64 = 0.04;  // 4 cm (a > b)
    const B: f64 = 0.02;  // 2 cm
    const R: f64 = 0.015; // 1.5 cm

    /// TE_10 カットオフ周波数: fc = c/(2a)
    #[test]
    fn test_te10_cutoff() {
        let mode = RectangularWaveguideMode::new(A, B, 1, 0, ModeType::TE).unwrap();
        let fc_expected = C_LIGHT / (2.0 * A);
        assert!((mode.cutoff_frequency() - fc_expected).abs() < 1e6,
            "fc = {:.3e}, expected {:.3e}", mode.cutoff_frequency(), fc_expected);
    }

    /// TE_20 のカットオフ周波数は TE_10 の 2 倍
    #[test]
    fn test_te20_cutoff_double() {
        let m10 = RectangularWaveguideMode::new(A, B, 1, 0, ModeType::TE).unwrap();
        let m20 = RectangularWaveguideMode::new(A, B, 2, 0, ModeType::TE).unwrap();
        let ratio = m20.cutoff_frequency() / m10.cutoff_frequency();
        assert!((ratio - 2.0).abs() < 1e-10);
    }

    /// fc より高い周波数では β > 0
    #[test]
    fn test_propagating_above_cutoff() {
        let mode = RectangularWaveguideMode::new(A, B, 1, 0, ModeType::TE).unwrap();
        let f = mode.fc * 1.5;
        assert!(mode.propagation_constant(f) > 0.0);
    }

    /// fc より低い周波数では β = 0 (エバネッセント)
    #[test]
    fn test_evanescent_below_cutoff() {
        let mode = RectangularWaveguideMode::new(A, B, 1, 0, ModeType::TE).unwrap();
        let f = mode.fc * 0.5;
        assert_eq!(mode.propagation_constant(f), 0.0);
    }

    /// TE_mn: m=n=0 はエラー
    #[test]
    fn test_te_mn_zero_error() {
        assert!(RectangularWaveguideMode::new(A, B, 0, 0, ModeType::TE).is_err());
    }

    /// TM_mn: m=0 はエラー
    #[test]
    fn test_tm_m_zero_error() {
        assert!(RectangularWaveguideMode::new(A, B, 0, 1, ModeType::TM).is_err());
    }

    /// TE_10 モードラベル
    #[test]
    fn test_label() {
        let mode = RectangularWaveguideMode::new(A, B, 1, 0, ModeType::TE).unwrap();
        assert_eq!(mode.label(), "TE_10");
    }

    /// モードリストがカットオフ周波数昇順
    #[test]
    fn test_modes_sorted() {
        let modes = rectangular_waveguide_modes(A, B, 6);
        for i in 1..modes.len() {
            assert!(modes[i].cutoff_frequency() >= modes[i-1].cutoff_frequency());
        }
    }

    // ── 円形導波管 ────────────────────────────────────────────────────────────

    /// TE_11 は円形導波管の最低次モード (χ'₁₁ = 1.8412)
    #[test]
    fn test_circ_te11_lowest() {
        let te11 = CircularWaveguideMode::new(R, 1, 1, ModeType::TE).unwrap();
        let tm01 = CircularWaveguideMode::new(R, 0, 1, ModeType::TM).unwrap();
        assert!(te11.cutoff_frequency() < tm01.cutoff_frequency());
    }

    /// TE_11 カットオフ周波数: fc = c·χ'₁₁/(2πR)
    #[test]
    fn test_circ_te11_cutoff() {
        let chi_11_prime = 1.8412;
        let expected = C_LIGHT * chi_11_prime / (2.0 * PI * R);
        let mode = CircularWaveguideMode::new(R, 1, 1, ModeType::TE).unwrap();
        assert!((mode.cutoff_frequency() - expected).abs() / expected < 0.001);
    }

    /// TM_01 カットオフ周波数: fc = c·χ₀₁/(2πR)  (χ₀₁ = 2.4048)
    #[test]
    fn test_circ_tm01_cutoff() {
        let chi_01 = 2.4048;
        let expected = C_LIGHT * chi_01 / (2.0 * PI * R);
        let mode = CircularWaveguideMode::new(R, 0, 1, ModeType::TM).unwrap();
        assert!((mode.cutoff_frequency() - expected).abs() / expected < 0.001);
    }

    /// n=0 はエラー
    #[test]
    fn test_circ_n_zero_error() {
        assert!(CircularWaveguideMode::new(R, 0, 0, ModeType::TM).is_err());
    }

    /// TE モードでは Ez = 0
    #[test]
    fn test_circ_te_ez_zero() {
        let mode = CircularWaveguideMode::new(R, 1, 1, ModeType::TE).unwrap();
        let f = mode.fc * 1.5;
        let p = mode.fields_polar(0.005, 0.0, 0.0, f, 0.0);
        assert!(p.ez.abs() < 1e-20);
    }

    /// TM モードでは Hz = 0
    #[test]
    fn test_circ_tm_hz_zero() {
        let mode = CircularWaveguideMode::new(R, 0, 1, ModeType::TM).unwrap();
        let f = mode.fc * 1.5;
        let p = mode.fields_polar(0.005, 0.0, 0.0, f, 0.0);
        assert!(p.hz.abs() < 1e-20);
    }
}
