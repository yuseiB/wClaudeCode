/*!
 * cavity.rs — 電磁気共振キャビティの解析解 (Rust)
 *
 * 対応キャビティ:
 *   RectangularCavityMode — 直方体 PEC キャビティ a×b×d
 *   CylindricalCavityMode — 円筒型 PEC キャビティ (半径 R, 高さ L)
 *
 * 物理的な背景:
 *   PEC 境界条件: 導体壁面で接線成分 E_tan = 0
 *   時間依存性: E(r,t)=E₀(r)cos(ωt), H(r,t)=H₀(r)sin(ωt)
 *   エネルギーが電場 ↔ 磁場の間で振動 (90° 位相差)
 */

use crate::constants::*;
use crate::EMPoint;

// ─────────────────────────────────────────────────────────────────────────────
// Bessel 零点テーブル
//
// J_m の零点 χ_mn (TM モード用): rows=m(0..4), cols=n(1..3)
// J_m' の零点 χ'_mn (TE モード用): rows=m(0..4), cols=n(1..3)
//
// 値: Pozar "Microwave Engineering" Table 3.4, 3.5
// ─────────────────────────────────────────────────────────────────────────────

/// J_m(χ_mn) = 0 の零点
pub const TM_ZEROS: [[f64; 3]; 5] = [
    [2.4048, 5.5201, 8.6537],   // m=0
    [3.8317, 7.0156, 10.1735],  // m=1
    [5.1356, 8.4172, 11.6198],  // m=2
    [6.3802, 9.7610, 13.0152],  // m=3
    [7.5883, 11.0647, 14.3725], // m=4
];

/// J_m'(χ'_mn) = 0 の零点
pub const TE_ZEROS: [[f64; 3]; 5] = [
    [3.8317, 7.0156, 10.1735],  // m=0
    [1.8412, 5.3314, 8.5363],   // m=1 (最低次 TE モード χ'₁₁)
    [3.0542, 6.7061, 9.9695],   // m=2
    [4.2012, 8.0152, 11.3459],  // m=3
    [5.3175, 9.2824, 12.6819],  // m=4
];

/// Bessel 関数 J_0(x) — 級数展開 (|x| < 20 で十分な精度)
fn j0(x: f64) -> f64 {
    if x == 0.0 { return 1.0; }
    let mut s = 1.0_f64;
    let mut term = 1.0_f64;
    let x2 = x * x / 4.0;
    for k in 1..=30 {
        term *= -x2 / (k as f64 * k as f64);
        s += term;
        if term.abs() < 1e-15 * s.abs() { break; }
    }
    s
}

/// Bessel 関数 J_1(x) — 級数展開
fn j1(x: f64) -> f64 {
    if x == 0.0 { return 0.0; }
    let mut s = 0.5_f64 * x;
    let mut term = 0.5_f64 * x;
    let x2 = x * x / 4.0;
    for k in 1..=30 {
        term *= -x2 / (k as f64 * (k + 1) as f64);
        s += term;
        if term.abs() < 1e-15 * s.abs() { break; }
    }
    s
}

/// Bessel 関数 J_m(x) (m ≥ 0)
pub fn jm(m: i32, x: f64) -> f64 {
    match m {
        0 => j0(x),
        1 => j1(x),
        n if n > 1 => {
            // 前進再帰: J_{n+1}(x) = (2n/x) J_n(x) - J_{n-1}(x)
            if x.abs() < 1e-30 { return 0.0; }
            let mut jn_prev = j0(x);
            let mut jn_curr = j1(x);
            for k in 1..n {
                let jn_next = (2.0 * k as f64 / x) * jn_curr - jn_prev;
                jn_prev = jn_curr;
                jn_curr = jn_next;
            }
            jn_curr
        }
        _ => 0.0,
    }
}

/// Bessel 関数の導関数 J_m'(x) ≈ (J_{m-1}(x) − J_{m+1}(x)) / 2
pub fn djm(m: i32, x: f64) -> f64 {
    if m == 0 {
        -jm(1, x)
    } else {
        0.5 * (jm(m - 1, x) - jm(m + 1, x))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 直方体キャビティ
// ─────────────────────────────────────────────────────────────────────────────

/// モードの種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeType { TE, TM }

/// 直方体 PEC キャビティ (0≤x≤a, 0≤y≤b, 0≤z≤d) の TE/TM モード
///
/// TE_mnp: E_z=0, m,n≥0 (両方 0 は不可), p≥1
/// TM_mnp: H_z=0, m,n≥1, p≥0
///
/// 共振周波数: f = (c/2) √((m/a)² + (n/b)² + (p/d)²)
#[derive(Debug, Clone)]
pub struct RectangularCavityMode {
    pub a: f64, pub b: f64, pub d: f64,
    pub m: u32, pub n: u32, pub p: u32,
    pub mode_type: ModeType,
    // 事前計算値
    kx: f64, ky: f64, kz: f64,
    kc2: f64,   // 横断面波数の 2 乗 kx²+ky²
    omega: f64, // 角周波数 (rad/s)
}

impl RectangularCavityMode {
    /// 新しいモードを生成する。
    ///
    /// # Errors
    /// 無効なモード指数の組み合わせで `Err(&str)` を返す。
    pub fn new(a: f64, b: f64, d: f64,
               m: u32, n: u32, p: u32,
               mode_type: ModeType) -> Result<Self, &'static str>
    {
        // バリデーション
        match mode_type {
            ModeType::TE => {
                if m == 0 && n == 0 {
                    return Err("TE: m and n cannot both be 0");
                }
                if p < 1 {
                    return Err("TE: p must be >= 1");
                }
            }
            ModeType::TM => {
                if m < 1 || n < 1 {
                    return Err("TM: m and n must be >= 1");
                }
            }
        }

        // 事前計算
        let kx   = m as f64 * PI / a;
        let ky   = n as f64 * PI / b;
        let kz   = p as f64 * PI / d;
        let k    = (kx*kx + ky*ky + kz*kz).sqrt();
        let kc2  = kx*kx + ky*ky;
        let omega = C_LIGHT * k;

        Ok(Self { a, b, d, m, n, p, mode_type, kx, ky, kz, kc2, omega })
    }

    /// 共振周波数 (Hz)
    pub fn resonant_frequency(&self) -> f64 {
        self.omega / (2.0 * PI)
    }

    /// モードラベル (例: "TE_101")
    pub fn label(&self) -> String {
        let t = match self.mode_type { ModeType::TE => "TE", ModeType::TM => "TM" };
        format!("{}_{}{}{}", t, self.m, self.n, self.p)
    }

    /// 位置 (x, y, z) での EM 場を計算する。
    ///
    /// 時間依存性:
    ///   E(r,t) = E₀(r)·cos(ωt + phase)  → phase=ωt で評価
    ///   H(r,t) = H₀(r)·sin(ωt + phase)
    ///
    /// # Arguments
    /// * `x,y,z` - 空間座標 (m)
    /// * `phase` - ωt (rad)
    pub fn fields(&self, x: f64, y: f64, z: f64, phase: f64) -> EMPoint {
        let ct = phase.cos();  // cos(ωt): 電場の時間因子
        let st = phase.sin();  // sin(ωt): 磁場の時間因子

        let cx = (self.kx * x).cos();
        let sx = (self.kx * x).sin();
        let cy = (self.ky * y).cos();
        let sy = (self.ky * y).sin();
        let cz = (self.kz * z).cos();
        let sz = (self.kz * z).sin();

        let mut f = EMPoint::default();

        match self.mode_type {
            ModeType::TE => {
                // H_z = cos(kx·x) cos(ky·y) sin(kz·z) · sin(ωt)
                f.hz = cx * cy * sz * st;
                if self.kc2 > 0.0 {
                    let g2 = self.kc2;
                    f.ex = ( self.omega * MU0 * self.ky / g2) * cx * sy * sz * ct;
                    f.ey = (-self.omega * MU0 * self.kx / g2) * sx * cy * sz * ct;
                    f.hx = -(self.kx * self.kz / g2) * sx * cy * cz * st;
                    f.hy = -(self.ky * self.kz / g2) * cx * sy * cz * st;
                }
            }
            ModeType::TM => {
                // E_z = sin(kx·x) sin(ky·y) cos(kz·z) · cos(ωt)
                f.ez = sx * sy * cz * ct;
                if self.kc2 > 0.0 {
                    let g2 = self.kc2;
                    f.ex = (self.kx * self.kz / g2) * cx * sy * sz * ct;
                    f.ey = (self.ky * self.kz / g2) * sx * cy * sz * ct;
                    f.hx = ( self.omega * EPS0 * self.ky / g2) * sx * cy * cz * st;
                    f.hy = (-self.omega * EPS0 * self.kx / g2) * cx * sy * cz * st;
                }
            }
        }
        f
    }
}

/// 直方体キャビティの最低次 `n_modes` モードを周波数昇順で返す。
pub fn rectangular_cavity_modes(a: f64, b: f64, d: f64,
                                 n_modes: usize) -> Vec<RectangularCavityMode>
{
    let mut modes = Vec::new();
    for m in 0u32..=5 {
        for n in 0u32..=5 {
            for p in 0u32..=5 {
                for &t in &[ModeType::TE, ModeType::TM] {
                    if let Ok(mo) = RectangularCavityMode::new(a, b, d, m, n, p, t) {
                        modes.push(mo);
                    }
                }
            }
        }
    }
    modes.sort_by(|a, b| {
        a.resonant_frequency().partial_cmp(&b.resonant_frequency()).unwrap()
    });
    modes.truncate(n_modes);
    modes
}

// ─────────────────────────────────────────────────────────────────────────────
// 円筒型キャビティ
// ─────────────────────────────────────────────────────────────────────────────

/// 円筒型 PEC キャビティ (0≤ρ≤R, 0≤z≤L) の TM/TE モード
///
/// TM_mnp: kc = χ_mn/R (J_m の零点), p≥0
/// TE_mnp: kc = χ'_mn/R (J_m' の零点), p≥1
///
/// 共振周波数: f = (c/2π) √((χ/R)² + (pπ/L)²)
#[derive(Debug, Clone)]
pub struct CylindricalCavityMode {
    pub r: f64, pub l: f64,
    pub m: u32, pub n: u32, pub p: u32,
    pub mode_type: ModeType,
    // 事前計算値
    chi: f64,    // Bessel 零点
    kc: f64,     // 横断面波数
    kz: f64,     // 軸方向波数
    omega: f64,  // 角周波数
}

impl CylindricalCavityMode {
    /// 新しいモードを生成する。
    pub fn new(r: f64, l: f64,
               m: u32, n: u32, p: u32,
               mode_type: ModeType) -> Result<Self, &'static str>
    {
        if n < 1 { return Err("n must be >= 1"); }
        if m > 4 || n > 3 { return Err("m/n out of Bessel table range"); }

        match mode_type {
            ModeType::TM => {
                if p > 10 { return Err("p out of range"); }
            }
            ModeType::TE => {
                if p < 1 { return Err("TE: p must be >= 1"); }
            }
        }

        // Bessel 零点 χ を取得
        let chi = match mode_type {
            ModeType::TM => TM_ZEROS[m as usize][(n - 1) as usize],
            ModeType::TE => TE_ZEROS[m as usize][(n - 1) as usize],
        };

        let kc    = chi / r;
        let kz    = p as f64 * PI / l;
        let omega = C_LIGHT * (kc*kc + kz*kz).sqrt();

        Ok(Self { r, l, m, n, p, mode_type, chi, kc, kz, omega })
    }

    /// 共振周波数 (Hz)
    pub fn resonant_frequency(&self) -> f64 {
        self.omega / (2.0 * PI)
    }

    /// モードラベル (例: "TM_010")
    pub fn label(&self) -> String {
        let t = match self.mode_type { ModeType::TE => "TE", ModeType::TM => "TM" };
        format!("{}_{}{}{}", t, self.m, self.n, self.p)
    }

    /// ρ-z 断面上の EM 場を計算する。
    ///
    /// 戻り値の (ex,ey,ez) は (Eρ,Eφ,Ez)、(hx,hy,hz) は (Hρ,Hφ,Hz) に対応。
    ///
    /// # Arguments
    /// * `rho`   - 動径座標 (m)
    /// * `z`     - 軸方向座標 (m)
    /// * `phi`   - 方位角 (rad)
    /// * `phase` - ωt (rad)
    pub fn fields_rz(&self, rho: f64, z: f64, phi: f64, phase: f64) -> EMPoint {
        let ct = phase.cos();
        let st = phase.sin();

        let kc_rho = self.kc * rho;
        let jm_val = jm(self.m as i32, kc_rho);
        let djm_val = djm(self.m as i32, kc_rho) * self.kc;  // dJ/dρ

        let cos_mphi = (self.m as f64 * phi).cos();
        let sin_mphi = (self.m as f64 * phi).sin();
        let cz = (self.kz * z).cos();
        let sz = (self.kz * z).sin();
        let rho_safe = rho.max(1e-30);
        let kc2 = self.kc * self.kc;

        let mut f = EMPoint::default();

        match self.mode_type {
            ModeType::TM => {
                // E_z = J_m(kc·ρ) cos(mφ) cos(kz·z) · cos(ωt)
                f.ez = jm_val * cos_mphi * cz * ct;
                if self.kc > 0.0 {
                    f.ex =  -(self.kz / self.kc) * djm_val * cos_mphi * sz * ct;
                    f.ey =   (self.kz * self.m as f64 / (kc2 * rho_safe)) * jm_val * sin_mphi * sz * ct;
                    f.hx =   (self.omega * EPS0 * self.m as f64 / (kc2 * rho_safe)) * jm_val * sin_mphi * cz * st;
                    f.hy =  -(self.omega * EPS0 / self.kc) * djm_val * cos_mphi * cz * st;
                }
            }
            ModeType::TE => {
                // H_z = J_m(kc·ρ) cos(mφ) sin(kz·z) · sin(ωt)
                f.hz = jm_val * cos_mphi * sz * st;
                if self.kc > 0.0 {
                    f.hx =  -(self.kz / self.kc) * djm_val * cos_mphi * cz * st;
                    f.hy =   (self.kz * self.m as f64 / (kc2 * rho_safe)) * jm_val * sin_mphi * cz * st;
                    f.ex =  -(self.omega * MU0 * self.m as f64 / (kc2 * rho_safe)) * jm_val * sin_mphi * sz * ct;
                    f.ey =  -(self.omega * MU0 / self.kc) * djm_val * cos_mphi * sz * ct;
                }
            }
        }
        f
    }
}

/// 円筒型キャビティの最低次 `n_modes` モードを周波数昇順で返す。
pub fn cylindrical_cavity_modes(r: f64, l: f64,
                                 n_modes: usize) -> Vec<CylindricalCavityMode>
{
    let mut modes = Vec::new();
    for m in 0u32..=4 {
        for n in 1u32..=3 {
            for p in 0u32..=3 {
                for &t in &[ModeType::TM, ModeType::TE] {
                    if let Ok(mo) = CylindricalCavityMode::new(r, l, m, n, p, t) {
                        modes.push(mo);
                    }
                }
            }
        }
    }
    modes.sort_by(|a, b| {
        a.resonant_frequency().partial_cmp(&b.resonant_frequency()).unwrap()
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

    const A: f64 = 0.04;  // 4 cm
    const B: f64 = 0.02;  // 2 cm
    const D: f64 = 0.03;  // 3 cm

    #[test]
    fn test_te101_frequency() {
        let mode = RectangularCavityMode::new(A, B, D, 1, 0, 1, ModeType::TE).unwrap();
        let expected = C_LIGHT / 2.0 * ((1.0/A).powi(2) + (1.0/D).powi(2)).sqrt();
        assert!((mode.resonant_frequency() - expected).abs() < 1e6);
    }

    #[test]
    fn test_tm110_frequency() {
        let mode = RectangularCavityMode::new(A, B, D, 1, 1, 0, ModeType::TM).unwrap();
        let expected = C_LIGHT / 2.0 * ((1.0/A).powi(2) + (1.0/B).powi(2)).sqrt();
        assert!((mode.resonant_frequency() - expected).abs() < 1e6);
    }

    #[test]
    fn test_te_mn_zero_error() {
        assert!(RectangularCavityMode::new(A, B, D, 0, 0, 1, ModeType::TE).is_err());
    }

    #[test]
    fn test_tm_m_zero_error() {
        assert!(RectangularCavityMode::new(A, B, D, 0, 1, 1, ModeType::TM).is_err());
    }

    #[test]
    fn test_te101_e_nonzero_at_phase0() {
        let mode = RectangularCavityMode::new(A, B, D, 1, 0, 1, ModeType::TE).unwrap();
        let f = mode.fields(A/2.0, B/2.0, D/2.0, 0.0);
        assert!(f.e_mag() > 0.0);
    }

    #[test]
    fn test_te101_h_zero_at_phase0() {
        let mode = RectangularCavityMode::new(A, B, D, 1, 0, 1, ModeType::TE).unwrap();
        let f = mode.fields(A/2.0, B/2.0, D/2.0, 0.0);
        assert!(f.h_mag() < 1e-20);
    }

    #[test]
    fn test_te_ez_zero() {
        let mode = RectangularCavityMode::new(A, B, D, 1, 0, 1, ModeType::TE).unwrap();
        let f = mode.fields(A/3.0, B/3.0, D/3.0, 0.5);
        assert_eq!(f.ez, 0.0);
    }

    #[test]
    fn test_tm_hz_zero() {
        let mode = RectangularCavityMode::new(A, B, D, 1, 1, 1, ModeType::TM).unwrap();
        let f = mode.fields(A/3.0, B/3.0, D/3.0, 0.5);
        assert_eq!(f.hz, 0.0);
    }

    #[test]
    fn test_label() {
        let mode = RectangularCavityMode::new(A, B, D, 1, 0, 1, ModeType::TE).unwrap();
        assert_eq!(mode.label(), "TE_101");
    }

    #[test]
    fn test_modes_sorted() {
        let modes = rectangular_cavity_modes(A, B, D, 6);
        for i in 1..modes.len() {
            assert!(modes[i].resonant_frequency() >= modes[i-1].resonant_frequency());
        }
    }

    const R_CYL: f64 = 0.015;
    const L_CYL: f64 = 0.03;

    #[test]
    fn test_cyl_tm010_frequency() {
        let chi01 = TM_ZEROS[0][0];
        let expected = C_LIGHT * chi01 / (2.0 * PI * R_CYL);
        let mode = CylindricalCavityMode::new(R_CYL, L_CYL, 0, 1, 0, ModeType::TM).unwrap();
        assert!((mode.resonant_frequency() - expected).abs() / expected < 0.001);
    }

    #[test]
    fn test_cyl_tm_hz_zero() {
        let mode = CylindricalCavityMode::new(R_CYL, L_CYL, 0, 1, 0, ModeType::TM).unwrap();
        let f = mode.fields_rz(0.005, 0.01, 0.0, 0.5);
        assert_eq!(f.hz, 0.0);
    }

    #[test]
    fn test_cyl_te_ez_zero() {
        let mode = CylindricalCavityMode::new(R_CYL, L_CYL, 1, 1, 1, ModeType::TE).unwrap();
        let f = mode.fields_rz(0.005, 0.01, 0.0, 0.5);
        assert_eq!(f.ez, 0.0);
    }

    #[test]
    fn test_cyl_n_zero_error() {
        assert!(CylindricalCavityMode::new(R_CYL, L_CYL, 0, 0, 0, ModeType::TM).is_err());
    }

    #[test]
    fn test_cyl_modes_sorted() {
        let modes = cylindrical_cavity_modes(R_CYL, L_CYL, 6);
        for i in 1..modes.len() {
            assert!(modes[i].resonant_frequency() >= modes[i-1].resonant_frequency());
        }
    }
}
