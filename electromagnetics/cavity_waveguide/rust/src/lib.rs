/*!
 * em_cavity_waveguide — 電磁気キャビティ・導波管解析ライブラリ (Rust)
 *
 * モジュール構成:
 *   constants   — 物理定数 (SI 単位)
 *   cavity      — 直方体・円筒型キャビティ共振モード
 *   waveguide   — 直方体・円形導波管伝搬モード
 *
 * 物理的な背景:
 *   キャビティ: PEC 壁で囲まれた空洞内の EM 定常波
 *   導波管:     PEC 管内を +z 方向へ伝搬する EM 進行波
 *
 * 参考文献:
 *   Pozar, "Microwave Engineering", 4th ed., ch. 3, 6
 */

pub mod constants;
pub mod cavity;
pub mod waveguide;

/// EMField: 6 成分の電磁場をまとめた構造体
///
/// 直交座標: (Ex, Ey, Ez, Hx, Hy, Hz)
/// 曲線座標: (Eρ, Eφ, Ez, Hρ, Hφ, Hz) など (文脈依存)
#[derive(Debug, Clone, Copy, Default)]
pub struct EMPoint {
    pub ex: f64, pub ey: f64, pub ez: f64,  // 電場 (V/m)
    pub hx: f64, pub hy: f64, pub hz: f64,  // 磁場 (A/m)
}

impl EMPoint {
    /// 電場強度 |E| (V/m)
    pub fn e_mag(&self) -> f64 {
        (self.ex * self.ex + self.ey * self.ey + self.ez * self.ez).sqrt()
    }

    /// 磁場強度 |H| (A/m)
    pub fn h_mag(&self) -> f64 {
        (self.hx * self.hx + self.hy * self.hy + self.hz * self.hz).sqrt()
    }
}
