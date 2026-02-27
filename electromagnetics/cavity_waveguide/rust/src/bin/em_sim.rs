/*!
 * em_sim.rs — 電磁気キャビティ・導波管シミュレーション (Rust バイナリ)
 *
 * 出力:
 *   1. 直方体キャビティ: 最低次 8 モードの共振周波数
 *   2. 円筒型キャビティ: 最低次 8 モードの共振周波数
 *   3. 直方体導波管: 最低次 6 モードのカットオフ周波数 + β/β₀
 *   4. 円形導波管: 最低次 6 モードのカットオフ周波数 + β/β₀
 *
 * 使用法:
 *   cargo run --bin em_sim
 */

use em_cavity_waveguide::{
    cavity::{rectangular_cavity_modes, cylindrical_cavity_modes, ModeType},
    waveguide::{rectangular_waveguide_modes, circular_waveguide_modes, RectangularWaveguideMode},
};

fn main() {
    // ── 1. 直方体キャビティ ──────────────────────────────────────────────────
    println!("=== Rectangular Cavity Resonant Modes ===");
    println!("  Dimensions: a=4 cm, b=2 cm, d=3 cm");
    println!("{:<12} {:>15}", "Mode", "f_res (GHz)");
    println!("{}", "-".repeat(30));

    let a = 0.04_f64;
    let b = 0.02_f64;
    let d = 0.03_f64;
    let rect_modes = rectangular_cavity_modes(a, b, d, 8);
    for mo in &rect_modes {
        println!("{:<12} {:>15.4}", mo.label(), mo.resonant_frequency() * 1e-9);
    }

    // ── 2. 円筒型キャビティ ──────────────────────────────────────────────────
    println!("\n=== Cylindrical Cavity Resonant Modes ===");
    println!("  Dimensions: R=1.5 cm, L=3 cm");
    println!("{:<12} {:>15}", "Mode", "f_res (GHz)");
    println!("{}", "-".repeat(30));

    let r_cyl = 0.015_f64;
    let l_cyl = 0.03_f64;
    let cyl_modes = cylindrical_cavity_modes(r_cyl, l_cyl, 8);
    for mo in &cyl_modes {
        println!("{:<12} {:>15.4}", mo.label(), mo.resonant_frequency() * 1e-9);
    }

    // ── 3. 直方体導波管 ──────────────────────────────────────────────────────
    println!("\n=== Rectangular Waveguide Modes ===");
    println!("  Dimensions: a=4 cm, b=2 cm");
    println!("  Operating frequency: 1.5 × fc");
    println!("{:<10} {:>14} {:>14} {:>14}", "Mode", "fc (GHz)", "f_op (GHz)", "β (rad/m)");
    println!("{}", "-".repeat(55));

    let wg_rect_modes = rectangular_waveguide_modes(a, b, 6);
    for mo in &wg_rect_modes {
        let fc  = mo.cutoff_frequency();
        let fop = fc * 1.5;
        let beta = mo.propagation_constant(fop);
        println!("{:<10} {:>14.4} {:>14.4} {:>14.4}",
                 mo.label(), fc * 1e-9, fop * 1e-9, beta);
    }

    // ── 4. 円形導波管 ────────────────────────────────────────────────────────
    println!("\n=== Circular Waveguide Modes ===");
    println!("  Radius: R=1.5 cm");
    println!("  Operating frequency: 1.5 × fc");
    println!("{:<10} {:>14} {:>14} {:>14}", "Mode", "fc (GHz)", "f_op (GHz)", "β (rad/m)");
    println!("{}", "-".repeat(55));

    let r_wg = 0.015_f64;
    let wg_circ_modes = circular_waveguide_modes(r_wg, 6);
    for mo in &wg_circ_modes {
        let fc  = mo.cutoff_frequency();
        let fop = fc * 1.5;
        let beta = mo.propagation_constant(fop);
        println!("{:<10} {:>14.4} {:>14.4} {:>14.4}",
                 mo.label(), fc * 1e-9, fop * 1e-9, beta);
    }

    // ── 5. 分散関係のスニペット ──────────────────────────────────────────────
    println!("\n=== TE_10 Dispersion: β vs f (a=4 cm) ===");
    println!("{:>10} {:>14}", "f (GHz)", "β (rad/m)");
    println!("{}", "-".repeat(27));

    let te10 = RectangularWaveguideMode::new(a, b, 1, 0, ModeType::TE).unwrap();
    let fc10 = te10.cutoff_frequency();
    for i in 0..=10 {
        let f = fc10 * (0.5 + 1.5 * i as f64 / 10.0);
        let beta = te10.propagation_constant(f);
        println!("{:>10.3} {:>14.3}", f * 1e-9, beta);
    }
}
