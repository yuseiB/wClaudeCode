//! 2D Ising Model — Temperature Sweep
//!
//! Generates ising2d_sweep.csv with columns: T,E_mean,M_mean,Cv,chi
//!
//! Usage:
//!   cd statistical_physics/ising_model_2d/rust
//!   cargo run --bin ising_sim --release

use ising2d::{IsingModel2D, T_CRITICAL};
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let n: usize = 32;
    let n_therm: usize = 5_000;
    let n_meas: usize = 10_000;
    let t_min: f64 = 1.0;
    let t_max: f64 = 4.0;
    let n_t: usize = 40;

    println!("2D Ising Model sweep  N={n}  T=[{t_min:.1}, {t_max:.1}]  {n_t} points");
    println!("Onsager T_c = {T_CRITICAL:.4}");

    let temps: Vec<f64> = (0..n_t)
        .map(|k| t_min + k as f64 * (t_max - t_min) / (n_t - 1) as f64)
        .collect();

    let file = File::create("ising2d_sweep.csv").expect("Cannot create CSV");
    let mut w = BufWriter::new(file);
    writeln!(w, "T,E_mean,M_mean,Cv,chi").unwrap();

    for (k, &t) in temps.iter().enumerate() {
        let mut model = IsingModel2D::new(n, 1.0, k as u64);
        if t < T_CRITICAL {
            model.lattice.iter_mut().for_each(|s| *s = 1); // start ordered
        }
        let r = model.simulate(t, n_therm, n_meas);
        writeln!(w, "{:.4},{:.6},{:.6},{:.6},{:.6}",
                 t, r.e_mean, r.m_mean, r.cv, r.chi).unwrap();

        if (k + 1) % 10 == 0 {
            println!("  {}/{n_t}  T={t:.2}  |M|={:.3}  Cv={:.3}",
                     k + 1, r.m_mean, r.cv);
        }
    }
    println!("\nSaved → ising2d_sweep.csv");
}
