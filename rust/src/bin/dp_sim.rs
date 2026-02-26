//! Runs eight double-pendulum scenarios and writes CSV files to the current
//! working directory.
//!
//! Run from rust/examples/:
//!   cargo build --bin dp_sim --release   (from rust/)
//!   ../target/release/dp_sim

use std::fs::File;
use std::io::{BufWriter, Write};

use mathphys::double_pendulum::{DoublePendulum, SimResult};

const PI: f64 = std::f64::consts::PI;

fn write_csv(path: &str, r: &SimResult) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "t,theta1,omega1,theta2,omega2,x2,y2,energy")?;
    for i in 0..r.t.len() {
        writeln!(
            w,
            "{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10}",
            r.t[i], r.theta1[i], r.omega1[i],
            r.theta2[i], r.omega2[i],
            r.x2[i], r.y2[i], r.energy[i]
        )?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let dp = DoublePendulum::new(1.0, 1.0, 1.0, 1.0, 9.81);

    // ── section 1: three dynamical regimes ───────────────────────────────────
    write_csv("case_nearlinear.csv",
        &dp.simulate(10.0*PI/180.0, 10.0*PI/180.0, 0.0, 0.0, 30.0, 0.005))?;
    write_csv("case_intermediate.csv",
        &dp.simulate(90.0*PI/180.0, 0.0, 0.0, 0.0, 30.0, 0.005))?;
    write_csv("case_chaotic.csv",
        &dp.simulate(120.0*PI/180.0, -30.0*PI/180.0, 0.0, 0.0, 30.0, 0.005))?;

    // ── section 2: sensitivity to initial conditions ─────────────────────────
    let delta = 0.001 * PI / 180.0;
    write_csv("sensitivity_a.csv",
        &dp.simulate(120.0*PI/180.0,         -30.0*PI/180.0, 0.0, 0.0, 25.0, 0.005))?;
    write_csv("sensitivity_b.csv",
        &dp.simulate(120.0*PI/180.0 + delta, -30.0*PI/180.0, 0.0, 0.0, 25.0, 0.005))?;

    // ── section 3: mass ratios ────────────────────────────────────────────────
    for (ratio, fname) in &[
        (0.25_f64, "mass_ratio_0.25.csv"),
        (1.00_f64, "mass_ratio_1.00.csv"),
        (4.00_f64, "mass_ratio_4.00.csv"),
    ] {
        let dp_m = DoublePendulum::new(1.0, *ratio, 1.0, 1.0, 9.81);
        write_csv(fname, &dp_m.simulate(90.0*PI/180.0, 0.0, 0.0, 0.0, 30.0, 0.005))?;
    }

    println!("Wrote 8 CSV files.");
    Ok(())
}
