//! 2D Ising Model — Metropolis-Hastings Monte Carlo
//!
//! Square N×N lattice with periodic boundary conditions.
//! Hamiltonian:  H = -J Σ_{<i,j>} s_i s_j
//! Spins:        s_i ∈ {-1, +1}
//!
//! Onsager's exact critical temperature (J = k_B = 1):
//!   T_c = 2 / ln(1 + √2) ≈ 2.2692

use rand::prelude::*;

/// Onsager's exact critical temperature for J = k_B = 1.
pub const T_CRITICAL: f64 = 2.0 / 0.8813_7358_7027_5155; // 2/ln(1+√2)

/// Observable snapshot from a single simulation run.
#[derive(Debug, Clone)]
pub struct SimResult {
    pub t: f64,
    pub e_mean: f64,
    pub e2_mean: f64,
    pub m_mean: f64,
    pub m2_mean: f64,
    pub cv: f64,
    pub chi: f64,
}

/// 2D Ising model on an N×N square lattice (periodic boundary conditions).
pub struct IsingModel2D {
    pub n: usize,
    pub j: f64,
    /// Row-major flat array of ±1 spins: index = row * n + col
    pub lattice: Vec<i8>,
    rng: StdRng,
}

impl IsingModel2D {
    /// Create a new model with random ±1 initialisation.
    pub fn new(n: usize, j: f64, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let lattice: Vec<i8> = (0..n * n)
            .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
            .collect();
        Self { n, j, lattice, rng }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    #[inline]
    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.n + col
    }

    #[inline]
    fn spin(&self, row: usize, col: usize) -> i8 {
        self.lattice[self.idx(row, col)]
    }

    // ------------------------------------------------------------------
    // Observables
    // ------------------------------------------------------------------

    /// Total energy E = -J Σ_{<i,j>} s_i s_j
    pub fn energy(&self) -> f64 {
        let n = self.n;
        let mut e = 0i32;
        for i in 0..n {
            for jj in 0..n {
                let s = self.spin(i, jj) as i32;
                let right = self.spin(i, (jj + 1) % n) as i32;
                let down  = self.spin((i + 1) % n, jj) as i32;
                e += s * (right + down);
            }
        }
        -self.j * e as f64
    }

    /// Total magnetisation M = Σ_i s_i
    pub fn magnetization(&self) -> f64 {
        self.lattice.iter().map(|&s| s as f64).sum()
    }

    // ------------------------------------------------------------------
    // Monte Carlo
    // ------------------------------------------------------------------

    /// One full MC sweep: N² single-spin Metropolis-Hastings attempts.
    pub fn metropolis_step(&mut self, temperature: f64) {
        let n = self.n;
        let n2 = n * n;
        let beta = 1.0 / temperature;

        // Pre-compute acceptance probs for ΔE/J ∈ {4, 8}
        let exp4 = (-beta * self.j * 4.0_f64).exp();
        let exp8 = (-beta * self.j * 8.0_f64).exp();

        for _ in 0..n2 {
            let i  = self.rng.gen_range(0..n);
            let jj = self.rng.gen_range(0..n);
            let s  = self.spin(i, jj) as i32;

            let nb = self.spin((i + n - 1) % n, jj) as i32
                   + self.spin((i + 1)     % n, jj) as i32
                   + self.spin(i, (jj + n - 1) % n) as i32
                   + self.spin(i, (jj + 1)     % n) as i32;

            let de_over_j = 2 * s * nb;

            let accept = if de_over_j <= 0 {
                true
            } else {
                let prob = match de_over_j {
                    4 => exp4,
                    8 => exp8,
                    _ => 0.0,
                };
                self.rng.gen::<f64>() < prob
            };

            if accept {
                let idx = self.idx(i, jj);
                self.lattice[idx] = -self.lattice[idx];
            }
        }
    }

    /// Thermalise then measure observables.
    pub fn simulate(
        &mut self,
        temperature: f64,
        n_therm: usize,
        n_measure: usize,
    ) -> SimResult {
        let n2 = (self.n * self.n) as f64;

        for _ in 0..n_therm {
            self.metropolis_step(temperature);
        }

        let mut e_sum  = 0.0_f64;
        let mut e2_sum = 0.0_f64;
        let mut m_sum  = 0.0_f64;
        let mut m2_sum = 0.0_f64;

        for _ in 0..n_measure {
            self.metropolis_step(temperature);
            let e = self.energy() / n2;
            let m = self.magnetization().abs() / n2;
            e_sum  += e;
            e2_sum += e * e;
            m_sum  += m;
            m2_sum += m * m;
        }

        let nm = n_measure as f64;
        let e_mean  = e_sum  / nm;
        let e2_mean = e2_sum / nm;
        let m_mean  = m_sum  / nm;
        let m2_mean = m2_sum / nm;

        SimResult {
            t:       temperature,
            e_mean,
            e2_mean,
            m_mean,
            m2_mean,
            cv:  (e2_mean - e_mean * e_mean) / (temperature * temperature),
            chi: (m2_mean - m_mean * m_mean) / temperature,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn all_up(n: usize) -> IsingModel2D {
        let mut model = IsingModel2D::new(n, 1.0, 0);
        model.lattice.iter_mut().for_each(|s| *s = 1);
        model
    }

    fn all_down(n: usize) -> IsingModel2D {
        let mut model = IsingModel2D::new(n, 1.0, 0);
        model.lattice.iter_mut().for_each(|s| *s = -1);
        model
    }

    fn checkerboard(n: usize) -> IsingModel2D {
        let mut model = IsingModel2D::new(n, 1.0, 0);
        for i in 0..n {
            for j in 0..n {
                model.lattice[i * n + j] = if (i + j) % 2 == 0 { 1 } else { -1 };
            }
        }
        model
    }

    #[test]
    fn test_energy_all_up() {
        let n = 8;
        let model = all_up(n);
        assert_abs_diff_eq!(model.energy(), -2.0 * (n * n) as f64, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_all_down() {
        let n = 8;
        let model = all_down(n);
        assert_abs_diff_eq!(model.energy(), -2.0 * (n * n) as f64, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_checkerboard() {
        let n = 8;
        let model = checkerboard(n);
        assert_abs_diff_eq!(model.energy(), 2.0 * (n * n) as f64, epsilon = 1e-10);
    }

    #[test]
    fn test_magnetization_all_up() {
        let n = 10;
        let model = all_up(n);
        assert_abs_diff_eq!(model.magnetization(), (n * n) as f64, epsilon = 1e-10);
    }

    #[test]
    fn test_magnetization_all_down() {
        let n = 10;
        let model = all_down(n);
        assert_abs_diff_eq!(model.magnetization(), -((n * n) as f64), epsilon = 1e-10);
    }

    #[test]
    fn test_low_temperature_ordering() {
        let mut model = IsingModel2D::new(16, 1.0, 2);
        for _ in 0..2000 {
            model.metropolis_step(0.5);
        }
        let m = model.magnetization().abs() / (16.0 * 16.0);
        assert!(m > 0.9, "Low-T lattice should be ordered, got |M|/N²={m:.3}");
    }

    #[test]
    fn test_simulate_below_tc() {
        let mut model = IsingModel2D::new(24, 1.0, 3);
        model.lattice.iter_mut().for_each(|s| *s = 1);
        let r = model.simulate(1.5, 3000, 5000);
        assert!(r.m_mean > 0.5, "Below Tc expect |M|/N²>0.5, got {:.3}", r.m_mean);
    }

    #[test]
    fn test_simulate_above_tc() {
        let mut model = IsingModel2D::new(24, 1.0, 4);
        let r = model.simulate(3.5, 3000, 5000);
        assert!(r.m_mean < 0.3, "Above Tc expect |M|/N²<0.3, got {:.3}", r.m_mean);
    }

    #[test]
    fn test_tc_value() {
        assert_abs_diff_eq!(T_CRITICAL, 2.2692, epsilon = 1e-4);
    }
}
