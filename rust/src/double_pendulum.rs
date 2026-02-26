//! Exact nonlinear double pendulum (Lagrangian formulation, RK4 integrator).
//!
//! State vector: \[θ₁, ω₁, θ₂, ω₂\].
//! Angles are measured from the downward vertical.

/// Time-series output of a double-pendulum simulation.
pub struct SimResult {
    pub t:      Vec<f64>,
    pub theta1: Vec<f64>,
    pub omega1: Vec<f64>,
    pub theta2: Vec<f64>,
    pub omega2: Vec<f64>,
    /// Cartesian x-position of bob-2.
    pub x2:     Vec<f64>,
    /// Cartesian y-position of bob-2.
    pub y2:     Vec<f64>,
    /// Total mechanical energy E = T + V.
    pub energy: Vec<f64>,
}

/// Double pendulum with configurable masses, rod lengths, and gravity.
pub struct DoublePendulum {
    pub m1: f64,
    pub m2: f64,
    pub l1: f64,
    pub l2: f64,
    pub g:  f64,
}

impl DoublePendulum {
    pub fn new(m1: f64, m2: f64, l1: f64, l2: f64, g: f64) -> Self {
        Self { m1, m2, l1, l2, g }
    }

    /// RHS of the first-order ODE system.
    fn eom(&self, s: [f64; 4]) -> [f64; 4] {
        let (th1, w1, th2, w2) = (s[0], s[1], s[2], s[3]);
        let dlt = th1 - th2;
        let d   = 2.0 * self.m1 + self.m2 - self.m2 * (2.0 * dlt).cos();

        let a1 = (
            -self.g * (2.0 * self.m1 + self.m2) * th1.sin()
            - self.m2 * self.g * (th1 - 2.0 * th2).sin()
            - 2.0 * dlt.sin() * self.m2 * (w2 * w2 * self.l2 + w1 * w1 * self.l1 * dlt.cos())
        ) / (self.l1 * d);

        let a2 = (
            2.0 * dlt.sin() * (
                w1 * w1 * self.l1 * (self.m1 + self.m2)
                + self.g * (self.m1 + self.m2) * th1.cos()
                + w2 * w2 * self.l2 * self.m2 * dlt.cos()
            )
        ) / (self.l2 * d);

        [w1, a1, w2, a2]
    }

    /// Total mechanical energy E = T + V.
    pub fn energy(&self, s: [f64; 4]) -> f64 {
        let (th1, w1, th2, w2) = (s[0], s[1], s[2], s[3]);
        let dlt = th1 - th2;
        let kinetic =
            0.5 * self.m1 * self.l1 * self.l1 * w1 * w1
            + 0.5 * self.m2 * (
                self.l1 * self.l1 * w1 * w1
                + self.l2 * self.l2 * w2 * w2
                + 2.0 * self.l1 * self.l2 * w1 * w2 * dlt.cos()
            );
        let potential = -self.g * (
            (self.m1 + self.m2) * self.l1 * th1.cos()
            + self.m2 * self.l2 * th2.cos()
        );
        kinetic + potential
    }

    /// Fixed-step RK4 integration.
    pub fn simulate(
        &self,
        theta1_0: f64,
        theta2_0: f64,
        omega1_0: f64,
        omega2_0: f64,
        t_end: f64,
        dt: f64,
    ) -> SimResult {
        let n = (t_end / dt) as usize + 1;
        let mut res = SimResult {
            t:      Vec::with_capacity(n),
            theta1: Vec::with_capacity(n),
            omega1: Vec::with_capacity(n),
            theta2: Vec::with_capacity(n),
            omega2: Vec::with_capacity(n),
            x2:     Vec::with_capacity(n),
            y2:     Vec::with_capacity(n),
            energy: Vec::with_capacity(n),
        };

        let mut y = [theta1_0, omega1_0, theta2_0, omega2_0];
        let mut t = 0.0_f64;

        for _ in 0..n {
            res.t.push(t);
            res.theta1.push(y[0]);
            res.omega1.push(y[1]);
            res.theta2.push(y[2]);
            res.omega2.push(y[3]);
            res.x2.push(self.l1 * y[0].sin() + self.l2 * y[2].sin());
            res.y2.push(-self.l1 * y[0].cos() - self.l2 * y[2].cos());
            res.energy.push(self.energy(y));

            // RK4 step
            let k1 = self.eom(y);
            let k2 = self.eom(add(y, scale(k1, dt / 2.0)));
            let k3 = self.eom(add(y, scale(k2, dt / 2.0)));
            let k4 = self.eom(add(y, scale(k3, dt)));
            y = add(y, scale(add4(k1, k2, k3, k4), dt / 6.0));
            t += dt;
        }
        res
    }
}

// ── small vector helpers ──────────────────────────────────────────────────────

#[inline]
fn add(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]]
}

#[inline]
fn scale(a: [f64; 4], s: f64) -> [f64; 4] {
    [a[0]*s, a[1]*s, a[2]*s, a[3]*s]
}

/// Weighted sum k1 + 2*k2 + 2*k3 + k4.
#[inline]
fn add4(k1: [f64; 4], k2: [f64; 4], k3: [f64; 4], k4: [f64; 4]) -> [f64; 4] {
    [
        k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0],
        k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1],
        k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2],
        k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3],
    ]
}
