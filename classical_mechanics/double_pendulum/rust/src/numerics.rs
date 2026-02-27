//! Basic numerical methods for Mathematical Physics.

/// Trapezoidal-rule integration of `f` over `x`.
///
/// # Panics
/// Panics if `f.len() != x.len()` or `f.len() < 2`.
pub fn integrate_trapezoid(f: &[f64], x: &[f64]) -> f64 {
    assert_eq!(f.len(), x.len(), "f and x must have the same length");
    assert!(f.len() >= 2, "Need at least 2 points");

    f.windows(2)
        .zip(x.windows(2))
        .map(|(fi, xi)| 0.5 * (fi[0] + fi[1]) * (xi[1] - xi[0]))
        .sum()
}

/// Central finite-difference derivative (order 1 or 2).
///
/// # Panics
/// Panics if `order` is not 1 or 2, or if `f.len() < 3`.
pub fn finite_difference(f: &[f64], x: &[f64], order: u32) -> Vec<f64> {
    let n = f.len();
    assert!(n >= 3, "Need at least 3 points");
    assert_eq!(f.len(), x.len());
    assert!(order == 1 || order == 2, "Order must be 1 or 2");

    let mut df = vec![0.0f64; n];

    match order {
        1 => {
            df[0] = (f[1] - f[0]) / (x[1] - x[0]);
            for i in 1..n - 1 {
                df[i] = (f[i + 1] - f[i - 1]) / (x[i + 1] - x[i - 1]);
            }
            df[n - 1] = (f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2]);
        }
        2 => {
            let h0 = x[1] - x[0];
            df[0] = (f[2] - 2.0 * f[1] + f[0]) / (h0 * h0);
            for i in 1..n - 1 {
                let h = (x[i + 1] - x[i - 1]) / 2.0;
                df[i] = (f[i + 1] - 2.0 * f[i] + f[i - 1]) / (h * h);
            }
            df[n - 1] = df[n - 2];
        }
        _ => unreachable!(),
    }
    df
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| start + (end - start) * i as f64 / (n - 1) as f64)
            .collect()
    }

    #[test]
    fn test_integrate_constant() {
        let x = linspace(0.0, 1.0, 1000);
        let f = vec![1.0f64; 1000];
        assert_abs_diff_eq!(integrate_trapezoid(&f, &x), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_integrate_x_squared() {
        let x = linspace(0.0, 1.0, 10_000);
        let f: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        assert_abs_diff_eq!(integrate_trapezoid(&f, &x), 1.0 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_finite_difference_linear() {
        let x = linspace(0.0, 1.0, 100);
        let slope = 3.7_f64;
        let f: Vec<f64> = x.iter().map(|&xi| slope * xi + 1.5).collect();
        let df = finite_difference(&f, &x, 1);
        // Interior points should match slope
        for &d in &df[1..df.len() - 1] {
            assert_abs_diff_eq!(d, slope, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_finite_difference_second_order() {
        let x = linspace(0.0, 1.0, 1000);
        let f: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let d2f = finite_difference(&f, &x, 2);
        for &d in &d2f[2..d2f.len() - 2] {
            assert_abs_diff_eq!(d, 2.0, epsilon = 1e-3);
        }
    }
}
