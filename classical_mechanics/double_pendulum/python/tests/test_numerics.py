"""Tests for mathphys.numerics."""

import numpy as np
import pytest

from mathphys.numerics import integrate_trapezoid, finite_difference


def test_integrate_trapezoid_constant():
    """Integral of 1 over [0, 1] should be 1."""
    x = np.linspace(0, 1, 1000)
    f = np.ones_like(x)
    result = integrate_trapezoid(f, x)
    assert abs(result - 1.0) < 1e-6


def test_integrate_trapezoid_sin():
    """Integral of sin(x) over [0, pi] should be 2."""
    x = np.linspace(0, np.pi, 10_000)
    f = np.sin(x)
    result = integrate_trapezoid(f, x)
    assert abs(result - 2.0) < 1e-5


def test_integrate_trapezoid_x_squared():
    """Integral of x^2 over [0, 1] should be 1/3."""
    x = np.linspace(0, 1, 10_000)
    result = integrate_trapezoid(x**2, x)
    assert abs(result - 1.0 / 3.0) < 1e-5


def test_finite_difference_linear():
    """Derivative of a linear function should be its slope."""
    x = np.linspace(0, 1, 100)
    slope = 3.7
    f = slope * x + 1.5
    df = finite_difference(f, x, order=1)
    # Interior points should match slope exactly for linear functions
    assert np.allclose(df[1:-1], slope, atol=1e-10)


def test_finite_difference_quadratic_second_order():
    """Second derivative of x^2 should be 2."""
    x = np.linspace(0, 1, 1000)
    f = x**2
    d2f = finite_difference(f, x, order=2)
    assert np.allclose(d2f[2:-2], 2.0, atol=1e-4)


def test_finite_difference_invalid_order():
    x = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="Unsupported derivative order"):
        finite_difference(x, x, order=3)
