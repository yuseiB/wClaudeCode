"""Unit tests for the 2D Ising model."""

import numpy as np
import pytest

from mathphys.ising_model import IsingModel2D, T_CRITICAL


# ---------------------------------------------------------------------------
# Energy calculations
# ---------------------------------------------------------------------------

def test_energy_all_aligned_positive():
    """All spins +1: E = -2J N² (each of N² sites has 2 unique bonds)."""
    n = 8
    model = IsingModel2D(n=n, j=1.0, seed=0)
    model.lattice[:] = 1
    expected = -2.0 * n ** 2
    assert np.isclose(model.energy(), expected), f"Expected {expected}, got {model.energy()}"


def test_energy_all_aligned_negative():
    """All spins -1: energy same as all +1 (product s_i s_j = (+1)(+1))."""
    n = 8
    model = IsingModel2D(n=n, j=1.0, seed=0)
    model.lattice[:] = -1
    expected = -2.0 * n ** 2
    assert np.isclose(model.energy(), expected)


def test_energy_checkerboard():
    """Checkerboard pattern (antiferromagnet): E = +2J N²."""
    n = 8
    model = IsingModel2D(n=n, j=1.0, seed=0)
    for i in range(n):
        for j in range(n):
            model.lattice[i, j] = 1 if (i + j) % 2 == 0 else -1
    expected = +2.0 * n ** 2
    assert np.isclose(model.energy(), expected), f"Expected {expected}, got {model.energy()}"


def test_energy_coupling_scaling():
    """Energy scales linearly with coupling constant J."""
    n = 6
    m1 = IsingModel2D(n=n, j=1.0, seed=7)
    m2 = IsingModel2D(n=n, j=2.0, seed=7)
    # Same initial random lattice (same seed)
    assert np.isclose(m2.energy(), 2.0 * m1.energy())


# ---------------------------------------------------------------------------
# Magnetisation
# ---------------------------------------------------------------------------

def test_magnetization_all_up():
    """All +1: M = N²."""
    n = 10
    model = IsingModel2D(n=n, seed=0)
    model.lattice[:] = 1
    assert model.magnetization() == n ** 2


def test_magnetization_all_down():
    """All -1: M = -N²."""
    n = 10
    model = IsingModel2D(n=n, seed=0)
    model.lattice[:] = -1
    assert model.magnetization() == -(n ** 2)


# ---------------------------------------------------------------------------
# Metropolis dynamics
# ---------------------------------------------------------------------------

def test_metropolis_preserves_lattice_shape():
    """metropolis_step must not change the lattice shape."""
    model = IsingModel2D(n=16, seed=0)
    model.metropolis_step(temperature=2.0)
    assert model.lattice.shape == (16, 16)


def test_metropolis_high_temperature_disordering():
    """At very high T, starting from all-up, |M|/N² should decrease toward 0."""
    n = 32
    model = IsingModel2D(n=n, seed=1)
    model.lattice[:] = 1
    initial_m = abs(model.magnetization()) / n ** 2
    for _ in range(500):
        model.metropolis_step(temperature=100.0)
    final_m = abs(model.magnetization()) / n ** 2
    assert final_m < initial_m, "High-T dynamics should disorder the lattice"


def test_metropolis_low_temperature_ordering():
    """At very low T, random lattice should order (|M|/N² → 1)."""
    n = 16
    model = IsingModel2D(n=n, seed=2)
    for _ in range(2000):
        model.metropolis_step(temperature=0.5)
    m = abs(model.magnetization()) / n ** 2
    assert m > 0.9, f"Low-T lattice should be nearly ordered, got |M|/N²={m:.3f}"


# ---------------------------------------------------------------------------
# Phase transition (statistical test)
# ---------------------------------------------------------------------------

def test_simulate_magnetization_below_tc():
    """Below T_c, equilibrium |M|/N² should be significantly > 0."""
    model = IsingModel2D(n=24, seed=3)
    result = model.simulate(temperature=1.5, n_therm=3000, n_measure=5000)
    assert result["M_mean"] > 0.5, (
        f"Below T_c, expect ordered phase; got M_mean={result['M_mean']:.3f}"
    )


def test_simulate_magnetization_above_tc():
    """Above T_c, equilibrium |M|/N² should be close to 0."""
    model = IsingModel2D(n=24, seed=4)
    result = model.simulate(temperature=3.5, n_therm=3000, n_measure=5000)
    assert result["M_mean"] < 0.3, (
        f"Above T_c, expect disordered phase; got M_mean={result['M_mean']:.3f}"
    )


def test_tc_value():
    """Onsager's T_c should equal 2/ln(1+√2) to 4 decimal places."""
    assert abs(T_CRITICAL - 2.2692) < 1e-4
