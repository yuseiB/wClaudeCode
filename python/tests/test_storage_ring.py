"""Tests for storage_ring Monte Carlo tracker."""

import numpy as np
import pytest

from mathphys.storage_ring import BeamParams, RingParams, StorageRing, rms_emittance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def linear_ring(aperture=np.inf) -> RingParams:
    return RingParams(
        tune=0.28,
        beta=10.0,
        alpha=0.0,
        chromaticity=0.0,
        sextupole_strength=0.0,
        aperture=aperture,
    )


def small_beam(eps=1e-6, sigma_delta=0.0, n=500) -> BeamParams:
    return BeamParams(emittance=eps, momentum_spread=sigma_delta, n_particles=n, seed=0)


# ---------------------------------------------------------------------------
# Linear optics
# ---------------------------------------------------------------------------


class TestLinearOptics:
    def test_emittance_conserved(self):
        """RMS emittance must stay constant to <1 % in pure linear optics."""
        tracker = StorageRing(linear_ring(), small_beam(sigma_delta=0.0))
        res = tracker.track(n_turns=500)
        eps0 = res.emittance[0]
        assert np.all(np.abs(res.emittance - eps0) / eps0 < 0.01)

    def test_all_particles_survive(self):
        """Without an aperture cut, no particle should be lost."""
        tracker = StorageRing(linear_ring(), small_beam())
        res = tracker.track(n_turns=200)
        assert np.all(res.survival == 1.0)

    def test_courant_snyder_invariant(self):
        """Single-particle Courant-Snyder invariant J must be machine-exact."""
        ring = linear_ring()
        beam = BeamParams(emittance=1e-6, momentum_spread=0.0, n_particles=1, seed=7)
        tracker = StorageRing(ring, beam)
        x0, xp0 = tracker._x0[0], tracker._xp0[0]
        J0 = ring.gamma * x0**2 + 2 * ring.alpha * x0 * xp0 + ring.beta * xp0**2

        res = tracker.track(n_turns=1000, sample_turns=[0, 1000])
        xf, xpf = res.snapshots[1000]
        Jf = ring.gamma * xf[0] ** 2 + 2 * ring.alpha * xf[0] * xpf[0] + ring.beta * xpf[0] ** 2
        assert abs(Jf - J0) / J0 < 1e-10, f"J: {J0:.6e} → {Jf:.6e}"

    def test_sigma_x_mean_stable(self):
        """Mean σ_x must not drift over turns (linear optics, no damping).

        The instantaneous σ_x oscillates with the betatron phase, so we
        compare the first-half mean to the second-half mean; they must agree
        to within 1 %.
        """
        tracker = StorageRing(linear_ring(), small_beam(sigma_delta=0.0))
        res = tracker.track(n_turns=300)
        half = len(res.sigma_x) // 2
        mean_first = np.mean(res.sigma_x[:half])
        mean_second = np.mean(res.sigma_x[half:])
        assert abs(mean_second - mean_first) / mean_first < 0.01


# ---------------------------------------------------------------------------
# Aperture and particle loss
# ---------------------------------------------------------------------------


class TestAperture:
    def test_particles_lost_beyond_aperture(self):
        """Large emittance + tight aperture must lose particles."""
        ring = linear_ring(aperture=0.005)  # 5 mm — very tight
        beam = small_beam(eps=1e-4, n=1000)  # large initial spread
        res = StorageRing(ring, beam).track(n_turns=100)
        assert res.survival[-1] < 1.0

    def test_survival_monotonically_decreasing(self):
        """Once lost, a particle can never be recovered."""
        ring = linear_ring(aperture=0.01)
        beam = small_beam(eps=5e-5, n=500)
        res = StorageRing(ring, beam).track(n_turns=200)
        assert np.all(np.diff(res.survival) <= 1e-12)

    def test_no_loss_within_aperture(self):
        """Tiny emittance beam well inside aperture: zero losses."""
        ring = linear_ring(aperture=1.0)  # 1 m — enormous
        beam = small_beam(eps=1e-8, n=200)
        res = StorageRing(ring, beam).track(n_turns=100)
        assert res.survival[-1] == 1.0


# ---------------------------------------------------------------------------
# Snapshot API
# ---------------------------------------------------------------------------


class TestSnapshots:
    def test_default_snapshot_turns(self):
        """Default sample_turns must include turn 0 and turn n_turns."""
        res = StorageRing(linear_ring(), small_beam()).track(n_turns=200)
        assert 0 in res.snapshots
        assert 200 in res.snapshots

    def test_custom_snapshot_turns(self):
        res = StorageRing(linear_ring(), small_beam()).track(
            n_turns=100, sample_turns=[0, 50, 100]
        )
        assert set(res.snapshots.keys()) == {0, 50, 100}

    def test_snapshot_shape(self):
        """Each snapshot must be an (x, x') pair with n_alive entries."""
        beam = small_beam(n=300)
        res = StorageRing(linear_ring(), beam).track(n_turns=10, sample_turns=[10])
        x, xp = res.snapshots[10]
        assert x.shape == xp.shape
        assert len(x) == 300  # all alive (no aperture cut)


# ---------------------------------------------------------------------------
# rms_emittance helper
# ---------------------------------------------------------------------------


class TestRmsEmittance:
    def test_circle_in_phase_space(self):
        """Uniform ring of radius R in phase space → ε ≈ R²/2."""
        rng = np.random.default_rng(0)
        theta = rng.uniform(0, 2 * np.pi, 100_000)
        R = 1e-3
        x = R * np.cos(theta)
        xp = R * np.sin(theta)
        eps = rms_emittance(x, xp)
        assert abs(eps - R**2 / 2) / (R**2 / 2) < 0.01

    def test_returns_zero_for_single_particle(self):
        assert rms_emittance(np.array([1e-3]), np.array([0.0])) == 0.0

    def test_nonnegative(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(200)
        xp = rng.standard_normal(200)
        assert rms_emittance(x, xp) >= 0.0
