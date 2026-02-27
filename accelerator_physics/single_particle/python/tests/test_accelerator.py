"""Unit tests for single-particle accelerator dynamics (mathphys.accelerator).

Test strategy
-------------
1. Transfer matrix correctness — compare against analytical formulas.
2. Symplecticity          — det(M_4x4) = 1 for every linear element.
3. Twiss parameters       — Courant-Snyder invariants, propagation continuity.
4. Phase-space invariants — emittance preserved under linear tracking.
5. Longitudinal dynamics  — RF cavity phase-space rotation.
6. Factory functions      — FODO tune, ring stability.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mathphys.accelerator import (
    Beam,
    Dipole,
    Drift,
    Lattice,
    Marker,
    Particle,
    Quadrupole,
    RFCavity,
    Sextupole,
    make_fodo,
    make_ring,
    track,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_symplectic_4x4(M: np.ndarray, tol: float = 1e-10) -> bool:
    """Check that the transverse 4×4 block of M is symplectic (det = 1)."""
    M4 = M[:4, :4]
    return abs(np.linalg.det(M4) - 1.0) < tol


# ── Particle & Beam ───────────────────────────────────────────────────────────

class TestParticle:
    def test_default_state(self):
        p = Particle()
        assert np.allclose(p.state, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_round_trip(self):
        p = Particle(x=1e-3, px=2e-4, y=-3e-4, py=5e-5, delta=1e-3, l=-2e-3)
        p2 = Particle.from_state(p.state)
        assert np.allclose(p.state, p2.state)


class TestBeam:
    def test_gaussian_shape(self):
        beam = Beam.gaussian(n=1000, emittance_x=1e-6, beta_x=5.0, alpha_x=0.0,
                              emittance_y=1e-6, beta_y=5.0, alpha_y=0.0)
        assert beam.particles.shape == (1000, 6)

    def test_emittance_accuracy(self):
        """Measured emittance should match input within 5% for N=10000."""
        eps_in = 1e-6  # 1 μm·rad
        beam = Beam.gaussian(n=10_000, emittance_x=eps_in, beta_x=10.0, alpha_x=0.0,
                              emittance_y=eps_in, beta_y=10.0, alpha_y=0.0,
                              rng=np.random.default_rng(0))
        eps_x = beam.emittance("x")
        assert abs(eps_x / eps_in - 1.0) < 0.05

    def test_centroid_near_zero(self):
        # Statistical centroid fluctuation ≈ σ_x/√N = √(ε·β)/√N
        # ≈ √(1e-6·5)/√5000 ≈ 3e-5 m  → tolerance 1e-4 is safe
        beam = Beam.gaussian(n=5_000, emittance_x=1e-6, beta_x=5.0, alpha_x=0.0,
                              emittance_y=1e-6, beta_y=5.0, alpha_y=0.0,
                              rng=np.random.default_rng(7))
        c = beam.centroid()
        assert np.all(np.abs(c[:4]) < 1e-4)


# ── Drift ─────────────────────────────────────────────────────────────────────

class TestDrift:
    L = 2.5  # [m]

    def test_matrix_structure(self):
        M = Drift(self.L).matrix()
        expected = np.eye(6)
        expected[0, 1] = self.L
        expected[2, 3] = self.L
        assert np.allclose(M, expected)

    def test_symplecticity(self):
        assert is_symplectic_4x4(Drift(self.L).matrix())

    def test_track_single_particle(self):
        d = Drift(self.L)
        p = np.array([[1e-3, 2e-4, 0.0, 0.0, 0.0, 0.0]])
        out = d.track(p)
        assert pytest.approx(out[0, 0], rel=1e-9) == 1e-3 + self.L * 2e-4

    def test_chromatic_drift(self):
        """Off-momentum particle sees effective length L/(1+delta)."""
        d = Drift(self.L)
        delta = 1e-2
        p = np.array([[0.0, 1e-3, 0.0, 0.0, delta, 0.0]])
        out = d.track(p)
        L_eff = self.L / (1.0 + delta)
        assert pytest.approx(out[0, 0], rel=1e-9) == L_eff * 1e-3


# ── Quadrupole ────────────────────────────────────────────────────────────────

class TestQuadrupole:
    Lq = 0.5
    k1 = 2.0  # focusing in x

    def test_symplecticity_focusing(self):
        assert is_symplectic_4x4(Quadrupole(self.Lq, self.k1).matrix())

    def test_symplecticity_defocusing(self):
        assert is_symplectic_4x4(Quadrupole(self.Lq, -self.k1).matrix())

    def test_zero_gradient_is_drift(self):
        M_q = Quadrupole(self.Lq, k1=0.0).matrix()
        M_d = Drift(self.Lq).matrix()
        assert np.allclose(M_q, M_d)

    def test_focusing_x_defocusing_y(self):
        """k1 > 0: x-plane has cosine term, y-plane has cosh term."""
        M = Quadrupole(self.Lq, self.k1).matrix()
        phi = math.sqrt(self.k1) * self.Lq
        # x block: should be [[cos, ...], [..., cos]]
        assert pytest.approx(M[0, 0], rel=1e-9) == math.cos(phi)
        # y block: should be [[cosh, ...], [..., cosh]]
        assert pytest.approx(M[2, 2], rel=1e-9) == math.cosh(phi)

    def test_chromatic_scaling(self):
        """k1_eff = k1/(1+delta) → stronger focus for delta < 0."""
        d = 0.01
        phi_ref = math.sqrt(self.k1) * self.Lq
        phi_off = math.sqrt(self.k1 / (1.0 + d)) * self.Lq
        M_ref = Quadrupole(self.Lq, self.k1).matrix(delta=0.0)
        M_off = Quadrupole(self.Lq, self.k1).matrix(delta=d)
        # The x[0,0] element differs
        assert abs(M_off[0, 0] - math.cos(phi_off)) < 1e-12
        assert abs(M_ref[0, 0] - math.cos(phi_ref)) < 1e-12


# ── Dipole ────────────────────────────────────────────────────────────────────

class TestDipole:
    Lb = 2.0
    ang = math.pi / 8  # 22.5°

    def test_symplecticity(self):
        assert is_symplectic_4x4(Dipole(self.Lb, self.ang).matrix())

    def test_dispersion_nonzero(self):
        """After a dipole, a on-momentum off-axis particle develops dispersion."""
        M = Dipole(self.Lb, self.ang).matrix()
        # M[0, 4]: dispersion term (coupling delta → x)
        rho = self.Lb / self.ang
        expected_d = rho * (1.0 - math.cos(self.ang))
        assert pytest.approx(M[0, 4], rel=1e-9) == expected_d

    def test_vertical_drift(self):
        """Vertical plane of a pure dipole is a drift."""
        M = Dipole(self.Lb, self.ang).matrix()
        M_y_block = M[2:4, 2:4]
        assert np.allclose(M_y_block, [[1.0, self.Lb], [0.0, 1.0]])


# ── FODO lattice ──────────────────────────────────────────────────────────────

class TestFODO:
    def test_stable(self):
        """make_fodo default parameters produce a stable lattice."""
        lat = make_fodo(Lq=0.5, Ld=2.0, n_cells=1)
        Qx, Qy = lat.tune()
        # Stable: tune in (0, 0.5)
        assert 0.0 < Qx < 0.5
        assert 0.0 < Qy < 0.5

    def test_tune_approx_quarter(self):
        """With f ≈ L_cell/(2√2) the thin-lens tune is ~0.25."""
        lat = make_fodo(Lq=0.5, Ld=2.0, n_cells=1)
        Qx, _ = lat.tune()
        # Thick-lens correction shifts the tune slightly below 0.25
        assert 0.20 < Qx < 0.27

    def test_twiss_courant_snyder_invariant(self):
        """βγ − α² = 1 at every point in the lattice."""
        lat = make_fodo(Lq=0.5, Ld=2.0, n_cells=1)
        t = lat.twiss()
        cs_x = t["beta_x"] * t["gamma_x"] - t["alpha_x"]**2
        cs_y = t["beta_y"] * t["gamma_y"] - t["alpha_y"]**2
        assert np.allclose(cs_x, 1.0, atol=1e-10)
        assert np.allclose(cs_y, 1.0, atol=1e-10)

    def test_twiss_beta_positive(self):
        """β-functions must be positive everywhere (use n_cells=1 to avoid Q≈1)."""
        lat = make_fodo(n_cells=1)
        t = lat.twiss()
        assert np.all(t["beta_x"] > 0.0)
        assert np.all(t["beta_y"] > 0.0)

    def test_circumference(self):
        Lq, Ld, n = 0.5, 2.0, 4
        lat = make_fodo(Lq=Lq, Ld=Ld, n_cells=n)
        # Cell: QF/2 + D1 + QD + D2 + QF/2 = Lq/2 + Ld + Lq + Ld + Lq/2 = 2*(Lq+Ld)
        C_expected = n * 2 * (Lq + Ld)
        assert pytest.approx(lat.circumference, rel=1e-9) == C_expected


# ── Tracking: Liouville's theorem ─────────────────────────────────────────────

class TestTracking:
    def test_linear_emittance_conservation(self):
        """Under purely linear tracking, geometric emittance must be conserved."""
        lat = make_fodo(Lq=0.5, Ld=2.0, n_cells=1)
        t0 = lat.twiss_at_start()

        beam = Beam.gaussian(
            n=2_000,
            emittance_x=1e-6, beta_x=t0["beta_x"], alpha_x=t0["alpha_x"],
            emittance_y=1e-6, beta_y=t0["beta_y"], alpha_y=t0["alpha_y"],
            rng=np.random.default_rng(42),
        )

        history = track(beam.particles, lat, n_turns=20)
        # history shape: (21, 1, N, 6)
        eps_initial = Beam(history[0, 0]).emittance("x")
        eps_final   = Beam(history[-1, 0]).emittance("x")
        # Relative change should be < 1%
        assert abs(eps_final / eps_initial - 1.0) < 0.01

    def test_track_output_shape(self):
        lat = make_fodo(n_cells=2)
        n_p = 10
        pts = np.zeros((n_p, 6))
        h = track(pts, lat, n_turns=5)
        assert h.shape == (6, 1, n_p, 6)

    def test_on_axis_particle_stable(self):
        """A particle starting exactly on the closed orbit stays on it."""
        lat = make_fodo(n_cells=4)
        pts = np.zeros((1, 6))
        h = track(pts, lat, n_turns=50)
        assert np.allclose(h[:, 0, 0, :], 0.0, atol=1e-12)


# ── RF cavity ─────────────────────────────────────────────────────────────────

class TestRFCavity:
    def test_longitudinal_phase_rotation(self):
        """RF cavity should rotate the (δ, l) phase space."""
        rf = RFCavity(length=0.5, voltage=1e5, frequency=400e6, phi_s=math.pi / 2,
                      energy0=1e9)
        # A particle displaced in l should get a delta kick
        p = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.01]])  # l displaced
        out = rf.track(p)
        # The kick V*sin(phi_s - k_rf * l) - V*sin(phi_s) should be negative
        # for phi_s = π/2 and small positive l (cos term)
        assert out[0, 4] != 0.0, "RF cavity should change delta"

    def test_on_momentum_no_energy_change(self):
        """Synchronous particle (l=0) at phi_s = π/2 gets no energy change."""
        rf = RFCavity(length=0.5, voltage=1e5, frequency=400e6, phi_s=math.pi / 2,
                      energy0=1e9)
        p = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        out = rf.track(p)
        # sin(pi/2 - 0) - sin(pi/2) = 1 - 1 = 0
        assert pytest.approx(out[0, 4], abs=1e-15) == 0.0


# ── Ring ──────────────────────────────────────────────────────────────────────

class TestRing:
    def test_ring_stable(self):
        """make_ring with default parameters should be stable (tune ∈ (0, 1))."""
        lat = make_ring(n_bends=8, Lbend=2.0, Lq=0.3, Ld=0.5)
        Qx, Qy = lat.tune()
        assert 0.0 < Qx < 1.0
        assert 0.0 < Qy < 1.0

    def test_dispersion_nonzero_in_ring(self):
        """A ring with dipoles must have non-zero dispersion somewhere."""
        lat = make_ring(n_bends=8)
        t = lat.twiss()
        assert np.max(np.abs(t["Dx"])) > 1e-4

    def test_momentum_compaction_positive(self):
        """In a standard ring, αc > 0 (above transition energy)."""
        lat = make_ring(n_bends=8)
        alpha_c = lat.momentum_compaction()
        assert alpha_c > 0.0


# ── Marker ────────────────────────────────────────────────────────────────────

class TestMarker:
    def test_identity(self):
        m = Marker("OBS")
        p = np.random.default_rng(0).standard_normal((5, 6))
        assert np.allclose(m.track(p), p)
        assert np.allclose(m.matrix(), np.eye(6))
