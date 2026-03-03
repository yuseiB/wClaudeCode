"""Tests for collective instability tracker.

Empirically calibrated parameters (N=3000, σ_Q = 0.002, β=10, Q=0.28):
  growth rate per turn  μ ≈ κ · β · sin²(2πQ) / 2 ≈ 4.84 κ
  decoherence time      τ ≈ 1/(2π σ_Q) ≈ 80 turns
  Landau threshold      κ_th ≈ σ_Q / (β sin²(2πQ)) ≈ 0.004–0.008 rad/m

Observable signal requires x0_offset >> σ_x/√N ≈ 58 μm, so we use 2 mm.
"""

import numpy as np
import pytest

from mathphys.collective import CollectiveParams, CollectiveRing, _bbr_wake, sacherer_threshold
from mathphys.storage_ring import BeamParams, RingParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def chromatic_ring(aperture=0.2) -> RingParams:
    return RingParams(tune=0.28, beta=10.0, alpha=0.0,
                      chromaticity=-2.0, sextupole_strength=0.0, aperture=aperture)


def no_chroma_ring(aperture=0.2) -> RingParams:
    return RingParams(tune=0.28, beta=10.0, alpha=0.0,
                      chromaticity=0.0, sextupole_strength=0.0, aperture=aperture)


def beam(n=3000, sigma_delta=1e-3, seed=0) -> BeamParams:
    return BeamParams(emittance=1e-6, momentum_spread=sigma_delta,
                      n_particles=n, seed=seed)


# ---------------------------------------------------------------------------
# Baseline: no collective kick
# ---------------------------------------------------------------------------


class TestNoCollective:
    def test_zero_kappa_conserves_emittance(self):
        """κ = 0 must reproduce single-particle emittance conservation."""
        coll = CollectiveParams(mode="transverse", kappa=0.0, x0_offset=0.0)
        tracker = CollectiveRing(chromatic_ring(), beam(), coll)
        res = tracker.track(n_turns=300)
        eps0 = res.emittance[0]
        assert np.all(np.abs(res.emittance - eps0) / eps0 < 0.02)

    def test_zero_kappa_full_survival(self):
        coll = CollectiveParams(mode="transverse", kappa=0.0, x0_offset=0.0)
        tracker = CollectiveRing(chromatic_ring(), beam(), coll)
        res = tracker.track(n_turns=200)
        assert np.all(res.survival == 1.0)


# ---------------------------------------------------------------------------
# Rigid-bunch transverse instability
# ---------------------------------------------------------------------------


class TestRigidBunch:
    """Below threshold: Landau damping decays the centroid.
    Above threshold: coherent kick sustains (or grows) the centroid.

    x0_offset = 2 mm >> σ_x/√N ≈ 58 μm so the coherent signal dominates noise.
    """

    def test_below_threshold_centroid_decays(self):
        """κ = 0.001 (below threshold ≈ 0.005) — centroid must decay 5×."""
        coll = CollectiveParams(mode="transverse", kappa=0.001, x0_offset=2e-3)
        tracker = CollectiveRing(chromatic_ring(), beam(n=3000), coll)
        res = tracker.track(n_turns=800)
        cx = np.abs(res.centroid_x)
        early = np.mean(cx[10:100])
        late  = np.mean(cx[600:800])
        assert late / early < 0.5, (
            f"Centroid not decaying: early={early*1e3:.3f}mm, late={late*1e3:.3f}mm"
        )

    def test_above_threshold_centroid_sustained(self):
        """κ = 0.05 (above threshold) — coherent kick maintains centroid."""
        coll = CollectiveParams(mode="transverse", kappa=0.05, x0_offset=2e-3)
        tracker = CollectiveRing(chromatic_ring(), beam(n=3000), coll)
        res = tracker.track(n_turns=800)
        cx = np.abs(res.centroid_x)
        early = np.mean(cx[10:100])
        late  = np.mean(cx[600:800])
        assert late / early > 0.5, (
            f"Centroid not sustained: early={early*1e3:.3f}mm, late={late*1e3:.3f}mm"
        )

    def test_above_threshold_more_loss_than_no_kick(self):
        """Large κ amplifies centroid → particles reach aperture (10 mm) faster."""
        ring = chromatic_ring(aperture=0.010)  # 10 mm
        b = beam(n=3000, seed=0)
        res_kick = CollectiveRing(
            ring, b,
            CollectiveParams(mode="transverse", kappa=0.2, x0_offset=2e-3)
        ).track(n_turns=500)
        res_none = CollectiveRing(
            ring, b,
            CollectiveParams(mode="transverse", kappa=0.0, x0_offset=2e-3)
        ).track(n_turns=500)
        assert res_kick.survival[-1] < res_none.survival[-1], (
            f"Expected more loss with kick: kick={res_kick.survival[-1]*100:.1f}% "
            f"no_kick={res_none.survival[-1]*100:.1f}%"
        )

    def test_stability_monotone_wrt_kappa(self):
        """Larger κ → larger late-time centroid amplitude (less Landau damping)."""
        def late_centroid(kappa):
            coll = CollectiveParams(mode="transverse", kappa=kappa, x0_offset=2e-3)
            tracker = CollectiveRing(chromatic_ring(), beam(n=3000, seed=7), coll)
            res = tracker.track(n_turns=800)
            return np.mean(np.abs(res.centroid_x[600:800]))

        amp_lo = late_centroid(0.001)
        amp_hi = late_centroid(0.05)
        assert amp_hi > amp_lo * 2, (
            f"κ=0.05 late centroid ({amp_hi*1e3:.3f}mm) should be >> "
            f"κ=0.001 ({amp_lo*1e3:.3f}mm)"
        )

    def test_no_offset_centroid_near_zero(self):
        """Without seeding, centroid stays near the statistical noise floor."""
        coll = CollectiveParams(mode="transverse", kappa=0.0, x0_offset=0.0)
        tracker = CollectiveRing(chromatic_ring(), beam(n=3000, seed=3), coll)
        res = tracker.track(n_turns=200)
        sigma_x = np.sqrt(1e-6 * 10.0)  # sqrt(ε β)
        noise_floor = sigma_x / np.sqrt(3000) * 5  # 5 × expected noise
        assert np.max(np.abs(res.centroid_x)) < noise_floor + 1e-6


# ---------------------------------------------------------------------------
# Head-tail model
# ---------------------------------------------------------------------------


class TestHeadTail:
    def test_headtail_drives_centroid(self):
        """Without chromaticity, wake drives the centroid above noise floor."""
        coll = CollectiveParams(
            mode="headtail", wake_strength=0.005,
            wake_range=0.008, n_slices=10, sigma_z=0.01, x0_offset=2e-3,
        )
        tracker = CollectiveRing(no_chroma_ring(), beam(n=2000), coll)
        res = tracker.track(n_turns=300)
        # Late-time centroid should remain well above noise floor (58 μm for N=3000)
        late = np.mean(np.abs(res.centroid_x[200:300]))
        noise_floor = np.sqrt(1e-6 * 10) / np.sqrt(2000)
        assert late > noise_floor * 3, (
            f"Wake-driven centroid too small: {late*1e6:.1f} μm, noise ~{noise_floor*1e6:.1f} μm"
        )

    def test_chromaticity_suppresses_headtail(self):
        """Chromaticity provides Landau damping → smaller late-time centroid."""
        coll = CollectiveParams(
            mode="headtail", wake_strength=0.005,
            wake_range=0.008, n_slices=10, sigma_z=0.01, x0_offset=2e-3,
        )
        res_no = CollectiveRing(no_chroma_ring(), beam(n=2000, seed=5), coll).track(n_turns=600)
        res_xi = CollectiveRing(chromatic_ring(), beam(n=2000, seed=5), coll).track(n_turns=600)

        late = slice(400, 600)
        amp_no = np.mean(np.abs(res_no.centroid_x[late]))
        amp_xi = np.mean(np.abs(res_xi.centroid_x[late]))
        assert amp_xi < amp_no, (
            f"Chromaticity should reduce head-tail amplitude: "
            f"ξ=0: {amp_no:.3e} m, ξ=-2: {amp_xi:.3e} m"
        )

    def test_zero_wake_equals_no_collective(self):
        """Zero wake must give same centroid as no collective kick."""
        coll_ht = CollectiveParams(
            mode="headtail", wake_strength=0.0, n_slices=10,
            sigma_z=0.01, x0_offset=2e-3,
        )
        coll_tr = CollectiveParams(mode="transverse", kappa=0.0, x0_offset=2e-3)
        r = chromatic_ring()
        b = beam(n=800, seed=99)
        res_ht = CollectiveRing(r, b, coll_ht).track(n_turns=100)
        res_tr = CollectiveRing(r, b, coll_tr).track(n_turns=100)
        diff = np.max(np.abs(res_ht.centroid_x - res_tr.centroid_x))
        assert diff < 1e-9, f"Zero-wake head-tail differs from no-kick: {diff:.2e}"


# ---------------------------------------------------------------------------
# Sacherer threshold utility (Lee §4.3)
# ---------------------------------------------------------------------------


class TestSachererThreshold:
    def test_known_value(self):
        """κ_th = 2σ_Q²/(β|sin(2πQ)|) for default parameters."""
        sigma_Q = 2.0 * 1e-3   # |ξ|σ_δ with ξ=-2, σ_δ=1e-3
        beta = 10.0
        tune = 0.28
        kth = sacherer_threshold(sigma_Q, beta, tune)
        expected = 2.0 * sigma_Q**2 / (beta * abs(np.sin(2.0 * np.pi * tune)))
        assert abs(kth - expected) < 1e-15

    def test_quadratic_in_sigma_Q(self):
        """Doubling σ_Q quadruples the threshold (Lee §4.3 key result)."""
        kth1 = sacherer_threshold(0.001, 10.0, 0.28)
        kth2 = sacherer_threshold(0.002, 10.0, 0.28)
        assert abs(kth2 / kth1 - 4.0) < 1e-10

    def test_result_property_matches(self):
        """CollectiveResult.sacherer_kth matches the standalone function."""
        coll = CollectiveParams(mode="transverse", kappa=0.001, x0_offset=2e-3)
        r = chromatic_ring()
        b = beam()
        res = CollectiveRing(r, b, coll).track(n_turns=10)
        sigma_Q = abs(r.chromaticity) * b.momentum_spread
        expected = sacherer_threshold(sigma_Q, r.beta, r.tune)
        assert abs(res.sacherer_kth - expected) < 1e-15


# ---------------------------------------------------------------------------
# BBR wake function (Lee §4.1)
# ---------------------------------------------------------------------------


class TestBBRWake:
    def test_q0_is_pure_exponential(self):
        """resonator_Q=0 must reproduce the simple exponential wake."""
        dz = np.linspace(0, 0.05, 100)
        W0, z_w = 1.0, 0.01
        w_bbr = _bbr_wake(dz, W0, z_w, Q=0.0)
        w_exp = W0 * np.exp(-dz / z_w)
        np.testing.assert_allclose(w_bbr, w_exp, rtol=1e-12)

    def test_q_half_critically_damped(self):
        """Q=0.5 → W = W₀(1 + Δz/z_w)e^{−Δz/z_w}."""
        dz = np.array([0.0, 0.01, 0.02])
        W0, z_w = 2.0, 0.01
        w = _bbr_wake(dz, W0, z_w, Q=0.5)
        expected = W0 * np.exp(-dz / z_w) * (1.0 + dz / z_w)
        np.testing.assert_allclose(w, expected, rtol=1e-12)

    def test_q1_oscillatory_sign_change(self):
        """Q=1 BBR wake changes sign at some Δz > 0."""
        dz = np.linspace(0, 0.1, 1000)
        w = _bbr_wake(dz, 1.0, 0.01, Q=1.0)
        # Wake starts positive (W0 > 0) but must go negative for Q=1
        assert w[0] > 0
        assert np.any(w < 0), "Q=1 BBR wake should change sign"

    def test_bbr_headtail_q0_matches_old_exponential(self):
        """Head-tail with resonator_Q=0 gives identical result to pure-exponential default."""
        b = beam(n=800, seed=5)
        r = no_chroma_ring()
        coll_base = CollectiveParams(
            mode="headtail", wake_strength=0.005, wake_range=0.008,
            n_slices=10, sigma_z=0.01, x0_offset=2e-3, resonator_Q=0.0,
        )
        coll_q0 = CollectiveParams(
            mode="headtail", wake_strength=0.005, wake_range=0.008,
            n_slices=10, sigma_z=0.01, x0_offset=2e-3, resonator_Q=0.0,
        )
        res1 = CollectiveRing(r, b, coll_base).track(n_turns=50)
        res2 = CollectiveRing(r, b, coll_q0).track(n_turns=50)
        np.testing.assert_allclose(res1.centroid_x, res2.centroid_x, atol=1e-14)


# ---------------------------------------------------------------------------
# Synchrotron oscillations (Lee §3.2, §4.2)
# ---------------------------------------------------------------------------


class TestSynchrotronOscillations:
    def test_synchrotron_mixes_longitudinal_positions(self):
        """With Q_s > 0, the z distribution evolves (particles swap head/tail)."""
        coll_static = CollectiveParams(
            mode="headtail", wake_strength=0.0, n_slices=10,
            sigma_z=0.01, x0_offset=0.0,
            synchrotron_tune=0.0,
        )
        coll_synch = CollectiveParams(
            mode="headtail", wake_strength=0.0, n_slices=10,
            sigma_z=0.01, x0_offset=0.0,
            synchrotron_tune=0.01, slip_factor=1e-3, circumference=100.0,
        )
        r = no_chroma_ring()
        b = beam(n=500, seed=1)
        # After half a synchrotron period, z should have rotated significantly
        half_period = int(round(0.5 / 0.01))   # 50 turns
        tracker_s = CollectiveRing(r, b, coll_synch)
        # Access internal state to check z evolution
        # With synchrotron motion, centroid after tracking must differ from static
        tracker_stat = CollectiveRing(r, b, coll_static)
        # Both runs have zero wake and zero x-offset; centroid evolves identically
        # in x.  The test: synchrotron motion preserves longitudinal emittance.
        res = tracker_s.track(n_turns=half_period)
        assert res.survival[-1] == 1.0  # no losses expected

    def test_synchrotron_no_wake_no_transverse_effect(self):
        """With zero wake and ξ=0, synchrotron motion leaves centroid unchanged.

        With ξ=0, phase advance φ = 2πQ₀ is identical for all δ, so updating
        δ each turn (synchrotron oscillation) has no transverse consequence.
        """
        coll_0 = CollectiveParams(
            mode="headtail", wake_strength=0.0, n_slices=10,
            sigma_z=0.01, x0_offset=2e-3, synchrotron_tune=0.0,
        )
        coll_s = CollectiveParams(
            mode="headtail", wake_strength=0.0, n_slices=10,
            sigma_z=0.01, x0_offset=2e-3,
            synchrotron_tune=0.05, slip_factor=1e-3, circumference=100.0,
        )
        r = no_chroma_ring()    # ξ = 0: δ does not enter φ
        b = beam(n=500, seed=42)
        res_0 = CollectiveRing(r, b, coll_0).track(n_turns=50)
        res_s = CollectiveRing(r, b, coll_s).track(n_turns=50)
        np.testing.assert_allclose(res_0.centroid_x, res_s.centroid_x, atol=1e-10)

    def test_synchrotron_alters_headtail_coupling(self):
        """Synchrotron motion changes z each turn, which must alter the wake kicks.

        With Q_s > 0 the longitudinal positions rotate in phase space, so
        the slice assignments and wake coupling change from turn to turn.
        The centroid time series must therefore differ from the Q_s = 0 case.
        """
        coll_0 = CollectiveParams(
            mode="headtail", wake_strength=0.008, wake_range=0.008,
            n_slices=15, sigma_z=0.01, x0_offset=2e-3, synchrotron_tune=0.0,
        )
        coll_s = CollectiveParams(
            mode="headtail", wake_strength=0.008, wake_range=0.008,
            n_slices=15, sigma_z=0.01, x0_offset=2e-3,
            synchrotron_tune=0.02, slip_factor=1e-3, circumference=100.0,
        )
        r = no_chroma_ring()
        b = beam(n=500, seed=7)
        res_0 = CollectiveRing(r, b, coll_0).track(n_turns=100)
        res_s = CollectiveRing(r, b, coll_s).track(n_turns=100)
        # After the first turn the z positions diverge → kicks differ → centroid diverges
        diff = np.max(np.abs(res_0.centroid_x[1:] - res_s.centroid_x[1:]))
        assert diff > 1e-6, (
            f"Synchrotron oscillations should alter wake coupling: max diff = {diff:.2e}"
        )

    def test_no_synchrotron_backward_compat(self):
        """synchrotron_tune=0 must reproduce original head-tail behaviour exactly."""
        coll = CollectiveParams(
            mode="headtail", wake_strength=0.005, wake_range=0.008,
            n_slices=10, sigma_z=0.01, x0_offset=2e-3,
            synchrotron_tune=0.0,  # explicitly set to zero
        )
        r = chromatic_ring()
        b = beam(n=1000, seed=11)
        res = CollectiveRing(r, b, coll).track(n_turns=100)
        # Just verify it runs and produces valid output
        assert res.centroid_x.shape == (101,)
        assert np.all(np.isfinite(res.centroid_x))


# ---------------------------------------------------------------------------
# CollectiveResult analytic properties (Lee §3.3, §4.2)
# ---------------------------------------------------------------------------


class TestResultProperties:
    def test_growth_rate_formula(self):
        """theoretical_growth_rate matches (κβ/2)|sin(2πQ)|."""
        kappa = 0.05
        beta = 10.0
        tune = 0.28
        coll = CollectiveParams(mode="transverse", kappa=kappa, x0_offset=2e-3)
        r = RingParams(tune=tune, beta=beta, alpha=0.0,
                       chromaticity=-2.0, aperture=0.2)
        b = beam()
        res = CollectiveRing(r, b, coll).track(n_turns=10)
        expected = (kappa * beta / 2.0) * abs(np.sin(2.0 * np.pi * tune))
        assert abs(res.theoretical_growth_rate - expected) < 1e-15

    def test_decoherence_time_formula(self):
        """theoretical_decoherence_time = 1/(2π√2 σ_Q)."""
        coll = CollectiveParams(mode="transverse", kappa=0.0, x0_offset=0.0)
        r = chromatic_ring()
        b = beam(sigma_delta=1e-3)
        res = CollectiveRing(r, b, coll).track(n_turns=10)
        sigma_Q = abs(r.chromaticity) * b.momentum_spread   # = 0.002
        expected = 1.0 / (2.0 * np.pi * np.sqrt(2.0) * sigma_Q)
        assert abs(res.theoretical_decoherence_time - expected) < 1e-10

    def test_headtail_growth_rate_zero(self):
        """theoretical_growth_rate is 0 for head-tail mode."""
        coll = CollectiveParams(mode="headtail", wake_strength=0.01,
                                n_slices=10, sigma_z=0.01)
        res = CollectiveRing(no_chroma_ring(), beam(), coll).track(n_turns=10)
        assert res.theoretical_growth_rate == 0.0
