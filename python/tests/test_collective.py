"""Tests for collective instability tracker.

Empirically calibrated parameters (N=3000, σ_Q = 0.002, β=10, Q=0.28):
  growth rate per turn  μ ≈ κ · β · sin²(2πQ) / 2 ≈ 4.84 κ
  decoherence time      τ ≈ 1/(2π σ_Q) ≈ 80 turns
  Landau threshold      κ_th ≈ σ_Q / (β sin²(2πQ)) ≈ 0.004–0.008 rad/m

Observable signal requires x0_offset >> σ_x/√N ≈ 58 μm, so we use 2 mm.
"""

import numpy as np
import pytest

from mathphys.collective import CollectiveParams, CollectiveRing
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
