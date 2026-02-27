"""Storage-ring single-particle tracking with Monte Carlo beam sampling.

Model
-----
* 2-D (horizontal) one-turn map built from Courant-Snyder (Twiss) parameters.
* Optional chromaticity: tune shifts linearly with momentum offset δ.
* Optional thin-lens sextupole kick applied once per turn (introduces
  amplitude-dependent tune shift and limits the dynamic aperture).
* Particles outside the physical aperture are marked as permanently lost.
* Collective instabilities are *not* modelled.

State vector per particle: (x [m], x' [rad], δ [1])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------


@dataclass
class RingParams:
    """Machine parameters for a simplified storage ring.

    Parameters
    ----------
    tune : float
        Horizontal fractional tune Q (e.g. 0.28).
    beta : float
        Beta function [m] at the tracking reference point.
    alpha : float
        Alpha Twiss parameter (= -β'/2) at the reference point.
    chromaticity : float
        Chromaticity ξ = dQ/dδ.  Natural chromaticity is typically negative
        (e.g. −2).  Set to 0 to ignore momentum spread.
    sextupole_strength : float
        Integrated thin-lens sextupole strength k₂L/2 [m⁻²].  A nonzero
        value introduces a nonlinear kick Δx' = −(k₂L/2)·x² each turn.
        Set to 0 for pure linear optics.
    aperture : float
        Physical half-aperture [m].  Particles with |x| > aperture are lost.
        Use ``np.inf`` for no aperture cut.
    """

    tune: float = 0.28
    beta: float = 10.0
    alpha: float = 0.0
    chromaticity: float = 0.0
    sextupole_strength: float = 0.0
    aperture: float = np.inf

    @property
    def gamma(self) -> float:
        """Courant-Snyder γ = (1 + α²) / β."""
        return (1.0 + self.alpha**2) / self.beta


@dataclass
class BeamParams:
    """Initial beam distribution parameters.

    Parameters
    ----------
    emittance : float
        RMS geometric emittance ε₀ [m·rad].
    momentum_spread : float
        RMS relative momentum spread σ_δ (e.g. 1e-3).
    n_particles : int
        Number of macro-particles.
    seed : int or None
        Random seed for reproducibility.
    """

    emittance: float = 1e-6
    momentum_spread: float = 1e-3
    n_particles: int = 1000
    seed: int | None = 42


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class StorageRing:
    """Monte Carlo single-particle tracker for a simplified storage ring.

    Examples
    --------
    >>> ring = RingParams(tune=0.28, beta=10.0, alpha=0.5, chromaticity=-2.0)
    >>> beam = BeamParams(emittance=1e-6, momentum_spread=1e-3, n_particles=1000)
    >>> tracker = StorageRing(ring, beam)
    >>> result = tracker.track(n_turns=1000)
    >>> result.survival[-1]   # fraction of surviving particles at end
    """

    def __init__(self, ring: RingParams, beam: BeamParams) -> None:
        self.ring = ring
        self.beam = beam
        rng = np.random.default_rng(beam.seed)
        self._x0, self._xp0, self._delta0 = self._sample_beam(rng)

    # ------------------------------------------------------------------
    # Beam sampling
    # ------------------------------------------------------------------

    def _sample_beam(
        self, rng: np.random.Generator
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Draw particles from a matched Gaussian in Courant-Snyder phase space."""
        n = self.beam.n_particles
        r = self.ring
        eps = self.beam.emittance

        # Two independent standard-normal samples
        u = rng.standard_normal(n)
        v = rng.standard_normal(n)

        # Map to (x, x') matched to Twiss ellipse
        x = np.sqrt(eps * r.beta) * u
        xp = np.sqrt(eps / r.beta) * (-r.alpha * u + v)
        delta = rng.standard_normal(n) * self.beam.momentum_spread

        return x, xp, delta

    # ------------------------------------------------------------------
    # Main tracking loop
    # ------------------------------------------------------------------

    def track(
        self,
        n_turns: int,
        sample_turns: list[int] | None = None,
    ) -> "TrackingResult":
        """Track all particles for *n_turns* revolution turns.

        Parameters
        ----------
        n_turns : int
            Total number of turns to simulate.
        sample_turns : list[int] or None
            Turns at which full (x, x') phase-space snapshots are stored.
            Defaults to four equally-spaced points including turn 0 and
            turn n_turns.

        Returns
        -------
        TrackingResult
        """
        if sample_turns is None:
            sample_turns = sorted({0, n_turns // 4, n_turns // 2, n_turns})
        sample_set = set(sample_turns)

        r = self.ring
        n = self.beam.n_particles

        x = self._x0.copy()
        xp = self._xp0.copy()
        delta = self._delta0.copy()
        alive = np.ones(n, dtype=bool)

        emittance_arr = np.zeros(n_turns + 1)
        sigma_x_arr = np.zeros(n_turns + 1)
        survival_arr = np.zeros(n_turns + 1)
        snapshots: dict[int, tuple[NDArray, NDArray]] = {}

        def _record(turn: int) -> None:
            mask = alive
            emittance_arr[turn] = rms_emittance(x[mask], xp[mask])
            sigma_x_arr[turn] = float(np.std(x[mask])) if mask.any() else 0.0
            survival_arr[turn] = mask.sum() / n
            if turn in sample_set:
                snapshots[turn] = (x[mask].copy(), xp[mask].copy())

        _record(0)

        for turn in range(1, n_turns + 1):
            # Chromatic phase advance: φ = 2π (Q + ξ δ)
            phi = 2.0 * np.pi * (r.tune + r.chromaticity * delta)
            c = np.cos(phi)
            s = np.sin(phi)

            # Vectorised Courant-Snyder one-turn map
            x_new = (c + r.alpha * s) * x + r.beta * s * xp
            xp_new = -r.gamma * s * x + (c - r.alpha * s) * xp
            x, xp = x_new, xp_new

            # Thin-lens sextupole kick: Δx' = −(k₂L/2) x²
            if r.sextupole_strength != 0.0:
                xp -= r.sextupole_strength * x**2

            # Aperture cut — lost particles stay at origin (don't affect stats)
            alive &= np.abs(x) <= r.aperture
            x[~alive] = 0.0
            xp[~alive] = 0.0

            _record(turn)

        return TrackingResult(
            turns=np.arange(n_turns + 1),
            emittance=emittance_arr,
            sigma_x=sigma_x_arr,
            survival=survival_arr,
            snapshots=snapshots,
            ring=r,
            beam=self.beam,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rms_emittance(x: NDArray, xp: NDArray) -> float:
    """RMS geometric emittance ε = sqrt(<x²><x'²> − <x x'>²).

    Parameters
    ----------
    x, xp : array-like
        Transverse position [m] and angle [rad] of the particle ensemble.

    Returns
    -------
    float
        RMS emittance [m·rad].  Returns 0 if fewer than 2 particles.
    """
    if len(x) < 2:
        return 0.0
    x2 = np.mean(x**2)
    xp2 = np.mean(xp**2)
    xxp = np.mean(x * xp)
    return float(np.sqrt(max(x2 * xp2 - xxp**2, 0.0)))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TrackingResult:
    """Turn-by-turn output from a Monte Carlo tracking run.

    Attributes
    ----------
    turns : ndarray, shape (n_turns+1,)
        Turn numbers 0 … n_turns.
    emittance : ndarray, shape (n_turns+1,)
        RMS geometric emittance [m·rad] of surviving particles.
    sigma_x : ndarray, shape (n_turns+1,)
        RMS beam size σ_x [m] of surviving particles.
    survival : ndarray, shape (n_turns+1,)
        Fraction of particles still alive (0 … 1).
    snapshots : dict[int, tuple[ndarray, ndarray]]
        Phase-space (x, x') arrays for selected turns.
    ring : RingParams
        Ring parameters used for this run.
    beam : BeamParams
        Beam parameters used for this run.
    """

    turns: NDArray
    emittance: NDArray
    sigma_x: NDArray
    survival: NDArray
    snapshots: dict[int, tuple[NDArray, NDArray]]
    ring: RingParams
    beam: BeamParams
