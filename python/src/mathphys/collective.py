"""Collective Instabilities in Storage Rings
===========================================

Theory Background
-----------------

**What are collective instabilities?**

In the single-particle model (``storage_ring.py``) each particle moves
independently under the external lattice fields.  At sufficiently high
beam current the electromagnetic fields *radiated by the beam itself*
become comparable to the external fields.  These self-fields — called
*wake fields* or, in the frequency domain, *impedances* — act back on
subsequent particles or on the same particle on a later pass.  The
result is that the *ensemble* as a whole can develop coherent oscillation
modes that grow exponentially: **collective instabilities**.

Two classes of collective effects are modelled here:

1. **Rigid-bunch transverse instability** — the simplest possible model.
   The whole bunch is treated as a single macro-particle.  Its centroid
   displacement x̄ generates a coherent wake kick that is proportional to
   x̄ itself, providing positive feedback that may overcome Landau damping.

2. **Sliced head-tail instability** — a within-bunch effect.  The bunch is
   divided into longitudinal slices.  Each "head" slice (arriving first at
   a cavity or resistive-wall location) excites a transverse wake that
   kicks the "tail" slices trailing behind it.  Chromaticity and momentum
   spread modify this coupling and can either stabilize or destabilize
   different coherent modes.

Neither **space charge** nor **longitudinal instabilities** nor
**multi-bunch coupled-bunch modes** are included.

---

**Impedance and Wake Functions**

A real vacuum chamber responds to a passing bunch with a transverse wake
function W_⊥(s) [V/C/m] where *s* is the longitudinal separation
(s > 0: the kicked particle is *behind* the source particle).

For a **broad-band resonator** (Q = 1), the transverse wake is::

    W_⊥(s) = W₀ · exp(−s / z_w),    s ≥ 0

where W₀ [rad/m per unit charge per metre] is the peak strength and
z_w [m] is the wake decay length.  This model approximates a cavity
or resistive-wall dominated ring with a single effective resonance.

The **integrated kick** (in rad) received by a particle at position s₂
from a source particle at s₁ < s₂ carrying offset x₁ is::

    Δx' = (N_b · r₀ / γ) · W_⊥(s₂ − s₁) · x₁

where N_b is the bunch population, r₀ is the classical particle radius,
and γ is the Lorentz factor.  In the code we absorb these constants into
a single dimensionless ``wake_strength`` κ_w.

---

**Rigid-Bunch Coherent Instability and Landau Damping**

For the rigid-bunch model the coherent kick per turn is::

    Δx'_i = κ_coh · x̄_n   (applied to every particle i)

where x̄_n = (1/N) Σ xᵢ is the centroid at turn n.

*Without* any tune spread the one-turn matrix for the centroid acquires an
extra element and its eigenvalues leave the unit circle → the centroid
grows without bound.

*With* chromaticity (ξ = dQ/dδ) and momentum spread σ_δ each particle
oscillates at its own tune Qᵢ = Q₀ + ξ δᵢ.  The centroid amplitude
decays through **Landau damping**::

    x̄(n) ∝ exp(−n²σ_Q²/2) · cos(2π Q₀ n),    σ_Q = |ξ| σ_δ

The coherent kick counteracts this damping.  The **stability criterion**
(Sacherer integral, simplified) is approximately::

    κ_coh  <  κ_th ≈ 4π σ_Q / (β |sin 2π Q₀|)

Below threshold: Landau damping wins — centroid oscillations decay.
Above threshold: coherent drive wins — centroid grows exponentially,
particles eventually hit the aperture.

In the Monte Carlo model this competition is *implicit*: individual
particles diffuse apart in phase (Landau damping) while the coherent
kick keeps re-phasing them.  No analytic approximation is needed.

---

**Sliced Head-Tail Instability**

Within a single bunch the longitudinal position z (with z > 0 at the
bunch head) varies particle to particle.  Particles at the head arrive
first at an impedance source and excite a wake that kicks particles at
the tail (Δz = z_head − z_tail > 0).

Dividing the bunch into K slices of equal width Δz, the coherent
kick received by slice j from all preceding slices i (head, zᵢ > zⱼ)::

    Δx'_j = κ_w · Σ_{i: zᵢ > zⱼ} W_⊥(zᵢ − zⱼ) · x̄ᵢ / K

*Without chromaticity* (ξ = 0): all particles at the same tune; the
head continuously drives the tail with no phase reversal → secular
(linear) growth of the tail amplitude — **head-tail instability**.

*With chromaticity* (ξ ≠ 0): each slice has a slightly different average
tune because the momentum spread σ_δ introduces a chromatic phase
advance between slices.  The coherent motion within each slice
decohere faster, providing effective Landau damping that can quench
the head-tail growth — **chromaticity stabilization**.

Note: a full treatment of the classical (Courant-Snyder) head-tail
instability also requires synchrotron oscillations (longitudinal
phase-space motion at tune Q_s).  This model omits synchrotron motion
for clarity; each particle's z is drawn once from N(0, σ_z) and held
fixed throughout the simulation.

---

Implementation Notes
--------------------
* All physics is tracked at the same level as ``StorageRing``: a
  vectorised NumPy loop over turns.
* The collective kick is applied *before* the one-turn lattice map,
  consistent with a lumped-kick lattice model.
* Particles marked as lost (|x| > aperture) contribute neither to the
  centroid nor to the wake sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mathphys.storage_ring import BeamParams, RingParams, rms_emittance


# ---------------------------------------------------------------------------
# Module-level physics utilities
# ---------------------------------------------------------------------------


def sacherer_threshold(sigma_Q: float, beta: float, tune: float) -> float:
    """Sacherer–Gaussian Landau-damping threshold for the rigid-bunch mode.

    Derived from the Vlasov dispersion integral for a Gaussian tune
    distribution (Lee §4.3, theory doc §4.2, eq. 4.6):

    .. math::

        \\kappa_{\\rm th} = \\frac{2\\,\\sigma_Q^2}{\\beta\\,|\\sin(2\\pi Q_0)|}

    Valid in the ``N → ∞`` Vlasov limit.  For finite ``N`` the empirical
    threshold is higher due to finite-ensemble noise (see theory doc §4.4).

    Parameters
    ----------
    sigma_Q : float
        RMS tune spread.  For chromaticity-dominated spread: ``σ_Q = |ξ| σ_δ``.
    beta : float
        Beta function at the kick location [m].
    tune : float
        Bare betatron tune ``Q₀``.

    Returns
    -------
    float
        Threshold coherent kick strength ``κ_th`` [rad/m].
    """
    return 2.0 * sigma_Q**2 / (beta * abs(np.sin(2.0 * np.pi * tune)))


def _bbr_wake(dz: NDArray, W0: float, z_w: float, Q: float) -> NDArray:
    """Transverse broad-band resonator (BBR) wake function (vectorized).

    Implements the exact inverse Fourier transform of the BBR impedance
    (Lee §4.1, theory doc §2.4):

    .. code-block:: text

        W_⊥(Δz) = W₀ e^{−Δz/z_w} × [cos(ω̄ Δz/c) + (α_d/ω̄) sin(ω̄ Δz/c)]

    where ``α_d = c/z_w`` is the damping rate and
    ``ω̄/c = sqrt(4Q²−1)/z_w`` is the damped oscillation wave-number
    (purely real for ``Q > ½``).

    Special cases:

    * ``Q = 0`` or ``Q < ½``:  pure-exponential / over-damped (no sign change).
    * ``Q = ½``:  critically damped, ``W = W₀(1 + Δz/z_w) e^{−Δz/z_w}``.
    * ``Q > ½``:  under-damped, oscillatory wake with sign reversals.

    Parameters
    ----------
    dz : array-like
        Longitudinal separation(s) Δz ≥ 0 [m] (head − tail).
    W0 : float
        Peak wake amplitude (at ``Δz = 0``).
    z_w : float
        Wake decay length [m] (``= 2Qc/ω_r``).
    Q : float
        BBR quality factor.  ``Q = 0`` → pure exponential (code default).

    Returns
    -------
    ndarray
        Wake values at each ``dz``.
    """
    dz = np.asarray(dz, dtype=float)
    base = W0 * np.exp(-dz / z_w)
    if Q == 0.0:
        return base
    alpha_d = 1.0 / z_w                    # damping rate [1/m]
    omega_r = 2.0 * Q * alpha_d             # ω_r/c [1/m]
    disc = omega_r**2 - alpha_d**2          # α_d²(4Q²−1)
    if disc > 0.0:                          # Q > ½: underdamped
        omega_bar = np.sqrt(disc)
        return base * (np.cos(omega_bar * dz)
                       + (alpha_d / omega_bar) * np.sin(omega_bar * dz))
    elif disc == 0.0:                       # Q = ½: critically damped
        return base * (1.0 + alpha_d * dz)
    else:                                   # Q < ½: overdamped
        Omega = np.sqrt(-disc)
        return base * (np.cosh(Omega * dz)
                       + (alpha_d / Omega) * np.sinh(Omega * dz))


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------


@dataclass
class CollectiveParams:
    """Parameters controlling the collective-instability model.

    Parameters
    ----------
    mode : {'transverse', 'headtail'}
        Which collective model to apply each turn.
    kappa : float
        **Transverse mode only.**  Coherent kick strength κ_coh [rad/m].
        The kick is Δx' = κ_coh · x̄_{n-1} (one-turn delay), which places
        the drive partly in quadrature with the betatron oscillation and
        allows true exponential growth above the Landau-damping threshold.
        Effective growth rate per turn: μ ≈ κ · β · sin²(2πQ) / 2.
        Approximate Landau threshold: κ_th ≈ σ_Q / (β · sin²(2πQ)).
    wake_strength : float
        **Head-tail mode only.**  Peak wake amplitude W₀ (dimensionless
        effective kick per unit centroid offset per slice).
    wake_range : float
        **Head-tail mode only.**  Wake decay length z_w [m].  Default 10 mm.
    n_slices : int
        Number of longitudinal slices for the head-tail model.
    sigma_z : float
        RMS bunch length σ_z [m].
    x0_offset : float
        Initial centroid displacement [m] added to all particles to seed
        the coherent mode.
    synchrotron_tune : float
        **Head-tail mode.**  Fractional synchrotron tune Q_s (default 0 =
        no synchrotron motion).  When non-zero, the longitudinal coordinates
        ``(z, δ)`` are rotated by the exact linearised synchrotron map each
        turn (Lee §3.2), making the head-tail model physically complete and
        enabling TMCI-threshold studies.
    slip_factor : float
        Slip factor η = α_c − 1/γ².  Positive above transition.  Defines
        the longitudinal beta function β_s = ηC/(2πQ_s) [m].
    circumference : float
        Ring circumference C [m].  Together with slip_factor and
        synchrotron_tune determines the longitudinal phase-space scale.
    resonator_Q : float
        **Head-tail mode.**  BBR quality factor Q for the transverse
        impedance model (Lee §4.1).  Q = 0 (default) → pure exponential
        wake W₀ exp(−Δz/z_w), valid for Q_resonator ≪ 1.  Q ≥ 0.5 →
        oscillatory wake that changes sign at half the resonator wavelength.
    """

    mode: str = "transverse"
    kappa: float = 0.0
    wake_strength: float = 0.0
    wake_range: float = 0.01
    n_slices: int = 20
    sigma_z: float = 0.01
    x0_offset: float = 2e-3
    # --- Synchrotron oscillations (Lee §3.2, §4.2) --------------------------
    synchrotron_tune: float = 0.0
    """Fractional synchrotron tune Q_s.  0 disables longitudinal motion."""
    slip_factor: float = 1e-3
    """Slip factor η = α_c − 1/γ².  Active only when synchrotron_tune > 0."""
    circumference: float = 100.0
    """Ring circumference C [m].  Active only when synchrotron_tune > 0."""
    # --- BBR impedance quality factor (Lee §4.1) ----------------------------
    resonator_Q: float = 0.0
    """BBR quality factor.  0 → pure exponential (default).  >0 → oscillatory."""


# ---------------------------------------------------------------------------
# Collective tracker
# ---------------------------------------------------------------------------


class CollectiveRing:
    """Monte Carlo particle tracker with collective-instability kicks.

    The single-particle dynamics are identical to ``StorageRing``.  After
    sampling the initial beam, an ``x0_offset`` is added to every particle
    (centroid displacement) to seed the coherent mode.  Each turn the
    appropriate collective kick is applied *before* the linear lattice map.

    Parameters
    ----------
    ring : RingParams
    beam : BeamParams
    collective : CollectiveParams
    """

    def __init__(
        self, ring: RingParams, beam: BeamParams, collective: CollectiveParams
    ) -> None:
        self.ring = ring
        self.beam = beam
        self.collective = collective

        rng = np.random.default_rng(beam.seed)
        self._x0, self._xp0, self._delta0 = self._sample_beam(rng)

        # Longitudinal positions (fixed — no synchrotron motion)
        self._z0 = rng.standard_normal(beam.n_particles) * collective.sigma_z

        # Seed centroid offset
        self._x0 += collective.x0_offset

    # ------------------------------------------------------------------

    def _sample_beam(
        self, rng: np.random.Generator
    ) -> tuple[NDArray, NDArray, NDArray]:
        n, r, eps = self.beam.n_particles, self.ring, self.beam.emittance
        u, v = rng.standard_normal(n), rng.standard_normal(n)
        x = np.sqrt(eps * r.beta) * u
        xp = np.sqrt(eps / r.beta) * (-r.alpha * u + v)
        delta = rng.standard_normal(n) * self.beam.momentum_spread
        return x, xp, delta

    # ------------------------------------------------------------------

    def track(
        self,
        n_turns: int,
        sample_turns: list[int] | None = None,
    ) -> "CollectiveResult":
        """Track all particles for *n_turns* turns with collective kicks.

        Returns
        -------
        CollectiveResult
        """
        if sample_turns is None:
            sample_turns = sorted({0, n_turns // 4, n_turns // 2, n_turns})
        sample_set = set(sample_turns)

        r, coll = self.ring, self.collective
        n = self.beam.n_particles

        x = self._x0.copy()
        xp = self._xp0.copy()
        delta = self._delta0.copy()
        z = self._z0.copy()
        alive = np.ones(n, dtype=bool)

        emittance_arr = np.zeros(n_turns + 1)
        centroid_x = np.zeros(n_turns + 1)
        survival_arr = np.zeros(n_turns + 1)
        snapshots: dict[int, tuple[NDArray, NDArray]] = {}

        def _record(turn: int) -> None:
            mask = alive
            emittance_arr[turn] = rms_emittance(x[mask], xp[mask])
            centroid_x[turn] = float(np.mean(x[mask])) if mask.any() else 0.0
            survival_arr[turn] = mask.sum() / n
            if turn in sample_set:
                snapshots[turn] = (x[mask].copy(), xp[mask].copy())

        _record(0)

        # One-turn delayed centroid for the transverse resistive-wake model.
        # A real kick ∝ x̄_current shifts the tune (always stable); using
        # x̄_{n-1} places the drive partly in quadrature so that the amplitude
        # grows at rate μ ≈ κ β sin²(2πQ)/2 — overcoming Landau damping
        # once κ exceeds the threshold κ_th ≈ σ_Q / (β sin²(2πQ)).
        prev_x_bar = float(np.mean(x[alive])) if alive.any() else 0.0

        for turn in range(1, n_turns + 1):
            # ── collective kick using one-turn delayed centroid ───────────
            if coll.mode == "transverse":
                self._rigid_bunch_kick(x, xp, alive, prev_x_bar, coll)
            elif coll.mode == "headtail":
                self._headtail_kick(x, xp, z, alive, coll)

            # ── linear one-turn map (chromaticity included) ───────────────
            phi = 2.0 * np.pi * (r.tune + r.chromaticity * delta)
            c, s = np.cos(phi), np.sin(phi)
            x_new = (c + r.alpha * s) * x + r.beta * s * xp
            xp_new = -r.gamma * s * x + (c - r.alpha * s) * xp
            x, xp = x_new, xp_new

            # ── optional sextupole kick ───────────────────────────────────
            if r.sextupole_strength != 0.0:
                xp -= r.sextupole_strength * x**2

            # ── synchrotron oscillations (Lee §3.2) ───────────────────────
            # Exact linearised (z, δ) rotation at tune Q_s.
            # β_s = ηC/(2πQ_s) is the longitudinal beta function.
            # Updating δ each turn couples the chromatic tune shift
            # Q₀ + ξδ to the longitudinal phase-space motion, which is
            # the essential physics of the head-tail instability.
            if coll.synchrotron_tune > 0.0:
                phi_s = 2.0 * np.pi * coll.synchrotron_tune
                beta_s = (coll.slip_factor * coll.circumference
                          / (2.0 * np.pi * coll.synchrotron_tune))
                cos_s, sin_s = np.cos(phi_s), np.sin(phi_s)
                z_new = cos_s * z - beta_s * sin_s * delta
                delta = (sin_s / beta_s) * z + cos_s * delta
                z = z_new

            # ── aperture ──────────────────────────────────────────────────
            alive &= np.abs(x) <= r.aperture
            x[~alive] = 0.0
            xp[~alive] = 0.0

            # Update one-turn delayed centroid (only surviving particles
            # contribute to the wake field in the next turn).
            prev_x_bar = float(np.mean(x[alive])) if alive.any() else 0.0

            _record(turn)

        return CollectiveResult(
            turns=np.arange(n_turns + 1),
            emittance=emittance_arr,
            centroid_x=centroid_x,
            survival=survival_arr,
            snapshots=snapshots,
            ring=r,
            beam=self.beam,
            collective=coll,
        )

    # ------------------------------------------------------------------
    # Collective kick implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _rigid_bunch_kick(
        x: NDArray,
        xp: NDArray,
        alive: NDArray,
        prev_x_bar: float,
        coll: CollectiveParams,
    ) -> None:
        """Coherent kick Δx'_i = κ_coh · x̄_{n-1}  (one-turn delayed, rigid-bunch).

        Using the *previous-turn* centroid places the drive partly in
        quadrature with the current betatron oscillation.  This is equivalent
        to a narrow resistive impedance and produces exponential growth of the
        centroid amplitude once κ exceeds the Landau-damping threshold.
        """
        if not alive.any():
            return
        xp[alive] += coll.kappa * prev_x_bar

    @staticmethod
    def _headtail_kick(
        x: NDArray,
        xp: NDArray,
        z: NDArray,
        alive: NDArray,
        coll: CollectiveParams,
    ) -> None:
        """Sliced wake kick — head slices drive tail slices (vectorised).

        Convention: larger z = head (arrives first at impedance source).
        Wake model selected by ``coll.resonator_Q``:

        * ``resonator_Q = 0`` (default): W(Δz) = W₀ exp(−Δz/z_w).
        * ``resonator_Q > 0``: full BBR oscillatory wake (Lee §4.1).

        The wake matrix ``W[j, i] = W(z_i − z_j)`` for ``z_i ≥ z_j``
        (upper-triangular in z-order) is built in a single numpy
        outer-difference call, replacing the O(K²) Python double loop.
        The kick per slice is then a matrix–vector product:

        .. code-block:: text

            kick[j] = Σ_i  W[j,i] · x̄_i
        """
        if not alive.any():
            return

        z_live = z[alive]
        x_live = x[alive]

        z_min = z_live.min() - 1e-12
        z_max = z_live.max() + 1e-12
        edges = np.linspace(z_min, z_max, coll.n_slices + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Centroid per slice (empty slices → 0)
        slice_idx = np.searchsorted(edges[1:], z_live)   # 0…K-1
        x_bar_slice = np.zeros(coll.n_slices)
        for k in range(coll.n_slices):
            mask_k = slice_idx == k
            if mask_k.any():
                x_bar_slice[k] = float(np.mean(x_live[mask_k]))

        # Vectorised wake matrix: dz_matrix[j, i] = centers[i] − centers[j]
        # Positive entry means slice i is ahead of slice j.
        dz_matrix = centers[np.newaxis, :] - centers[:, np.newaxis]  # (K, K)

        # Evaluate wake only for dz ≥ 0 (causal: head drives tail)
        W_matrix = np.where(
            dz_matrix >= 0.0,
            _bbr_wake(np.maximum(dz_matrix, 0.0),
                      coll.wake_strength, coll.wake_range, coll.resonator_Q),
            0.0,
        )

        # kick[j] = Σ_i W[j,i] · x̄_i  (matrix–vector product)
        kick_per_slice = W_matrix @ x_bar_slice

        xp[alive] += kick_per_slice[slice_idx]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CollectiveResult:
    """Turn-by-turn output from a collective-instability tracking run.

    Attributes
    ----------
    centroid_x : ndarray, shape (n_turns+1,)
        Mean x-position of surviving particles per turn [m].  The key
        observable for coherent instability: should decay (Landau-stable)
        or grow exponentially (unstable).
    emittance : ndarray, shape (n_turns+1,)
        RMS geometric emittance [m·rad].
    survival : ndarray, shape (n_turns+1,)
        Fraction of surviving particles.
    snapshots : dict[int, tuple[ndarray, ndarray]]
        (x, x') phase-space arrays at selected turns.
    """

    turns: NDArray
    emittance: NDArray
    centroid_x: NDArray
    survival: NDArray
    snapshots: dict[int, tuple[NDArray, NDArray]]
    ring: RingParams
    beam: BeamParams
    collective: CollectiveParams

    # ------------------------------------------------------------------
    # Analytic diagnostics (Lee §3.3, §4.2)
    # ------------------------------------------------------------------

    @property
    def theoretical_growth_rate(self) -> float:
        """Analytical growth rate μ_grow = (κ β / 2)|sin(2π Q₀)| [turns⁻¹].

        From the action-angle derivation (Lee §3.3 / theory doc eq. 3.5).
        Valid for the rigid-bunch transverse mode; returns 0 for head-tail.
        """
        if self.collective.mode != "transverse":
            return 0.0
        return (self.collective.kappa * self.ring.beta / 2.0) * abs(
            np.sin(2.0 * np.pi * self.ring.tune)
        )

    @property
    def theoretical_decoherence_time(self) -> float:
        """Decoherence time τ_deco = 1/(2π√2 σ_Q) [turns] (Lee §4.2, eq. 4.2).

        The Gaussian centroid envelope falls to 1/e after this many turns.
        Returns ``inf`` when chromaticity or momentum spread is zero.
        """
        sigma_Q = abs(self.ring.chromaticity) * self.beam.momentum_spread
        if sigma_Q == 0.0:
            return np.inf
        return 1.0 / (2.0 * np.pi * np.sqrt(2.0) * sigma_Q)

    @property
    def sacherer_kth(self) -> float:
        """Sacherer Landau threshold κ_th in the N→∞ Vlasov limit (Lee §4.3).

        Computed from :func:`sacherer_threshold` using the ring and beam
        parameters stored in this result.
        """
        sigma_Q = abs(self.ring.chromaticity) * self.beam.momentum_spread
        return sacherer_threshold(sigma_Q, self.ring.beta, self.ring.tune)
