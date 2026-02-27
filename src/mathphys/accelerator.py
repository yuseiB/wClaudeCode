"""Accelerator physics — single-particle dynamics.

Coordinate system (6D phase space)
------------------------------------
Vector  v = (x, px, y, py, delta, l)

  x      [m]   horizontal displacement from reference orbit
  px     [rad]  horizontal divergence  x' = dx/ds
  y      [m]   vertical displacement
  py     [rad]  vertical divergence    y' = dy/ds
  delta  [ ]   fractional momentum deviation  (p − p₀)/p₀
  l      [m]   longitudinal path-length difference  (l = −c·τ)

Sign convention: a positive δ means higher momentum than reference.
The longitudinal coordinate l decreases when the particle arrives early.

Transfer matrix ordering
------------------------
For a particle traversing elements E₀, E₁, …, E_{N−1} in sequence the
combined one-turn matrix is

    M = M_{N−1} · M_{N−2} · … · M₀

i.e. M₀ acts first (rightmost).  In code::

    M = np.eye(6)
    for e in elements:        # E₀, E₁, …
        M = e.matrix() @ M    # left-multiply: M₀ then M₁ ...

This file is part of the **mathphys** package.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
C_LIGHT: float = 2.998_792_458e8   # speed of light [m/s]


# ═══════════════════════════════════════════════════════════════════════════════
#  Particle & Beam
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Particle:
    """Single particle in 6D phase space."""

    x: float = 0.0
    px: float = 0.0
    y: float = 0.0
    py: float = 0.0
    delta: float = 0.0
    l: float = 0.0

    @property
    def state(self) -> np.ndarray:
        """Return coordinate vector of shape (6,)."""
        return np.array([self.x, self.px, self.y, self.py, self.delta, self.l])

    @classmethod
    def from_state(cls, s: np.ndarray) -> "Particle":
        return cls(x=float(s[0]), px=float(s[1]), y=float(s[2]),
                   py=float(s[3]), delta=float(s[4]), l=float(s[5]))


class Beam:
    """Collection of N particles in 6D phase space.

    Parameters
    ----------
    particles:  array of shape (N, 6) with coordinates (x, px, y, py, δ, l).
    energy0:    reference particle energy [GeV].
    """

    def __init__(self, particles: np.ndarray, energy0: float = 1.0) -> None:
        if particles.ndim != 2 or particles.shape[1] != 6:
            raise ValueError("particles must have shape (N, 6)")
        self.particles = np.asarray(particles, dtype=float)
        self.energy0 = float(energy0)

    @property
    def n_particles(self) -> int:
        return len(self.particles)

    @staticmethod
    def gaussian(
        n: int,
        emittance_x: float,
        beta_x: float,
        alpha_x: float,
        emittance_y: float,
        beta_y: float,
        alpha_y: float,
        delta_spread: float = 0.0,
        energy0: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> "Beam":
        """Generate a Gaussian beam from Courant-Snyder optics parameters.

        The transverse phase-space density is matched to the Courant-Snyder
        ellipse  ε = γx² + 2αxp + βp²  where γ = (1+α²)/β.

        Parameters
        ----------
        n:             Number of macro-particles.
        emittance_x/y: Geometric 1-σ emittance [m·rad].
        beta_x/y:      β-functions at the injection point [m].
        alpha_x/y:     α-functions at the injection point.
        delta_spread:  1-σ momentum spread (δ, dimensionless).
        energy0:       Reference energy [GeV].
        rng:           NumPy random generator (default: seed 42).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        gamma_x = (1.0 + alpha_x**2) / beta_x
        gamma_y = (1.0 + alpha_y**2) / beta_y

        # Covariance matrices of the phase-space distribution
        sig_x = np.array([[beta_x, -alpha_x], [-alpha_x, gamma_x]]) * emittance_x
        sig_y = np.array([[beta_y, -alpha_y], [-alpha_y, gamma_y]]) * emittance_y

        Lx = np.linalg.cholesky(sig_x)
        Ly = np.linalg.cholesky(sig_y)

        z = rng.standard_normal((n, 6))
        pts = np.zeros((n, 6))
        pts[:, 0:2] = z[:, 0:2] @ Lx.T
        pts[:, 2:4] = z[:, 2:4] @ Ly.T
        pts[:, 4] = delta_spread * z[:, 4]
        # l starts at 0

        return Beam(pts, energy0)

    def emittance(self, plane: str = "x") -> float:
        """RMS geometric emittance ε = √(σ_u² σ_pu² − σ_{u,pu}²) [m·rad].

        Parameters
        ----------
        plane: 'x', 'y', or 'l' (longitudinal).
        """
        if plane == "x":
            c = self.particles[:, 0:2]
        elif plane == "y":
            c = self.particles[:, 2:4]
        elif plane == "l":
            c = self.particles[:, 4:6]
        else:
            raise ValueError(f"Unknown plane '{plane}'. Choose 'x', 'y', or 'l'.")
        cov = np.cov(c.T)
        return float(np.sqrt(max(0.0, np.linalg.det(cov))))

    def centroid(self) -> np.ndarray:
        """Return phase-space centroid, shape (6,)."""
        return np.mean(self.particles, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Abstract Element
# ═══════════════════════════════════════════════════════════════════════════════

class Element(ABC):
    """Base class for accelerator lattice elements.

    Subclasses must implement:
    - ``length`` (property)
    - ``matrix(delta)`` → 6×6 transfer matrix
    - ``track(particles)`` → (N, 6) → (N, 6)
    """

    def __init__(self, name: str = "") -> None:
        self.name = str(name)

    @property
    @abstractmethod
    def length(self) -> float:
        """Element length [m]."""

    @abstractmethod
    def matrix(self, delta: float = 0.0) -> np.ndarray:
        """Return the 6×6 linear transfer matrix for momentum offset *delta*."""

    @abstractmethod
    def track(self, particles: np.ndarray) -> np.ndarray:
        """Propagate a batch of N particles.

        Parameters
        ----------
        particles: shape (N, 6) — input coordinates.

        Returns
        -------
        np.ndarray of shape (N, 6) — output coordinates.
        """


# ═══════════════════════════════════════════════════════════════════════════════
#  Drift
# ═══════════════════════════════════════════════════════════════════════════════

class Drift(Element):
    """Free-space propagation.

    Transfer matrix (on-momentum, delta = 0):

        M = diag(1,1,1,1,1,1)  with  M[0,1] = M[2,3] = L

    Chromatic correction: the effective length for off-momentum particles is
    ``L_eff = L / (1 + delta)`` (thin-magnet approximation).
    """

    def __init__(self, length: float, name: str = "D") -> None:
        super().__init__(name)
        self._length = float(length)

    @property
    def length(self) -> float:
        return self._length

    def matrix(self, delta: float = 0.0) -> np.ndarray:
        L = self._length / (1.0 + delta)
        M = np.eye(6)
        M[0, 1] = L
        M[2, 3] = L
        return M

    def track(self, particles: np.ndarray) -> np.ndarray:
        out = particles.copy()
        d = particles[:, 4]
        L = self._length / (1.0 + d)
        out[:, 0] += L * particles[:, 1]
        out[:, 2] += L * particles[:, 3]
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Quadrupole
# ═══════════════════════════════════════════════════════════════════════════════

class Quadrupole(Element):
    """Magnetic quadrupole (thick lens).

    Parameters
    ----------
    length: Magnetic length [m].
    k1:     Normalized gradient  k₁ = eB'/p₀  [1/m²].
            k₁ > 0 → focusing in x, defocusing in y.
            k₁ < 0 → defocusing in x, focusing in y.

    Transfer matrices (2×2 sub-blocks):

    Focusing plane (k₁ > 0, x):
        φ = √k₁ · L
        M_f = [[cos φ,      sin φ / √k₁   ],
               [−√k₁ sin φ, cos φ          ]]

    Defocusing plane (k₁ > 0, y):
        M_d = [[cosh φ,    sinh φ / √k₁  ],
               [√k₁ sinh φ, cosh φ        ]]
    """

    def __init__(self, length: float, k1: float, name: str = "Q") -> None:
        super().__init__(name)
        self._length = float(length)
        self.k1 = float(k1)

    @property
    def length(self) -> float:
        return self._length

    def _sub_matrices(self, k_eff: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (M_x, M_y) 2×2 sub-matrices for gradient k_eff."""
        L = self._length
        if abs(k_eff) < 1e-12:
            M_f = np.array([[1.0, L], [0.0, 1.0]])
            return M_f, M_f.copy()

        kp = math.sqrt(abs(k_eff))
        phi = kp * L
        sp, cp = math.sin(phi), math.cos(phi)
        sh, ch = math.sinh(phi), math.cosh(phi)

        if k_eff > 0:
            M_f = np.array([[cp, sp / kp], [-kp * sp, cp]])          # x (focusing)
            M_d = np.array([[ch, sh / kp], [kp * sh, ch]])            # y (defocusing)
        else:
            M_f = np.array([[ch, sh / kp], [kp * sh, ch]])            # x (defocusing)
            M_d = np.array([[cp, sp / kp], [-kp * sp, cp]])           # y (focusing)

        return M_f, M_d

    def matrix(self, delta: float = 0.0) -> np.ndarray:
        k_eff = self.k1 / (1.0 + delta)
        Mx, My = self._sub_matrices(k_eff)
        M = np.eye(6)
        M[0:2, 0:2] = Mx
        M[2:4, 2:4] = My
        return M

    def track(self, particles: np.ndarray) -> np.ndarray:
        out = particles.copy()
        for i in range(len(particles)):
            d = particles[i, 4]
            Mx, My = self._sub_matrices(self.k1 / (1.0 + d))
            out[i, 0:2] = Mx @ particles[i, 0:2]
            out[i, 2:4] = My @ particles[i, 2:4]
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Dipole
# ═══════════════════════════════════════════════════════════════════════════════

class Dipole(Element):
    """Sector bending magnet (constant field, variable radius).

    Parameters
    ----------
    length: Arc length [m].
    angle:  Total bending angle [rad].
    e1:     Entrance edge angle [rad] (for edge focusing).
    e2:     Exit edge angle [rad].

    Transfer matrix (on-momentum)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Let  φ = angle,  ρ = length/angle  (bending radius).

    x–px block (horizontal, sector geometry)::

        M_x = [[cos φ,      ρ sin φ  ],
               [−sin φ/ρ,   cos φ    ]]

    y–py block (vertical, no bending)::

        M_y = [[1, L],
               [0, 1]]

    Dispersion column (coupling to δ)::

        M[0, 4] = ρ(1 − cos φ)      M[1, 4] = sin φ

    Path-length row (coupling to x, px, δ)::

        M[5, 0] = −sin φ             M[5, 1] = −ρ(1 − cos φ)
        M[5, 4] = ρ(sin φ − φ)      [first-order path length change]

    Edge kicks (thin-lens, applied at entrance and exit)::

        M_edge[1, 0] = +tan(e)/ρ    (x: focusing)
        M_edge[3, 2] = −tan(e)/ρ    (y: defocusing)
    """

    def __init__(
        self,
        length: float,
        angle: float,
        e1: float = 0.0,
        e2: float = 0.0,
        name: str = "B",
    ) -> None:
        super().__init__(name)
        self._length = float(length)
        self.angle = float(angle)
        self.e1 = float(e1)
        self.e2 = float(e2)

    @property
    def length(self) -> float:
        return self._length

    @property
    def rho(self) -> float:
        """Bending radius ρ = L / φ  [m]."""
        if abs(self.angle) < 1e-15:
            return math.inf
        return self._length / self.angle

    def _edge_matrix(self, edge_angle: float) -> np.ndarray:
        """6×6 thin-lens edge-focusing matrix."""
        M = np.eye(6)
        rho = self.rho
        if abs(edge_angle) > 1e-15 and rho < 1e15:
            foc = math.tan(edge_angle) / rho
            M[1, 0] = foc    # x: focusing at entrance
            M[3, 2] = -foc   # y: defocusing at entrance
        return M

    def matrix(self, delta: float = 0.0) -> np.ndarray:
        L = self._length
        phi = self.angle
        rho = self.rho

        c, s = math.cos(phi), math.sin(phi)

        M_body = np.eye(6)
        # x–px block
        M_body[0, 0] = c
        M_body[0, 1] = rho * s
        M_body[1, 0] = -s / rho
        M_body[1, 1] = c
        # Dispersion coupling
        M_body[0, 4] = rho * (1.0 - c)
        M_body[1, 4] = s
        # y–py block (pure drift)
        M_body[2, 3] = L
        # Path-length row
        M_body[5, 0] = -s
        M_body[5, 1] = -rho * (1.0 - c)
        M_body[5, 4] = rho * (s - phi)

        M_e1 = self._edge_matrix(self.e1)
        M_e2 = self._edge_matrix(self.e2)

        return M_e2 @ M_body @ M_e1

    def track(self, particles: np.ndarray) -> np.ndarray:
        # Use the on-momentum matrix for all particles.
        # Full chromatic dipole tracking requires an extended Hamiltonian map;
        # the linear matrix is sufficient for Twiss and Poincaré studies.
        M = self.matrix(delta=0.0)
        return (M @ particles.T).T


# ═══════════════════════════════════════════════════════════════════════════════
#  Sextupole
# ═══════════════════════════════════════════════════════════════════════════════

class Sextupole(Element):
    """Sextupole magnet — thin-lens representation.

    Uses the split-operator map:  Drift(L/2) → thin kick → Drift(L/2).

    Thin kick (integrated over the full length L):

        Δpx = −k₂ L (x² − y²) / 2
        Δpy = +k₂ L · x · y

    Parameters
    ----------
    length: Physical length [m]  (used for path-length bookkeeping).
    k2:     Sextupole strength [1/m³]  (= ∂²By/∂x² · e/p₀).
    """

    def __init__(self, length: float, k2: float, name: str = "S") -> None:
        super().__init__(name)
        self._length = float(length)
        self.k2 = float(k2)

    @property
    def length(self) -> float:
        return self._length

    def matrix(self, delta: float = 0.0) -> np.ndarray:
        """Linear matrix = pure drift (sextupole is 2nd-order nonlinear)."""
        L = self._length / (1.0 + delta)
        M = np.eye(6)
        M[0, 1] = L
        M[2, 3] = L
        return M

    def track(self, particles: np.ndarray) -> np.ndarray:
        L = self._length
        k2L = self.k2 * L

        out = particles.copy()
        # Half drift
        d = out[:, 4]
        Lh = L / 2.0 / (1.0 + d)
        out[:, 0] += Lh * out[:, 1]
        out[:, 2] += Lh * out[:, 3]

        # Thin sextupole kick
        x, y = out[:, 0], out[:, 2]
        out[:, 1] -= k2L * (x**2 - y**2) / 2.0
        out[:, 3] += k2L * x * y

        # Half drift
        Lh2 = L / 2.0 / (1.0 + out[:, 4])
        out[:, 0] += Lh2 * out[:, 1]
        out[:, 2] += Lh2 * out[:, 3]

        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  RF Cavity
# ═══════════════════════════════════════════════════════════════════════════════

class RFCavity(Element):
    """Radio-frequency accelerating cavity (thin-lens map).

    Uses the split-operator map:  Drift(L/2) → RF kick → Drift(L/2).

    The cavity applies an energy change based on the particle's longitudinal
    position relative to the synchronous particle:

        Δδ = (eV / E₀) · [sin(φ_s − k_rf · l) − sin(φ_s)]

    where  k_rf = 2πf / c  is the RF wave number.

    Parameters
    ----------
    length:    Cavity length [m].
    voltage:   Peak voltage [V].
    frequency: RF frequency [Hz].
    phi_s:     Synchronous phase [rad]  (φ = 0 → pure accelerating/decelerating crest).
    energy0:   Reference particle energy [eV].
    """

    def __init__(
        self,
        length: float,
        voltage: float,
        frequency: float,
        phi_s: float,
        energy0: float = 1e9,
        name: str = "RF",
    ) -> None:
        super().__init__(name)
        self._length = float(length)
        self.voltage = float(voltage)
        self.frequency = float(frequency)
        self.phi_s = float(phi_s)
        self.energy0 = float(energy0)

    @property
    def length(self) -> float:
        return self._length

    def matrix(self, delta: float = 0.0) -> np.ndarray:
        """Linearised RF matrix for small-amplitude synchrotron oscillations."""
        L = self._length
        k_rf = 2.0 * math.pi * self.frequency / C_LIGHT
        V_norm = self.voltage / self.energy0
        M = np.eye(6)
        M[0, 1] = L / 2
        M[2, 3] = L / 2
        # δ–l coupling from linearised kick: Δδ/Δl ≈ V_norm·k_rf·cos(φ_s)·(-1)
        M[4, 5] = -V_norm * k_rf * math.cos(self.phi_s)
        return M

    def track(self, particles: np.ndarray) -> np.ndarray:
        L = self._length
        k_rf = 2.0 * math.pi * self.frequency / C_LIGHT
        V_norm = self.voltage / self.energy0

        out = particles.copy()
        # Half drift
        d = out[:, 4]
        Lh = L / 2.0 / (1.0 + d)
        out[:, 0] += Lh * out[:, 1]
        out[:, 2] += Lh * out[:, 3]

        # RF kick
        l = out[:, 5]
        out[:, 4] += V_norm * (np.sin(self.phi_s - k_rf * l) - math.sin(self.phi_s))

        # Half drift
        Lh2 = L / 2.0 / (1.0 + out[:, 4])
        out[:, 0] += Lh2 * out[:, 1]
        out[:, 2] += Lh2 * out[:, 3]

        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Marker
# ═══════════════════════════════════════════════════════════════════════════════

class Marker(Element):
    """Zero-length diagnostic element (passthrough)."""

    def __init__(self, name: str = "M") -> None:
        super().__init__(name)

    @property
    def length(self) -> float:
        return 0.0

    def matrix(self, delta: float = 0.0) -> np.ndarray:
        return np.eye(6)

    def track(self, particles: np.ndarray) -> np.ndarray:
        return particles.copy()


# ═══════════════════════════════════════════════════════════════════════════════
#  Lattice
# ═══════════════════════════════════════════════════════════════════════════════

class Lattice:
    """Sequence of accelerator elements forming a ring or beamline.

    Parameters
    ----------
    elements:  Ordered list of Element objects (one revolution = one period).
    n_repeats: Repeat the element list n_repeats times (default 1).
               Useful for multi-cell structures: ``make_fodo(n_cells=4)``
               internally sets n_repeats via the factory, but you can also do
               ``Lattice([cell_elements], n_repeats=4)``.
    """

    def __init__(
        self,
        elements: Sequence[Element],
        n_repeats: int = 1,
    ) -> None:
        self.elements: list[Element] = list(elements) * int(n_repeats)

    @property
    def circumference(self) -> float:
        """Total path length [m]."""
        return sum(e.length for e in self.elements)

    def s_positions(self) -> np.ndarray:
        """Return s-coordinate at the *exit* of every element, shape (N,)."""
        s = 0.0
        out = []
        for e in self.elements:
            s += e.length
            out.append(s)
        return np.array(out)

    # ── Linear optics ─────────────────────────────────────────────────────────

    def one_turn_matrix(self, delta: float = 0.0) -> np.ndarray:
        """Compute the one-turn 6×6 transfer matrix.

        The product is  M = M_{N-1} · … · M₀  (first element on the right).
        """
        M = np.eye(6)
        for e in self.elements:
            M = e.matrix(delta) @ M
        return M

    def twiss_at_start(self, delta: float = 0.0) -> dict:
        """Extract Twiss parameters and tunes from the one-turn matrix.

        Returns a dict with keys::

            beta_x, alpha_x, gamma_x, tune_x,
            beta_y, alpha_y, gamma_y, tune_y,
            Dx, Dpx

        Raises ValueError if the lattice is transversely unstable (|trace/2|>1).
        """
        M = self.one_turn_matrix(delta)
        return _twiss_from_one_turn(M)

    def twiss(self) -> dict:
        """Compute Courant-Snyder Twiss functions at every element exit.

        Returns a dict of 1-D arrays of length ``len(elements) + 1``:

            s, beta_x, alpha_x, gamma_x,
               beta_y, alpha_y, gamma_y,
               Dx, Dpx

        Index 0 corresponds to the ring entrance (= exit of last element).
        """
        t0 = self.twiss_at_start()

        bx,  ax  = t0["beta_x"],  t0["alpha_x"]
        by,  ay  = t0["beta_y"],  t0["alpha_y"]
        dx,  dpx = t0["Dx"],      t0["Dpx"]

        s_arr   = [0.0]
        beta_x  = [bx]; alpha_x = [ax]; gamma_x = [(1 + ax**2) / bx]
        beta_y  = [by]; alpha_y = [ay]; gamma_y = [(1 + ay**2) / by]
        Dx_arr  = [dx]; Dpx_arr = [dpx]

        s = 0.0
        for e in self.elements:
            M = e.matrix(delta=0.0)
            Mx, My = M[0:2, 0:2], M[2:4, 2:4]

            # Dispersion propagation: [Dx, Dpx] → M_x · [Dx, Dpx] + [M[0,4], M[1,4]]
            dx_new  = M[0, 0]*dx + M[0, 1]*dpx + M[0, 4]
            dpx_new = M[1, 0]*dx + M[1, 1]*dpx + M[1, 4]

            # Courant-Snyder propagation (beam-matrix transport)
            bx_new, ax_new = _propagate_twiss(bx, ax, Mx)
            by_new, ay_new = _propagate_twiss(by, ay, My)

            s += e.length
            bx, ax = bx_new, ax_new
            by, ay = by_new, ay_new
            dx, dpx = dx_new, dpx_new

            s_arr.append(s)
            beta_x.append(bx);  alpha_x.append(ax)
            gamma_x.append((1 + ax**2) / bx)
            beta_y.append(by);  alpha_y.append(ay)
            gamma_y.append((1 + ay**2) / by)
            Dx_arr.append(dx);  Dpx_arr.append(dpx)

        return dict(
            s       = np.array(s_arr),
            beta_x  = np.array(beta_x),
            alpha_x = np.array(alpha_x),
            gamma_x = np.array(gamma_x),
            beta_y  = np.array(beta_y),
            alpha_y = np.array(alpha_y),
            gamma_y = np.array(gamma_y),
            Dx      = np.array(Dx_arr),
            Dpx     = np.array(Dpx_arr),
        )

    def tune(self) -> tuple[float, float]:
        """Return fractional betatron tunes (Qx, Qy)."""
        t = self.twiss_at_start()
        return t["tune_x"], t["tune_y"]

    def chromaticity(self, ddelta: float = 1e-4) -> tuple[float, float]:
        """Numerical chromaticity  ξ_x = dQx/dδ,  ξ_y = dQy/dδ."""
        tp = self.twiss_at_start(delta=+ddelta)
        tm = self.twiss_at_start(delta=-ddelta)
        xi_x = (tp["tune_x"] - tm["tune_x"]) / (2.0 * ddelta)
        xi_y = (tp["tune_y"] - tm["tune_y"]) / (2.0 * ddelta)
        return xi_x, xi_y

    def momentum_compaction(self) -> float:
        """Momentum compaction factor  αc = (1/C) ∮ Dx/ρ ds.

        Only dipole elements contribute (ρ = length/angle).
        The dispersion Dx is averaged between element entrance and exit.
        """
        C = self.circumference
        if C < 1e-15:
            return 0.0
        t = self.twiss()
        alpha_c = 0.0
        for i, e in enumerate(self.elements):
            if isinstance(e, Dipole) and abs(e.angle) > 1e-15:
                Dx_avg = 0.5 * (t["Dx"][i] + t["Dx"][i + 1])
                alpha_c += Dx_avg * e.angle  # ∫ Dx/ρ ds ≈ Dx_avg * angle
        return alpha_c / C


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-turn tracking
# ═══════════════════════════════════════════════════════════════════════════════

def track(
    particles: np.ndarray,
    lattice: Lattice,
    n_turns: int,
    monitor_indices: list[int] | None = None,
) -> np.ndarray:
    """Track a batch of particles through a lattice for *n_turns* revolutions.

    Parameters
    ----------
    particles:       Initial coordinates, shape (N, 6).
    lattice:         Accelerator lattice.
    n_turns:         Number of complete revolutions.
    monitor_indices: Indices of elements *after which* coordinates are recorded.
                     ``None`` → record only after the last element (one sample
                     per turn, at the ring exit).

    Returns
    -------
    np.ndarray of shape  (n_turns + 1, n_monitors, N, 6).

    ``history[0]``  contains the *initial* state at every monitor.
    ``history[t+1]`` contains the state after *t+1* complete turns
    (or after passing the specified monitor element on turn t+1).
    """
    N = particles.shape[0]
    if monitor_indices is None:
        monitors = [len(lattice.elements) - 1]
    else:
        monitors = list(monitor_indices)
    n_mon = len(monitors)

    history = np.zeros((n_turns + 1, n_mon, N, 6))
    history[0] = particles[np.newaxis, :]   # broadcast initial state

    p = particles.copy()
    for turn in range(n_turns):
        for i_elem, elem in enumerate(lattice.elements):
            p = elem.track(p)
            if i_elem in monitors:
                m_idx = monitors.index(i_elem)
                history[turn + 1, m_idx] = p

    return history


# ═══════════════════════════════════════════════════════════════════════════════
#  Preset lattice factories
# ═══════════════════════════════════════════════════════════════════════════════

def make_fodo(
    Lq: float = 0.5,
    Ld: float = 2.0,
    k1: float | None = None,
    f: float | None = None,
    n_cells: int = 4,
    name_prefix: str = "",
) -> Lattice:
    """Build a FODO ring.

    One cell consists of::

        QF/2 → Drift(Ld) → QD → Drift(Ld) → QF/2

    with QF (k₁ > 0) focusing in x and QD (k₁ < 0) defocusing in x.

    Parameters
    ----------
    Lq:          Full quadrupole length [m].
    Ld:          Half-cell drift length [m].
    k1:          Quadrupole gradient [1/m²]  (takes precedence over *f*).
    f:           Thin-lens focal length [m]  (used when *k1* is None).
                 Defaults to ``2*(Lq+Ld)`` which gives ~90° phase advance.
    n_cells:     Number of FODO cells forming the ring.
    name_prefix: Optional string prepended to element names.
    """
    if k1 is None:
        if f is None:
            # Thin-lens approximation: 90° advance → f = L_cell / (2√2)
            L_cell = 2.0 * (Lq + Ld)
            f = L_cell / (2.0 * math.sqrt(2.0))
        k1 = 1.0 / (f * Lq)

    p = name_prefix
    cell: list[Element] = [
        Quadrupole(Lq / 2, +k1, name=f"{p}QF"),
        Drift(Ld,           name=f"{p}D1"),
        Quadrupole(Lq,     -k1, name=f"{p}QD"),
        Drift(Ld,           name=f"{p}D2"),
        Quadrupole(Lq / 2, +k1, name=f"{p}QF"),
    ]

    return Lattice(cell, n_repeats=n_cells)


def make_ring(
    n_bends: int = 8,
    Lbend: float = 3.0,
    Lq: float = 0.3,
    Ld: float = 0.8,
    k1: float | None = None,
) -> Lattice:
    """Build a simple alternating-gradient ring with sector dipoles.

    The ring has *n_bends* dipoles, each followed by a quadrupole doublet
    (QF–drift–QD–drift) for focusing.  The total bending angle is 2π.

    Parameters
    ----------
    n_bends: Number of dipole magnets (evenly distributed around the ring).
    Lbend:   Length of each sector dipole [m].
    Lq:      Quadrupole length [m].
    Ld:      Drift between quads and dipoles [m].
    k1:      Quadrupole gradient [1/m²].  Defaults to a value that produces
             a stable lattice for the given geometry.
    """
    bend_angle = 2.0 * math.pi / n_bends

    if k1 is None:
        # Scale k1 so the phase advance per cell is ~60°
        L_cell = Lbend + 2 * Lq + 2 * Ld
        f_approx = L_cell / (2.0 * math.sqrt(2.0)) * 1.5   # looser focus
        k1 = 1.0 / (f_approx * Lq)

    p = ""
    arc: list[Element] = [
        Dipole(Lbend, bend_angle, name=f"{p}B"),
        Drift(Ld,                 name=f"{p}Da"),
        Quadrupole(Lq, +k1,       name=f"{p}QF"),
        Drift(Ld,                 name=f"{p}Db"),
        Quadrupole(Lq, -k1,       name=f"{p}QD"),
        Drift(Ld,                 name=f"{p}Dc"),
    ]

    return Lattice(arc, n_repeats=n_bends)


# ═══════════════════════════════════════════════════════════════════════════════
#  Private helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _propagate_twiss(
    beta: float, alpha: float, M2: np.ndarray
) -> tuple[float, float]:
    """Propagate Courant-Snyder parameters through a 2×2 matrix M2.

    The beam sigma-matrix  Σ = ε [[β, −α], [−α, γ]]  transforms as
    Σ_out = M · Σ_in · Mᵀ.  Using γ = (1+α²)/β:

        β_out = m₁₁² β − 2 m₁₁ m₁₂ α + m₁₂² γ
        α_out = −m₁₁ m₂₁ β + (m₁₁ m₂₂ + m₁₂ m₂₁) α − m₁₂ m₂₂ γ
    """
    m11, m12 = M2[0, 0], M2[0, 1]
    m21, m22 = M2[1, 0], M2[1, 1]
    g = (1.0 + alpha**2) / beta   # gamma_CS

    beta_new  = m11**2 * beta - 2.0*m11*m12*alpha + m12**2 * g
    alpha_new = -m11*m21*beta + (m11*m22 + m12*m21)*alpha - m12*m22*g
    return beta_new, alpha_new


def _twiss_from_one_turn(M: np.ndarray) -> dict:
    """Extract Twiss parameters and fractional tunes from a one-turn 6×6 matrix.

    Raises
    ------
    ValueError if |trace/2| > 1  (unstable motion).
    """
    result: dict = {}

    for plane, (i0, key) in [("x", (0, "x")), ("y", (2, "y"))]:
        M2 = M[i0:i0+2, i0:i0+2]
        trace2 = (M2[0, 0] + M2[1, 1]) / 2.0
        if abs(trace2) > 1.0:
            raise ValueError(
                f"{plane.upper()}-motion is unstable: "
                f"|cos μ| = {abs(trace2):.6f} > 1  (|trace/2| > 1)"
            )
        mu = math.acos(trace2)  # in [0, π]

        # Determine correct branch of the tune.
        # The off-diagonal element M₁₂ = β · sin(2πQ).
        # sin(2πQ) > 0  for Q ∈ (0, 0.5)  → acos gives the right μ.
        # sin(2πQ) < 0  for Q ∈ (0.5, 1)  → the true advance is 2π − μ.
        if M2[0, 1] < 0.0:
            mu = 2.0 * math.pi - mu   # now μ ∈ (π, 2π)

        sin_mu = math.sin(mu)   # negative for μ > π; that's intentional
        Q = mu / (2.0 * math.pi)
        beta  =  M2[0, 1] / sin_mu                        # always positive
        alpha = (M2[0, 0] - M2[1, 1]) / (2.0 * sin_mu)
        gamma = (1.0 + alpha**2) / beta
        result[f"tune_{key}"]  = Q
        result[f"beta_{key}"]  = beta
        result[f"alpha_{key}"] = alpha
        result[f"gamma_{key}"] = gamma

    # Dispersion from  (I − M_2x2) [Dx, Dpx]ᵀ = [M[0,4], M[1,4]]ᵀ
    Mx2 = M[0:2, 0:2]
    A   = np.eye(2) - Mx2
    det = A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]
    if abs(det) < 1e-14:
        result["Dx"] = 0.0;  result["Dpx"] = 0.0
    else:
        d_vec = np.linalg.solve(A, M[0:2, 4])
        result["Dx"]  = float(d_vec[0])
        result["Dpx"] = float(d_vec[1])

    return result
