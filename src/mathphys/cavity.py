"""
cavity.py — Electromagnetic Cavity Resonance Modes (Analytical Solutions)

PEC (Perfect Electric Conductor) cavities:
  - RectangularCavity : a × b × d
  - CylindricalCavity : radius R, height L
  - SphericalCavity   : radius R

All modes satisfy: tangential E = 0 at conducting walls.
Coordinate systems: rectangular (x,y,z), cylindrical (ρ,φ,z), spherical (r,θ,φ).

Physical constants (SI):
  c  = 2.99792458e8 m/s  (speed of light)
  μ₀ = 4π×10⁻⁷ H/m
  ε₀ = 1/(μ₀c²) F/m
  η₀ = μ₀c ≈ 376.73 Ω  (free-space wave impedance)

References:
  Pozar, "Microwave Engineering", 4th ed., ch. 6
  Griffiths, "Introduction to Electrodynamics", 4th ed., ch. 9
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.special import jv, jvp, jn_zeros, jnp_zeros, spherical_jn
from scipy.optimize import brentq
from typing import NamedTuple

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT = 2.99792458e8          # m/s
MU0     = 4e-7 * np.pi          # H/m
EPS0    = 1.0 / (MU0 * C_LIGHT**2)  # F/m
ETA0    = MU0 * C_LIGHT         # Ω  ≈ 376.73


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class EMField(NamedTuple):
    """Six-component EM field vector (Ex,Ey,Ez,Hx,Hy,Hz) at an array of points."""
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    Hx: np.ndarray
    Hy: np.ndarray
    Hz: np.ndarray

    @property
    def E_magnitude(self) -> np.ndarray:
        return np.sqrt(self.Ex**2 + self.Ey**2 + self.Ez**2)

    @property
    def H_magnitude(self) -> np.ndarray:
        return np.sqrt(self.Hx**2 + self.Hy**2 + self.Hz**2)


# ─────────────────────────────────────────────────────────────────────────────
# Rectangular Cavity
# ─────────────────────────────────────────────────────────────────────────────

class RectangularCavityMode:
    """
    TE_mnp or TM_mnp mode in a rectangular PEC cavity (0 ≤ x ≤ a, 0 ≤ y ≤ b, 0 ≤ z ≤ d).

    TE_mnp  (E_z = 0): m,n ≥ 0 (not both 0), p ≥ 1
    TM_mnp  (H_z = 0): m,n ≥ 1, p ≥ 0

    Resonant frequency: f = (c/2) √( (m/a)² + (n/b)² + (p/d)² )
    """

    def __init__(self, a: float, b: float, d: float,
                 m: int, n: int, p: int,
                 mode_type: str = 'TE') -> None:
        """
        Parameters
        ----------
        a, b, d    : cavity dimensions (m)
        m, n, p    : mode indices (non-negative integers)
        mode_type  : 'TE' or 'TM'
        """
        self.a, self.b, self.d = a, b, d
        self.m, self.n, self.p = m, n, p
        self.mode_type = mode_type.upper()
        self._validate()
        self._precompute()

    def _validate(self) -> None:
        t = self.mode_type
        m, n, p = self.m, self.n, self.p
        if t == 'TE':
            if m == 0 and n == 0:
                raise ValueError("TE: m and n cannot both be 0")
            if p < 1:
                raise ValueError("TE: p must be ≥ 1")
        elif t == 'TM':
            if m < 1 or n < 1:
                raise ValueError("TM: m and n must be ≥ 1")
            if p < 0:
                raise ValueError("TM: p must be ≥ 0")
        else:
            raise ValueError(f"mode_type must be 'TE' or 'TM', got {t!r}")

    def _precompute(self) -> None:
        m, n, p = self.m, self.n, self.p
        a, b, d = self.a, self.b, self.d
        self.kx = m * np.pi / a
        self.ky = n * np.pi / b
        self.kz = p * np.pi / d
        self.k  = np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)
        self.gamma2 = self.kx**2 + self.ky**2  # transverse wavenumber squared
        self.omega  = C_LIGHT * self.k

    @property
    def resonant_frequency(self) -> float:
        """Resonant frequency in Hz."""
        return self.omega / (2 * np.pi)

    def fields(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
               phase: float = 0.0) -> EMField:
        """
        Compute EM fields at grid points (x, y, z).

        For a cavity mode, the time dependence is:
          E(r,t) = E_amp(r) · cos(ωt + phase)
          H(r,t) = H_amp(r) · sin(ωt + phase)   [E and H are 90° out of phase]

        Returns the spatial amplitude patterns at the given phase angle.

        Parameters
        ----------
        x, y, z : array_like  (can be meshgrid arrays)
        phase   : ωt (radians)
        """
        kx, ky, kz = self.kx, self.ky, self.kz
        g2 = self.gamma2
        ω  = self.omega

        ct, st = np.cos(phase), np.sin(phase)

        if self.mode_type == 'TE':
            # H_z = H₀ cos(kx·x) cos(ky·y) sin(kz·z) · sin(ωt)
            # E ~ cos(ωt),  H ~ sin(ωt)
            Cx  = np.cos(kx * x)
            Sx  = np.sin(kx * x)
            Cy  = np.cos(ky * y)
            Sy  = np.sin(ky * y)
            Cz  = np.cos(kz * z)
            Sz  = np.sin(kz * z)

            if g2 > 0:
                Ex = ( ω * MU0 * ky / g2) * Cx * Sy * Sz * ct
                Ey = (-ω * MU0 * kx / g2) * Sx * Cy * Sz * ct
            else:
                Ex = np.zeros_like(x, dtype=float)
                Ey = np.zeros_like(x, dtype=float)
            Ez = np.zeros_like(x, dtype=float)

            Hx = -(kx * kz / g2) * Sx * Cy * Cz * st if g2 > 0 else np.zeros_like(x, dtype=float)
            Hy = -(ky * kz / g2) * Cx * Sy * Cz * st if g2 > 0 else np.zeros_like(x, dtype=float)
            Hz = Cx * Cy * Sz * st

        else:  # TM
            # E_z = E₀ sin(kx·x) sin(ky·y) cos(kz·z) · cos(ωt)
            # E ~ cos(ωt),  H ~ sin(ωt)
            Cx  = np.cos(kx * x)
            Sx  = np.sin(kx * x)
            Cy  = np.cos(ky * y)
            Sy  = np.sin(ky * y)
            Cz  = np.cos(kz * z)
            Sz  = np.sin(kz * z)

            if g2 > 0:
                Ex = (kx * kz / g2) * Cx * Sy * Sz * ct
                Ey = (ky * kz / g2) * Sx * Cy * Sz * ct
            else:
                Ex = np.zeros_like(x, dtype=float)
                Ey = np.zeros_like(x, dtype=float)
            Ez = Sx * Sy * Cz * ct

            if g2 > 0:
                Hx = ( ω * EPS0 * ky / g2) * Sx * Cy * Cz * st
                Hy = (-ω * EPS0 * kx / g2) * Cx * Sy * Cz * st
            else:
                Hx = np.zeros_like(x, dtype=float)
                Hy = np.zeros_like(x, dtype=float)
            Hz = np.zeros_like(x, dtype=float)

        return EMField(Ex, Ey, Ez, Hx, Hy, Hz)

    def label(self) -> str:
        return f"{self.mode_type}_{self.m}{self.n}{self.p}"


def rectangular_cavity_modes(a: float, b: float, d: float,
                              n_modes: int = 10) -> list[RectangularCavityMode]:
    """Return the n_modes lowest-frequency rectangular cavity modes."""
    modes: list[RectangularCavityMode] = []
    max_idx = 5
    for m in range(0, max_idx + 1):
        for n in range(0, max_idx + 1):
            for p in range(0, max_idx + 1):
                for t in ('TE', 'TM'):
                    try:
                        mode = RectangularCavityMode(a, b, d, m, n, p, t)
                        modes.append(mode)
                    except ValueError:
                        pass
    modes.sort(key=lambda mo: mo.resonant_frequency)
    # Deduplicate by (type, m, n, p) — TE and TM can share same index set but are distinct
    seen: set[tuple] = set()
    unique: list[RectangularCavityMode] = []
    for mo in modes:
        key = (mo.mode_type, mo.m, mo.n, mo.p)
        if key not in seen:
            seen.add(key)
            unique.append(mo)
        if len(unique) >= n_modes:
            break
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Cylindrical Cavity
# ─────────────────────────────────────────────────────────────────────────────

# Pre-computed Bessel function zeros (J_m and J_m') for quick access
def _bessel_tm_zeros(m: int, n: int) -> float:
    """χ_mn: n-th zero of J_m (TM cutoff)."""
    return float(jn_zeros(m, n)[n - 1])

def _bessel_te_zeros(m: int, n: int) -> float:
    """χ'_mn: n-th zero of J_m' (TE cutoff)."""
    return float(jnp_zeros(m, n)[n - 1])


class CylindricalCavityMode:
    """
    TM_mnp or TE_mnp mode in a cylindrical PEC cavity (0 ≤ ρ ≤ R, 0 ≤ z ≤ L).

    TM_mnp  (H_z = 0): χ_mn = n-th zero of J_m, p ≥ 0
      f = (c/2π) √( (χ_mn/R)² + (pπ/L)² )

    TE_mnp  (E_z = 0): χ'_mn = n-th zero of J_m', p ≥ 1
      f = (c/2π) √( (χ'_mn/R)² + (pπ/L)² )
    """

    def __init__(self, R: float, L: float,
                 m: int, n: int, p: int,
                 mode_type: str = 'TM') -> None:
        self.R, self.L = R, L
        self.m, self.n, self.p = m, n, p
        self.mode_type = mode_type.upper()
        self._validate()
        self._precompute()

    def _validate(self) -> None:
        t = self.mode_type
        m, n, p = self.m, self.n, self.p
        if n < 1:
            raise ValueError("n must be ≥ 1")
        if m < 0:
            raise ValueError("m must be ≥ 0")
        if t == 'TM' and p < 0:
            raise ValueError("TM: p must be ≥ 0")
        if t == 'TE' and p < 1:
            raise ValueError("TE: p must be ≥ 1")

    def _precompute(self) -> None:
        m, n, p = self.m, self.n, self.p
        R, L = self.R, self.L
        if self.mode_type == 'TM':
            self.chi = _bessel_tm_zeros(m, n)  # zeros of J_m
        else:
            self.chi = _bessel_te_zeros(m, n)  # zeros of J_m'
        self.kc = self.chi / R
        self.kz = p * np.pi / L if L > 0 else 0.0
        self.k  = np.sqrt(self.kc**2 + self.kz**2)
        self.omega = C_LIGHT * self.k

    @property
    def resonant_frequency(self) -> float:
        return self.omega / (2 * np.pi)

    def fields_rz(self, rho: np.ndarray, z: np.ndarray,
                  phi: float = 0.0, phase: float = 0.0) -> EMField:
        """
        EM fields in the ρ-z cross-section at azimuthal angle φ.
        Returns (Eρ, Eφ, Ez, Hρ, Hφ, Hz) mapped to (Ex→Eρ, Ey→Eφ, Ez, ...).
        """
        m, p   = self.m, self.p
        kc, kz = self.kc, self.kz
        ω      = self.omega
        ct, st = np.cos(phase), np.sin(phase)

        Jm   = jv(m, kc * rho)
        dJm  = jvp(m, kc * rho)   # dJ_m/d(kc*ρ) * kc = J_m'(kc*ρ) * kc
        cos_mphi = np.cos(m * phi) * np.ones_like(rho)
        sin_mphi = np.sin(m * phi) * np.ones_like(rho)
        Cz   = np.cos(kz * z)
        Sz   = np.sin(kz * z)

        if self.mode_type == 'TM':
            # E_z = J_m(kc·ρ) cos(mφ) cos(kz·z)  ← amplitude, E ~ cos(ωt)
            # H_ρ, H_φ ~ sin(ωt)
            Ez  = Jm * cos_mphi * Cz * ct
            if kc > 0:
                Er  = -(kz / kc) * dJm * cos_mphi * Sz * ct
                Ep  =  (kz * m / (kc**2 * np.maximum(rho, 1e-30))) * Jm * sin_mphi * Sz * ct
                Hr  =  (ω * EPS0 * m / (kc**2 * np.maximum(rho, 1e-30))) * Jm * sin_mphi * Cz * st
                Hm  = -(ω * EPS0 / kc) * dJm * cos_mphi * Cz * st
            else:
                Er = Ep = Hr = Hm = np.zeros_like(rho)
            Hz  = np.zeros_like(rho)
        else:  # TE
            # H_z = J_m(kc·ρ) cos(mφ) sin(kz·z)  ← amplitude, H ~ sin(ωt)
            Hz  = Jm * cos_mphi * Sz * st
            if kc > 0:
                Hr  = -(kz / kc) * dJm * cos_mphi * Cz * st
                Hm  =  (kz * m / (kc**2 * np.maximum(rho, 1e-30))) * Jm * sin_mphi * Cz * st
                Er  = -(ω * MU0 * m / (kc**2 * np.maximum(rho, 1e-30))) * Jm * sin_mphi * Sz * ct
                Ep  = -(ω * MU0 / kc) * dJm * cos_mphi * Sz * ct
            else:
                Hr = Hm = Er = Ep = np.zeros_like(rho)
            Ez  = np.zeros_like(rho)

        return EMField(Er, Ep, Ez, Hr, Hm, Hz)

    def label(self) -> str:
        return f"{self.mode_type}_{self.m}{self.n}{self.p}"


def cylindrical_cavity_modes(R: float, L: float,
                              n_modes: int = 10) -> list[CylindricalCavityMode]:
    """Return the n_modes lowest-frequency cylindrical cavity modes."""
    modes: list[CylindricalCavityMode] = []
    max_m, max_n, max_p = 4, 3, 3
    for m in range(0, max_m + 1):
        for n in range(1, max_n + 1):
            for p in range(0, max_p + 1):
                for t in ('TM', 'TE'):
                    try:
                        mode = CylindricalCavityMode(R, L, m, n, p, t)
                        modes.append(mode)
                    except ValueError:
                        pass
    modes.sort(key=lambda mo: mo.resonant_frequency)
    seen: set[tuple] = set()
    unique: list[CylindricalCavityMode] = []
    for mo in modes:
        key = (mo.mode_type, mo.m, mo.n, mo.p)
        if key not in seen:
            seen.add(key)
            unique.append(mo)
        if len(unique) >= n_modes:
            break
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Spherical Cavity
# ─────────────────────────────────────────────────────────────────────────────

def _spherical_jn_zeros(l: int, n: int) -> float:
    """χ_ln: n-th zero of j_l(x) (TM modes of spherical cavity)."""
    # j_l(x) ≈ 0 near x = lπ/2 + π/2, π, 3π/2, ...
    # Use bracketing and bisection
    count = 0
    x = 0.1
    dx = 0.1
    while count < n:
        x += dx
        f = spherical_jn(l, x)
        while True:
            x_next = x + dx
            f_next = spherical_jn(l, x_next)
            if f * f_next < 0:  # sign change → bracket found
                root = brentq(lambda t: spherical_jn(l, t), x, x_next, xtol=1e-12)
                count += 1
                if count == n:
                    return root
                x = x_next
                break
            x = x_next
            f = f_next
    raise RuntimeError("Could not find enough zeros")


def _spherical_jn_deriv_zeros(l: int, n: int) -> float:
    """χ'_ln: n-th zero of d/dx[x·j_l(x)] (TE modes of spherical cavity)."""
    # d/dx[x j_l(x)] = x j_{l-1}(x) - l j_l(x) ... not quite. Use numerical derivative.
    def djn_func(x: float) -> float:
        h = 1e-7 * max(abs(x), 1.0)
        return (spherical_jn(l, x + h) * (x + h) - spherical_jn(l, x - h) * (x - h)) / (2 * h)

    count = 0
    x = 0.1
    dx = 0.1
    # Skip l=0 first zero near 0
    while x < l + 0.5:
        x += dx
    f = djn_func(x)
    while count < n:
        x_next = x + dx
        f_next = djn_func(x_next)
        if f * f_next < 0:
            root = brentq(djn_func, x, x_next, xtol=1e-12)
            count += 1
            if count == n:
                return root
        x = x_next
        f = f_next
        if x > 200:
            raise RuntimeError("Search range exceeded")
    raise RuntimeError("Could not find enough zeros")


class SphericalCavityMode:
    """
    TM or TE mode in a spherical PEC cavity of radius R.

    TM_ln: zeros of j_l(kR) → k R = χ_ln (l-th spherical Bessel zero)
    TE_ln: zeros of d/dr[r j_l(kr)] at r=R

    Resonant frequency: f = c χ_ln / (2π R)
    """

    def __init__(self, R: float, l: int, n: int,
                 mode_type: str = 'TM') -> None:
        self.R = R
        self.l, self.n_idx = l, n
        self.mode_type = mode_type.upper()
        self._precompute()

    def _precompute(self) -> None:
        l, n = self.l, self.n_idx
        R = self.R
        if self.mode_type == 'TM':
            self.chi = _spherical_jn_zeros(l, n)
        else:
            self.chi = _spherical_jn_deriv_zeros(l, n)
        self.k = self.chi / R
        self.omega = C_LIGHT * self.k

    @property
    def resonant_frequency(self) -> float:
        return self.omega / (2 * np.pi)

    def fields_rtheta(self, r: np.ndarray, theta: np.ndarray,
                      phi: float = 0.0, m_az: int = 0,
                      phase: float = 0.0) -> EMField:
        """
        EM fields in the r-θ cross-section.

        Returns (Er, Eθ, Eφ, Hr, Hθ, Hφ) packed as EMField.
        For TM modes with azimuthal order m_az and l:
          E_r, E_θ, E_φ — dominant E components
          H_θ, H_φ     — magnetic field (no H_r for TM)
        """
        l   = self.l
        k   = self.k
        ct, st = np.cos(phase), np.sin(phase)

        # Spherical Bessel function and its derivative
        kr  = k * r
        jl  = spherical_jn(l, kr)
        djl = spherical_jn(l, kr, derivative=True)  # d/d(kr) j_l(kr) = j_l'(kr)

        # Legendre polynomial P_l^m(cos θ) — use m_az = 0 for simplicity
        from scipy.special import lpmv, factorial
        cos_t = np.cos(theta)
        # Normalised associated Legendre P_l^0
        Plm   = lpmv(m_az, l, cos_t)
        sin_t = np.sin(theta) + 1e-30  # avoid /0 at poles

        if self.mode_type == 'TM':
            # E_r ~ l(l+1)/r² j_l(kr) P_l(cos θ)  × cos(ωt)
            Er   = l * (l + 1) / (k**2 * r**2 + 1e-30) * jl * Plm * ct
            # E_θ ~ (1/kr) d/d(kr)[kr j_l(kr)] dP_l/dθ × cos(ωt)
            # d/d(kr)[kr j_l] = j_l + kr j_l'
            dkrj = jl + kr * djl
            dPlm_dtheta = -lpmv(m_az, l + 1, cos_t) + (l + m_az) / (sin_t + 1e-30) * cos_t * Plm if l < 20 else np.zeros_like(theta)
            Et   = (1.0 / (kr + 1e-30)) * dkrj * dPlm_dtheta / (sin_t + 1e-30) * ct
            Ep   = np.zeros_like(r)
            Hr   = np.zeros_like(r)
            Ht   = np.zeros_like(r)
            # H_φ ~ -iωε₀/k²... use simplified: H_φ ~ (ωε₀/k) (1/kr) j_l P_l
            Hp   = -(ω := self.omega) * EPS0 / k * (1.0 / (kr + 1e-30)) * jl * Plm * st
        else:  # TE
            Er   = np.zeros_like(r)
            # E_φ ~ (ωμ₀/k) (1/kr) j_l P_l × cos(ωt)
            Ep   = (ω := self.omega) * MU0 / k * (1.0 / (kr + 1e-30)) * jl * Plm * ct
            Et   = np.zeros_like(r)
            Hr   = l * (l + 1) / (k**2 * r**2 + 1e-30) * jl * Plm * st
            Ht   = (1.0 / (kr + 1e-30)) * (jl + kr * djl) * Plm / (sin_t + 1e-30) * st
            Hp   = np.zeros_like(r)

        return EMField(Er, Et, Ep, Hr, Ht, Hp)

    def label(self) -> str:
        return f"{self.mode_type}_{self.l}{self.n_idx}"


def spherical_cavity_modes(R: float, n_modes: int = 8) -> list[SphericalCavityMode]:
    """Return the n_modes lowest-frequency spherical cavity modes."""
    modes: list[SphericalCavityMode] = []
    for l in range(1, 6):
        for n in range(1, 4):
            for t in ('TM', 'TE'):
                try:
                    mode = SphericalCavityMode(R, l, n, t)
                    modes.append(mode)
                except Exception:
                    pass
    modes.sort(key=lambda mo: mo.resonant_frequency)
    return modes[:n_modes]
