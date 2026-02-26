"""
waveguide.py — Electromagnetic Wave Propagation in Waveguides (Analytical Solutions)

PEC (Perfect Electric Conductor) waveguides:
  - RectangularWaveguide : a × b cross-section, uniform in z
  - CircularWaveguide    : radius R, uniform in z

All modes satisfy: tangential E = 0 at conducting walls.
Propagation direction: +z.

Physical constants (SI):
  c  = 2.99792458e8 m/s
  μ₀ = 4π×10⁻⁷ H/m
  ε₀ = 1/(μ₀c²) F/m
  η₀ = μ₀c ≈ 376.73 Ω

References:
  Pozar, "Microwave Engineering", 4th ed., ch. 3
  Griffiths, "Introduction to Electrodynamics", 4th ed., ch. 9
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple
from scipy.special import jv, jvp, jn_zeros, jnp_zeros

from mathphys.cavity import C_LIGHT, MU0, EPS0, ETA0, EMField

# ─────────────────────────────────────────────────────────────────────────────
# Rectangular Waveguide
# ─────────────────────────────────────────────────────────────────────────────

class RectangularWaveguideMode:
    """
    TE_mn or TM_mn mode in a rectangular PEC waveguide (0 ≤ x ≤ a, 0 ≤ y ≤ b).

    TE_mn  (E_z = 0): m,n ≥ 0 (not both 0)
    TM_mn  (H_z = 0): m,n ≥ 1

    Cutoff wavenumber: k_c = π√((m/a)² + (n/b)²)
    Cutoff frequency : f_c = c·k_c / (2π)
    Propagation const: β   = √((ω/c)² − k_c²)   for f > f_c
                       β   = −j√(k_c² − (ω/c)²)  for f < f_c (evanescent)

    Wave impedance:
      TE: Z_TE = η₀·(k/β)   = η₀ / √(1−(f_c/f)²)
      TM: Z_TM = η₀·(β/k)   = η₀ · √(1−(f_c/f)²)
    """

    def __init__(self, a: float, b: float,
                 m: int, n: int,
                 mode_type: str = 'TE') -> None:
        """
        Parameters
        ----------
        a, b       : waveguide cross-section dimensions (m), a > b by convention
        m, n       : mode indices (non-negative integers)
        mode_type  : 'TE' or 'TM'
        """
        self.a, self.b = a, b
        self.m, self.n = m, n
        self.mode_type = mode_type.upper()
        self._validate()
        self._precompute()

    def _validate(self) -> None:
        t, m, n = self.mode_type, self.m, self.n
        if t == 'TE':
            if m == 0 and n == 0:
                raise ValueError("TE: m and n cannot both be 0")
            if m < 0 or n < 0:
                raise ValueError("TE: m, n must be ≥ 0")
        elif t == 'TM':
            if m < 1 or n < 1:
                raise ValueError("TM: m and n must be ≥ 1")
        else:
            raise ValueError(f"mode_type must be 'TE' or 'TM', got {t!r}")

    def _precompute(self) -> None:
        m, n = self.m, self.n
        a, b = self.a, self.b
        self.kx  = m * np.pi / a
        self.ky  = n * np.pi / b
        self.kc  = np.sqrt(self.kx**2 + self.ky**2)
        self.fc  = C_LIGHT * self.kc / (2 * np.pi)
        self.kc2 = self.kc**2

    @property
    def cutoff_frequency(self) -> float:
        """Cutoff frequency in Hz."""
        return self.fc

    def propagation_constant(self, frequency: float) -> complex:
        """
        Complex propagation constant β at the given frequency.

        β is real (propagating) for f > f_c,
        imaginary (evanescent) for f < f_c.
        """
        k = 2 * np.pi * frequency / C_LIGHT
        k2_minus_kc2 = k**2 - self.kc2
        if k2_minus_kc2 >= 0:
            return float(np.sqrt(k2_minus_kc2))
        else:
            return 1j * float(np.sqrt(-k2_minus_kc2))

    def wave_impedance(self, frequency: float) -> complex:
        """Mode wave impedance in Ω."""
        k  = 2 * np.pi * frequency / C_LIGHT
        beta = self.propagation_constant(frequency)
        if self.mode_type == 'TE':
            return ETA0 * k / beta if beta != 0 else complex(np.inf)
        else:
            return ETA0 * beta / k

    def fields(self, x: np.ndarray, y: np.ndarray, z: float,
               frequency: float, phase: float = 0.0) -> EMField:
        """
        Compute EM fields in the cross-section at axial position z.

        The propagating mode has the form:
          E(x,y,z,t) = E_t(x,y) · cos(ωt − βz + phase)
          H(x,y,z,t) = H_t(x,y) · cos(ωt − βz + phase)  [E,H in-phase for propagating]

        Parameters
        ----------
        x, y      : 2-D meshgrid arrays (shape must match)
        z         : axial position (m)
        frequency : operating frequency (Hz)
        phase     : ωt + phase₀ (radians)

        Returns
        -------
        EMField with components (Ex, Ey, Ez, Hx, Hy, Hz)
        """
        kx, ky = self.kx, self.ky
        kc2    = self.kc2
        omega  = 2 * np.pi * frequency

        beta = self.propagation_constant(frequency)
        psi  = np.cos(phase - float(np.real(beta)) * z)   # phase factor

        if self.mode_type == 'TE':
            # H_z = H₀ cos(kx·x) cos(ky·y) exp(j(ωt−βz))
            Cx = np.cos(kx * x)
            Sx = np.sin(kx * x)
            Cy = np.cos(ky * y)
            Sy = np.sin(ky * y)

            Hz = Cx * Cy * psi
            if kc2 > 0:
                Hx = (float(np.real(beta)) * kx / kc2) * Sx * Cy * psi
                Hy = (float(np.real(beta)) * ky / kc2) * Cx * Sy * psi
                Ex = -(omega * MU0 * ky / kc2) * Cx * Sy * psi
                Ey =  (omega * MU0 * kx / kc2) * Sx * Cy * psi
            else:
                Hx = Hy = Ex = Ey = np.zeros_like(x, dtype=float)
            Ez = np.zeros_like(x, dtype=float)

        else:  # TM
            # E_z = E₀ sin(kx·x) sin(ky·y) exp(j(ωt−βz))
            Cx = np.cos(kx * x)
            Sx = np.sin(kx * x)
            Cy = np.cos(ky * y)
            Sy = np.sin(ky * y)

            Ez = Sx * Sy * psi
            if kc2 > 0:
                Ex = -(float(np.real(beta)) * kx / kc2) * Cx * Sy * psi
                Ey = -(float(np.real(beta)) * ky / kc2) * Sx * Cy * psi
                Hx =  (omega * EPS0 * ky / kc2) * Sx * Cy * psi
                Hy = -(omega * EPS0 * kx / kc2) * Cx * Sy * psi
            else:
                Ex = Ey = Hx = Hy = np.zeros_like(x, dtype=float)
            Hz = np.zeros_like(x, dtype=float)

        return EMField(Ex, Ey, Ez, Hx, Hy, Hz)

    def label(self) -> str:
        return f"{self.mode_type}_{self.m}{self.n}"


class RectangularWaveguide:
    """
    Helper class for a rectangular waveguide that collects multiple modes.

    Parameters
    ----------
    a, b : cross-section dimensions (m), a ≥ b by convention.
    """

    def __init__(self, a: float, b: float) -> None:
        self.a, self.b = a, b

    def mode(self, m: int, n: int, mode_type: str = 'TE') -> RectangularWaveguideMode:
        """Return the specified mode object."""
        return RectangularWaveguideMode(self.a, self.b, m, n, mode_type)

    def modes(self, n_modes: int = 8) -> list[RectangularWaveguideMode]:
        """Return the n_modes lowest-cutoff-frequency modes."""
        result: list[RectangularWaveguideMode] = []
        max_idx = 5
        for m in range(0, max_idx + 1):
            for n in range(0, max_idx + 1):
                for t in ('TE', 'TM'):
                    try:
                        result.append(RectangularWaveguideMode(self.a, self.b, m, n, t))
                    except ValueError:
                        pass
        result.sort(key=lambda mo: mo.cutoff_frequency)
        seen: set[tuple] = set()
        unique: list[RectangularWaveguideMode] = []
        for mo in result:
            key = (mo.mode_type, mo.m, mo.n)
            if key not in seen:
                seen.add(key)
                unique.append(mo)
            if len(unique) >= n_modes:
                break
        return unique

    def dispersion(self, mode: RectangularWaveguideMode,
                   f_range: tuple[float, float],
                   n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute dispersion relation β(f) for a given mode.

        Returns
        -------
        freqs  : frequency array (Hz)
        betas  : propagation constant array (rad/m), 0 for evanescent
        """
        freqs = np.linspace(f_range[0], f_range[1], n_points)
        betas = np.array([
            float(np.real(mode.propagation_constant(f))) for f in freqs
        ])
        return freqs, betas


# ─────────────────────────────────────────────────────────────────────────────
# Circular Waveguide
# ─────────────────────────────────────────────────────────────────────────────

class CircularWaveguideMode:
    """
    TE_mn or TM_mn mode in a circular PEC waveguide (0 ≤ ρ ≤ R).

    TM_mn  (H_z = 0): k_c = χ_mn / R  (χ_mn = n-th zero of J_m)
    TE_mn  (E_z = 0): k_c = χ'_mn / R (χ'_mn = n-th zero of J_m')

    Cutoff frequency: f_c = c·k_c / (2π)
    """

    def __init__(self, R: float, m: int, n: int,
                 mode_type: str = 'TE') -> None:
        self.R = R
        self.m, self.n = m, n
        self.mode_type = mode_type.upper()
        self._validate()
        self._precompute()

    def _validate(self) -> None:
        if self.n < 1:
            raise ValueError("n must be ≥ 1")
        if self.m < 0:
            raise ValueError("m must be ≥ 0")
        t = self.mode_type
        if t not in ('TE', 'TM'):
            raise ValueError(f"mode_type must be 'TE' or 'TM', got {t!r}")

    def _precompute(self) -> None:
        m, n, R = self.m, self.n, self.R
        if self.mode_type == 'TM':
            self.chi = float(jn_zeros(m, n)[n - 1])
        else:
            self.chi = float(jnp_zeros(m, n)[n - 1])
        self.kc  = self.chi / R
        self.fc  = C_LIGHT * self.kc / (2 * np.pi)
        self.kc2 = self.kc**2

    @property
    def cutoff_frequency(self) -> float:
        return self.fc

    def propagation_constant(self, frequency: float) -> complex:
        """Complex propagation constant β."""
        k = 2 * np.pi * frequency / C_LIGHT
        d = k**2 - self.kc2
        if d >= 0:
            return float(np.sqrt(d))
        return 1j * float(np.sqrt(-d))

    def wave_impedance(self, frequency: float) -> complex:
        k    = 2 * np.pi * frequency / C_LIGHT
        beta = self.propagation_constant(frequency)
        if self.mode_type == 'TE':
            return ETA0 * k / beta if beta != 0 else complex(np.inf)
        return ETA0 * beta / k

    def fields_polar(self, rho: np.ndarray, phi: np.ndarray,
                     z: float, frequency: float,
                     phase: float = 0.0) -> EMField:
        """
        EM fields in the transverse (ρ,φ) plane at axial position z.

        Returns (Eρ, Eφ, Ez, Hρ, Hφ, Hz) packed as EMField.

        Parameters
        ----------
        rho, phi  : 2-D polar-coordinate meshgrid arrays
        z         : axial position (m)
        frequency : operating frequency (Hz)
        phase     : ωt (radians)
        """
        m   = self.m
        kc  = self.kc
        kc2 = self.kc2
        omega = 2 * np.pi * frequency

        beta = self.propagation_constant(frequency)
        psi  = np.cos(phase - float(np.real(beta)) * z)

        Jm   = jv(m, kc * rho)
        dJm  = jvp(m, kc * rho)           # J_m'(kc·ρ) w.r.t. argument
        cos_mphi = np.cos(m * phi)
        sin_mphi = np.sin(m * phi)
        rho_safe = np.maximum(rho, 1e-30)

        if self.mode_type == 'TM':
            # E_z = J_m(kc·ρ) cos(mφ) exp(j(ωt−βz))
            Ez  = Jm * cos_mphi * psi
            if kc2 > 0:
                br  = float(np.real(beta))
                Er  = -(br * kc / kc2) * dJm * cos_mphi * psi
                Ephi= (br * m / (kc2 * rho_safe)) * Jm * sin_mphi * psi
                Hr  = -(omega * EPS0 * m / (kc2 * rho_safe)) * Jm * sin_mphi * psi
                Hphi= (omega * EPS0 * kc / kc2) * dJm * cos_mphi * psi
            else:
                Er = Ephi = Hr = Hphi = np.zeros_like(rho)
            Hz  = np.zeros_like(rho)

        else:  # TE
            # H_z = J_m(kc·ρ) cos(mφ) exp(j(ωt−βz))
            Hz  = Jm * cos_mphi * psi
            if kc2 > 0:
                br  = float(np.real(beta))
                Hr  = -(br * kc / kc2) * dJm * cos_mphi * psi
                Hphi= (br * m / (kc2 * rho_safe)) * Jm * sin_mphi * psi
                Er  = (omega * MU0 * m / (kc2 * rho_safe)) * Jm * sin_mphi * psi
                Ephi= -(omega * MU0 * kc / kc2) * dJm * cos_mphi * psi
            else:
                Hr = Hphi = Er = Ephi = np.zeros_like(rho)
            Ez  = np.zeros_like(rho)

        return EMField(Er, Ephi, Ez, Hr, Hphi, Hz)

    def label(self) -> str:
        return f"{self.mode_type}_{self.m}{self.n}"


class CircularWaveguide:
    """
    Helper class for a circular waveguide that collects multiple modes.

    Parameters
    ----------
    R : waveguide radius (m)
    """

    def __init__(self, R: float) -> None:
        self.R = R

    def mode(self, m: int, n: int, mode_type: str = 'TE') -> CircularWaveguideMode:
        return CircularWaveguideMode(self.R, m, n, mode_type)

    def modes(self, n_modes: int = 8) -> list[CircularWaveguideMode]:
        """Return the n_modes lowest-cutoff-frequency modes."""
        result: list[CircularWaveguideMode] = []
        for m in range(0, 5):
            for n in range(1, 4):
                for t in ('TE', 'TM'):
                    try:
                        result.append(CircularWaveguideMode(self.R, m, n, t))
                    except ValueError:
                        pass
        result.sort(key=lambda mo: mo.cutoff_frequency)
        seen: set[tuple] = set()
        unique: list[CircularWaveguideMode] = []
        for mo in result:
            key = (mo.mode_type, mo.m, mo.n)
            if key not in seen:
                seen.add(key)
                unique.append(mo)
            if len(unique) >= n_modes:
                break
        return unique

    def dispersion(self, mode: CircularWaveguideMode,
                   f_range: tuple[float, float],
                   n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Compute dispersion relation β(f) for a given mode."""
        freqs = np.linspace(f_range[0], f_range[1], n_points)
        betas = np.array([
            float(np.real(mode.propagation_constant(f))) for f in freqs
        ])
        return freqs, betas
