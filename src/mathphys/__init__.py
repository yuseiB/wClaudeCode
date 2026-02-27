"""Mathematical Physics toolkit — Python layer."""

from mathphys.numerics import integrate_trapezoid, finite_difference
from mathphys.double_pendulum import DoublePendulum
from mathphys.ising_model import IsingModel2D, T_CRITICAL
from mathphys.cavity import (
    RectangularCavityMode, CylindricalCavityMode, SphericalCavityMode,
    rectangular_cavity_modes, cylindrical_cavity_modes, spherical_cavity_modes,
    EMField, C_LIGHT, MU0, EPS0, ETA0,
)
from mathphys.waveguide import (
    RectangularWaveguideMode, RectangularWaveguide,
    CircularWaveguideMode, CircularWaveguide,
)
from mathphys.accelerator import (
    Particle, Beam,
    Drift, Quadrupole, Dipole, Sextupole, RFCavity, Marker,
    Lattice, track,
    make_fodo, make_ring,
)

__all__ = [
    "integrate_trapezoid", "finite_difference", "DoublePendulum",
    "IsingModel2D", "T_CRITICAL",
    "EMField", "C_LIGHT", "MU0", "EPS0", "ETA0",
    "RectangularCavityMode", "CylindricalCavityMode", "SphericalCavityMode",
    "rectangular_cavity_modes", "cylindrical_cavity_modes", "spherical_cavity_modes",
    "RectangularWaveguideMode", "RectangularWaveguide",
    "CircularWaveguideMode", "CircularWaveguide",
    # Accelerator physics
    "Particle", "Beam",
    "Drift", "Quadrupole", "Dipole", "Sextupole", "RFCavity", "Marker",
    "Lattice", "track",
    "make_fodo", "make_ring",
]
