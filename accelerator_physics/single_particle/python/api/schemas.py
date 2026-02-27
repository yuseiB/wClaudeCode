"""Pydantic schemas for the Accelerator Physics REST API.

These types define the JSON contract between the FastAPI backend and the
TypeScript browser frontend.  Every field that has a physical unit carries it
in a trailing comment.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Element specification ──────────────────────────────────────────────────────

class ElementSpec(BaseModel):
    """Single lattice element in JSON form."""

    type: Literal["drift", "quadrupole", "dipole", "sextupole", "rf", "marker"]
    name: str = ""
    length: float = Field(ge=0.0, description="Element length [m]")
    params: dict = Field(
        default_factory=dict,
        description=(
            "Type-specific parameters:\n"
            "  quadrupole: {k1 [1/m²]}\n"
            "  dipole:     {angle [rad], e1 [rad], e2 [rad]}\n"
            "  sextupole:  {k2 [1/m³]}\n"
            "  rf:         {voltage [V], frequency [Hz], phi_s [rad], energy0 [eV]}\n"
        ),
    )


# ── Lattice request ────────────────────────────────────────────────────────────

class LatticeRequest(BaseModel):
    """A complete lattice definition sent from the browser."""

    elements: list[ElementSpec]
    beam_energy_gev: float = Field(default=1.0, gt=0.0,
                                   description="Reference beam energy [GeV]")
    n_repeats: int = Field(default=1, ge=1,
                           description="Repeat the element list n times (for multi-cell rings)")


# ── Twiss response ─────────────────────────────────────────────────────────────

class TwissResponse(BaseModel):
    """Courant-Snyder Twiss functions at every element exit."""

    s: list[float]        = Field(description="s-coordinates [m]")
    beta_x: list[float]   = Field(description="Horizontal β-function [m]")
    alpha_x: list[float]  = Field(description="Horizontal α-function")
    gamma_x: list[float]  = Field(description="Horizontal γ-function [1/m]")
    beta_y: list[float]   = Field(description="Vertical β-function [m]")
    alpha_y: list[float]  = Field(description="Vertical α-function")
    gamma_y: list[float]  = Field(description="Vertical γ-function [1/m]")
    Dx: list[float]       = Field(description="Horizontal dispersion [m]")
    Dpx: list[float]      = Field(description="Horizontal dispersion derivative")
    tune_x: float         = Field(description="Fractional horizontal tune Qx")
    tune_y: float         = Field(description="Fractional vertical tune Qy")
    chromaticity_x: float = Field(description="Horizontal chromaticity ξx = dQx/dδ")
    chromaticity_y: float = Field(description="Vertical chromaticity ξy = dQy/dδ")
    momentum_compaction: float = Field(description="Momentum compaction factor αc")
    circumference: float  = Field(description="Ring circumference [m]")


# ── Tracking request & response ───────────────────────────────────────────────

class TrackRequest(BaseModel):
    """Multi-turn tracking request."""

    lattice: LatticeRequest
    particles: list[list[float]] = Field(
        description="Initial coordinates list of [x, px, y, py, delta, l] rows [m, rad, …]"
    )
    n_turns: int = Field(default=100, ge=1, le=10_000)


class TrackResponse(BaseModel):
    """Turn-by-turn coordinates at the ring exit.

    ``data[t][i]`` contains the 6D coordinate vector of particle ``i``
    after ``t`` complete turns (``t=0`` is the initial state).
    """

    data: list[list[list[float]]]   # shape (n_turns+1, n_particles, 6)
    n_turns: int
    n_particles: int


# ── Dynamic aperture ──────────────────────────────────────────────────────────

class ApertureRequest(BaseModel):
    """Dynamic aperture scan parameters."""

    lattice: LatticeRequest
    n_angles: int = Field(default=24, ge=4, le=128,
                          description="Number of angles in the polar scan")
    n_turns: int = Field(default=256, ge=32, le=2_000,
                          description="Number of tracking turns per particle")
    max_amp: float = Field(default=20e-3, gt=0.0,
                           description="Maximum search amplitude [m]")


class ApertureResponse(BaseModel):
    angles: list[float]    = Field(description="Scan angles [rad]")
    apertures: list[float] = Field(description="Stable amplitude at each angle [m]")
