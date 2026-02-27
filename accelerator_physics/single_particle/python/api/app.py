"""Accelerator Physics REST API.

FastAPI backend that exposes Python single-particle tracking computations
to a TypeScript browser frontend.

Usage
-----
Install extras::

    pip install "mathphys[api]"

Start the server::

    uvicorn app:app --reload --port 8000 \\
        --app-dir accelerator_physics/single_particle/python/api

Then open the browser app (dist/index.html or ``npm run dev``) and point it
to ``http://localhost:8000``.

Endpoints
---------
GET  /                     — health check
GET  /lattice/fodo         — FODO ring preset
GET  /lattice/ring         — simple AG ring preset
POST /twiss                — compute Twiss functions
POST /track                — multi-turn particle tracking
POST /aperture             — dynamic-aperture scan
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow running from within the api/ directory or from the repo root
_REPO = Path(__file__).resolve().parents[5]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from mathphys.accelerator import (
    Beam,
    Dipole,
    Drift,
    Lattice,
    Marker,
    Quadrupole,
    RFCavity,
    Sextupole,
    make_fodo,
    make_ring,
    track,
)

from schemas import (
    ApertureRequest,
    ApertureResponse,
    ElementSpec,
    LatticeRequest,
    TrackRequest,
    TrackResponse,
    TwissResponse,
)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Accelerator Physics API",
    description="Single-particle dynamics: Twiss, tracking, dynamic aperture.",
    version="0.1.0",
)

# Allow all origins for local development (browser → localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_element(spec: ElementSpec):
    """Convert an ElementSpec to an Element object."""
    p = spec.params
    t = spec.type
    L = spec.length
    n = spec.name or t

    if t == "drift":
        return Drift(L, name=n)

    if t == "quadrupole":
        k1 = float(p.get("k1", 0.0))
        return Quadrupole(L, k1, name=n)

    if t == "dipole":
        angle = float(p.get("angle", 0.0))
        e1 = float(p.get("e1", 0.0))
        e2 = float(p.get("e2", 0.0))
        return Dipole(L, angle, e1=e1, e2=e2, name=n)

    if t == "sextupole":
        k2 = float(p.get("k2", 0.0))
        return Sextupole(L, k2, name=n)

    if t == "rf":
        voltage   = float(p.get("voltage", 1e5))
        frequency = float(p.get("frequency", 400e6))
        phi_s     = float(p.get("phi_s", math.pi / 2))
        energy0   = float(p.get("energy0", 1e9))
        return RFCavity(L, voltage, frequency, phi_s, energy0, name=n)

    if t == "marker":
        return Marker(name=n)

    raise HTTPException(status_code=422, detail=f"Unknown element type: '{t}'")


def _build_lattice(req: LatticeRequest) -> Lattice:
    """Convert a LatticeRequest to a Lattice object."""
    elements = [_build_element(e) for e in req.elements]
    return Lattice(elements, n_repeats=req.n_repeats)


def _dynamic_aperture(lat: Lattice, n_angles: int, n_turns: int,
                       max_amp: float, beta0: float) -> tuple[list, list]:
    """Binary-search dynamic aperture at *n_angles* angles."""
    obs_idx = len(lat.elements) - 1
    angles = list(np.linspace(0, 2 * math.pi, n_angles, endpoint=False))
    apertures = []

    for angle in angles:
        lo, hi = 0.0, max_amp
        for _ in range(14):
            mid = (lo + hi) / 2.0
            x0  = mid * math.cos(angle)
            px0 = mid * math.sin(angle) / math.sqrt(beta0)
            pts = np.array([[x0, px0, 0.0, 0.0, 0.0, 0.0]])
            hist = track(pts, lat, n_turns=n_turns, monitor_indices=[obs_idx])
            max_coord = float(np.max(np.abs(hist[:, 0, 0, :4])))
            if max_coord < 0.5:   # still within ±0.5 m → stable
                lo = mid
            else:
                hi = mid
        apertures.append(lo)

    return angles, apertures


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "accelerator-physics-api"}


# ── Presets ───────────────────────────────────────────────────────────────────

def _elem_to_spec(e) -> ElementSpec:
    """Convert an Element object back to a JSON-serialisable ElementSpec."""
    t = type(e).__name__.lower()
    p: dict = {}

    if t == "quadrupole":
        p["k1"] = e.k1
    elif t == "dipole":
        p["angle"] = e.angle
        p["e1"] = e.e1
        p["e2"] = e.e2
    elif t == "sextupole":
        p["k2"] = e.k2
    elif t == "rfcavity":
        t = "rf"
        p["voltage"] = e.voltage
        p["frequency"] = e.frequency
        p["phi_s"] = e.phi_s
        p["energy0"] = e.energy0
    elif t == "marker":
        pass

    return ElementSpec(type=t, name=e.name, length=e.length, params=p)


@app.get("/lattice/fodo", response_model=LatticeRequest)
def preset_fodo():
    """Return a 4-cell FODO ring as a LatticeRequest (usable in /twiss, /track)."""
    lat = make_fodo(Lq=0.5, Ld=2.0, n_cells=1)
    specs = [_elem_to_spec(e) for e in lat.elements]
    return LatticeRequest(elements=specs, n_repeats=4)


@app.get("/lattice/ring", response_model=LatticeRequest)
def preset_ring():
    """Return a simple 8-dipole AG ring as a LatticeRequest."""
    bend_angle = 2 * math.pi / 8
    arc = [
        ElementSpec(type="dipole",     name="B",  length=2.0, params={"angle": bend_angle}),
        ElementSpec(type="drift",      name="Da", length=0.5, params={}),
        ElementSpec(type="quadrupole", name="QF", length=0.3, params={"k1": +1.8}),
        ElementSpec(type="drift",      name="Db", length=0.5, params={}),
        ElementSpec(type="quadrupole", name="QD", length=0.3, params={"k1": -1.8}),
        ElementSpec(type="drift",      name="Dc", length=0.5, params={}),
    ]
    return LatticeRequest(elements=arc, n_repeats=8)


# ── Twiss ─────────────────────────────────────────────────────────────────────

@app.post("/twiss", response_model=TwissResponse)
def compute_twiss(req: LatticeRequest):
    """Compute Courant-Snyder Twiss functions for the given lattice."""
    try:
        lat = _build_lattice(req)
        t = lat.twiss()
        t0 = lat.twiss_at_start()
        xi_x, xi_y = lat.chromaticity()
        alpha_c = lat.momentum_compaction()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return TwissResponse(
        s=t["s"].tolist(),
        beta_x=t["beta_x"].tolist(),
        alpha_x=t["alpha_x"].tolist(),
        gamma_x=t["gamma_x"].tolist(),
        beta_y=t["beta_y"].tolist(),
        alpha_y=t["alpha_y"].tolist(),
        gamma_y=t["gamma_y"].tolist(),
        Dx=t["Dx"].tolist(),
        Dpx=t["Dpx"].tolist(),
        tune_x=t0["tune_x"],
        tune_y=t0["tune_y"],
        chromaticity_x=xi_x,
        chromaticity_y=xi_y,
        momentum_compaction=alpha_c,
        circumference=lat.circumference,
    )


# ── Tracking ──────────────────────────────────────────────────────────────────

@app.post("/track", response_model=TrackResponse)
def track_beam(req: TrackRequest):
    """Multi-turn particle tracking through the given lattice."""
    try:
        lat = _build_lattice(req.lattice)
    except (ValueError, HTTPException) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    pts = np.array(req.particles, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 6:
        raise HTTPException(
            status_code=422,
            detail=f"particles must be a list of [x,px,y,py,delta,l] rows; "
                   f"got shape {list(pts.shape)}",
        )

    n_p = len(pts)
    if n_p > 500:
        raise HTTPException(status_code=422,
                            detail="Maximum 500 particles per request.")

    hist = track(pts, lat, n_turns=req.n_turns)
    # hist: (n_turns+1, 1, n_particles, 6)
    data = hist[:, 0, :, :].tolist()   # (n_turns+1, n_particles, 6)

    return TrackResponse(data=data, n_turns=req.n_turns, n_particles=n_p)


# ── Dynamic aperture ──────────────────────────────────────────────────────────

@app.post("/aperture", response_model=ApertureResponse)
def compute_aperture(req: ApertureRequest):
    """Dynamic-aperture polar scan (binary search at each angle)."""
    try:
        lat = _build_lattice(req.lattice)
        t0 = lat.twiss_at_start()
        beta0 = t0["beta_x"]
    except (ValueError, HTTPException) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    angles, apertures = _dynamic_aperture(
        lat,
        n_angles=req.n_angles,
        n_turns=req.n_turns,
        max_amp=req.max_amp,
        beta0=beta0,
    )

    return ApertureResponse(angles=angles, apertures=apertures)
