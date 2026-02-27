"""FODO Lattice — Single-Particle Dynamics Demo.

Demonstrates:
  1. Building a FODO ring with the make_fodo() factory.
  2. Computing Courant-Snyder Twiss functions (β, α, D) around the ring.
  3. Generating a matched Gaussian beam and tracking for 200 turns.
  4. Visualising phase space and verifying emittance conservation (Liouville).

Output
------
  fodo_twiss.png       — Twiss functions β_x, β_y, D_x vs s
  fodo_phase_space.png — Turn-by-turn phase-space portrait & emittance plot

Run from the repo root::

    python accelerator_physics/single_particle/python/examples/fodo_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make the shared package importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))

from mathphys.accelerator import (
    Beam,
    Lattice,
    make_fodo,
    track,
)


# ── Lattice setup ─────────────────────────────────────────────────────────────

# 4 FODO cells, each: QF/2 → Drift → QD → Drift → QF/2
# Parameters: L_quad = 0.5 m,  L_drift = 2.0 m,  90° phase advance/cell
LAT = make_fodo(Lq=0.5, Ld=2.0, n_cells=4)

Qx, Qy = LAT.tune()
xi_x, xi_y = LAT.chromaticity()
C = LAT.circumference

print("═" * 55)
print("  FODO Ring — Optics Summary")
print("═" * 55)
print(f"  Circumference  C  = {C:.2f} m")
print(f"  Tune           Qx = {Qx:.4f},  Qy = {Qy:.4f}")
print(f"  Chromaticity   ξx = {xi_x:.3f}, ξy = {xi_y:.3f}")
print("═" * 55)

# ── Twiss functions ───────────────────────────────────────────────────────────

twiss = LAT.twiss()

fig1, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig1.suptitle(
    f"FODO Ring — Twiss Functions\n"
    f"$Q_x={Qx:.4f}$, $Q_y={Qy:.4f}$,"
    f"  $C={C:.1f}$ m",
    fontsize=12,
)

s = twiss["s"]

# Panel 1: Beta functions
ax = axes[0]
ax.plot(s, twiss["beta_x"], color="#e74c3c", lw=2, label=r"$\beta_x$ [m]")
ax.plot(s, twiss["beta_y"], color="#3498db", lw=2, label=r"$\beta_y$ [m]")
ax.set_ylabel("β [m]")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

# Shade element regions
_s0 = 0.0
for elem in LAT.elements:
    _s1 = _s0 + elem.length
    etype = type(elem).__name__
    color = (
        "#c0392b" if elem.name.startswith("QF") else
        "#2980b9" if elem.name.startswith("QD") else None
    )
    if color:
        ax.axvspan(_s0, _s1, alpha=0.12, color=color, linewidth=0)
    _s0 = _s1

# Panel 2: Dispersion
ax = axes[1]
ax.plot(s, twiss["Dx"] * 1e2, color="#2ecc71", lw=2, label=r"$D_x$ [cm]")
ax.axhline(0, color="gray", lw=0.7, ls="--")
ax.set_xlabel("s [m]")
ax.set_ylabel("$D_x$ [cm]")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

# Draw element tiles below the dispersion plot
ax_bottom = ax.get_position()
_s0 = 0.0
for elem in LAT.elements:
    _s1 = _s0 + elem.length
    etype = type(elem).__name__
    if etype == "Quadrupole":
        fc = "#c0392b" if elem.k1 > 0 else "#2980b9"
        lbl = "QF" if elem.k1 > 0 else "QD"
        ax.axvspan(_s0, _s1, alpha=0.18, color=fc, linewidth=0)
    _s0 = _s1

plt.tight_layout()
out1 = Path("fodo_twiss.png")
fig1.savefig(out1, dpi=150)
print(f"  Saved → {out1}")
plt.close(fig1)

# ── Beam tracking ─────────────────────────────────────────────────────────────

N_PART = 1_000
N_TURNS = 200

t0 = LAT.twiss_at_start()
beam = Beam.gaussian(
    n=N_PART,
    emittance_x=1e-6,  # 1 μm·rad
    beta_x=t0["beta_x"],
    alpha_x=t0["alpha_x"],
    emittance_y=1e-6,
    beta_y=t0["beta_y"],
    alpha_y=t0["alpha_y"],
    energy0=1.0,
    rng=np.random.default_rng(2024),
)

print(f"\n  Tracking {N_PART} particles for {N_TURNS} turns …")
history = track(beam.particles, LAT, n_turns=N_TURNS)
# history: (N_TURNS+1, 1, N_PART, 6)

# Emittance vs turn
turns_arr = np.arange(N_TURNS + 1)
emittance_x = [
    Beam(history[t, 0]).emittance("x") * 1e6  # convert to μm·rad
    for t in turns_arr
]
emittance_y = [
    Beam(history[t, 0]).emittance("y") * 1e6
    for t in turns_arr
]

eps0_x = emittance_x[0]
eps0_y = emittance_y[0]
print(f"  Initial εx = {eps0_x:.4f} μm·rad,  εy = {eps0_y:.4f} μm·rad")
print(f"  Final   εx = {emittance_x[-1]:.4f} μm·rad,  εy = {emittance_y[-1]:.4f} μm·rad")
drift_x = abs(emittance_x[-1] / eps0_x - 1) * 100
print(f"  Emittance drift ΔεxÅ/ε₀ = {drift_x:.2f}%  (should be < 1%)")

# ── Phase-space figure ────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
fig2.suptitle(
    f"FODO Ring — Poincaré Section (Turn-by-Turn)  N={N_PART} particles",
    fontsize=11,
)

# ── x–px phase space (all turns, subsample 100 turns, 200 particles)
ax = axes2[0]
cmap = plt.cm.plasma
n_show = min(100, N_TURNS)
step = max(1, N_TURNS // n_show)
for i_t in range(0, N_TURNS + 1, step):
    frac = i_t / N_TURNS
    c = cmap(frac)
    ax.scatter(
        history[i_t, 0, :200, 0] * 1e3,   # x [mm]
        history[i_t, 0, :200, 1] * 1e3,   # px [mrad]
        s=0.8, color=c, alpha=0.5, linewidths=0,
    )
ax.set_xlabel("x [mm]")
ax.set_ylabel("px [mrad]")
ax.set_title("x–p_x phase space")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(alpha=0.3)

# ── y–py phase space
ax = axes2[1]
for i_t in range(0, N_TURNS + 1, step):
    frac = i_t / N_TURNS
    c = cmap(frac)
    ax.scatter(
        history[i_t, 0, :200, 2] * 1e3,   # y [mm]
        history[i_t, 0, :200, 3] * 1e3,   # py [mrad]
        s=0.8, color=c, alpha=0.5, linewidths=0,
    )
ax.set_xlabel("y [mm]")
ax.set_ylabel("py [mrad]")
ax.set_title("y–p_y phase space")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(alpha=0.3)

# ── Emittance vs turn
ax = axes2[2]
ax.plot(turns_arr, emittance_x, color="#e74c3c", lw=1.5, label=r"$\varepsilon_x$")
ax.plot(turns_arr, emittance_y, color="#3498db", lw=1.5, label=r"$\varepsilon_y$")
ax.axhline(eps0_x, color="gray", ls="--", lw=0.8)
ax.set_xlabel("Turn")
ax.set_ylabel("Geometric emittance [μm·rad]")
ax.set_title("Emittance conservation (Liouville)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
out2 = Path("fodo_phase_space.png")
fig2.savefig(out2, dpi=150)
print(f"  Saved → {out2}")
plt.close(fig2)

print("\n  Done.")
