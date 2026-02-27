"""Simple AG Ring — Nonlinear Dynamics & Poincaré Section Demo.

Demonstrates:
  1. Building a simple alternating-gradient ring (dipoles + FODO quads).
  2. Comparing Poincaré sections with and without sextupole correctors.
  3. Computing tune diagram with resonance lines (up to 4th order).
  4. Dynamic-aperture scan (maximum stable amplitude vs launch angle).

Output
------
  ring_twiss.png    — Twiss functions β_x, β_y, D_x around the ring
  ring_poincare.png — Poincaré sections (linear vs nonlinear lattice)
  ring_tune.png     — Tune diagram with resonance lines
  ring_aperture.png — Dynamic aperture polar plot

Run from the repo root::

    python accelerator_physics/single_particle/python/examples/ring_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))

from mathphys.accelerator import (
    Beam,
    Dipole,
    Drift,
    Lattice,
    Marker,
    Quadrupole,
    Sextupole,
    make_ring,
    track,
)


# ── Helper: build ring with optional sextupoles ───────────────────────────────

def build_ring(k2: float = 0.0, n_bends: int = 8) -> Lattice:
    """8-fold symmetric ring: dipole + QF + QD per arc cell.

    Parameters
    ----------
    k2:      Sextupole strength [1/m³].  0 → no sextupoles (linear ring).
    n_bends: Number of bending magnets (each = 1/n_bends of 2π).
    """
    bend_angle = 2 * np.pi / n_bends
    Lb = 2.0    # dipole length [m]
    Lq = 0.3    # quad length [m]
    Ld = 0.5    # drift length [m]
    Ls = 0.2    # sextupole length [m]
    k1 = 1.8    # quad gradient [1/m²]

    arc = [
        Dipole(Lb, bend_angle, name="B"),
        Drift(Ld / 2, name="Da"),
        Quadrupole(Lq, +k1, name="QF"),
    ]
    if abs(k2) > 1e-12:
        arc.append(Sextupole(Ls, +k2, name="SF"))
    arc += [
        Drift(Ld / 2, name="Db"),
        Quadrupole(Lq, -k1, name="QD"),
    ]
    if abs(k2) > 1e-12:
        arc.append(Sextupole(Ls, -k2, name="SD"))
    arc += [
        Drift(Ld, name="Dc"),
        Marker(name="OBS"),
    ]

    return Lattice(arc, n_repeats=n_bends)


# ── Build linear ring and print optics ───────────────────────────────────────

LAT_LIN = build_ring(k2=0.0)
Qx, Qy = LAT_LIN.tune()
xi_x, xi_y = LAT_LIN.chromaticity()
alpha_c = LAT_LIN.momentum_compaction()
C = LAT_LIN.circumference

print("═" * 60)
print("  Simple AG Ring — Optics Summary  (linear lattice)")
print("═" * 60)
print(f"  Circumference      C     = {C:.2f} m")
print(f"  Tune               Qx    = {Qx:.4f},  Qy = {Qy:.4f}")
print(f"  Chromaticity       ξx    = {xi_x:.3f}, ξy = {xi_y:.3f}")
print(f"  Mom. compaction    αc    = {alpha_c:.5f}")
print("═" * 60)

# ── Twiss figure ─────────────────────────────────────────────────────────────

twiss = LAT_LIN.twiss()
s = twiss["s"]

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
fig.suptitle(
    f"Simple AG Ring — Twiss Functions\n"
    f"$Q_x={Qx:.4f}$, $Q_y={Qy:.4f}$, $C={C:.1f}$ m",
    fontsize=11,
)

ax = axes[0]
ax.plot(s, twiss["beta_x"], "#e74c3c", lw=2, label=r"$\beta_x$")
ax.plot(s, twiss["beta_y"], "#3498db", lw=2, label=r"$\beta_y$")
ax.set_ylabel("β [m]")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

ax = axes[1]
ax.plot(s, twiss["Dx"] * 100, "#2ecc71", lw=2, label=r"$D_x$ [cm]")
ax.axhline(0, color="gray", lw=0.7, ls="--")
ax.set_xlabel("s [m]")
ax.set_ylabel("$D_x$ [cm]")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Shade element types
_s0 = 0.0
for e in LAT_LIN.elements:
    _s1 = _s0 + e.length
    if type(e).__name__ == "Dipole":
        axes[0].axvspan(_s0, _s1, alpha=0.10, color="gold", linewidth=0)
        axes[1].axvspan(_s0, _s1, alpha=0.10, color="gold", linewidth=0)
    elif type(e).__name__ == "Quadrupole":
        c = "#c0392b" if e.k1 > 0 else "#2980b9"
        axes[0].axvspan(_s0, _s1, alpha=0.15, color=c, linewidth=0)
    _s0 = _s1

plt.tight_layout()
fig.savefig("ring_twiss.png", dpi=150)
print(f"  Saved → ring_twiss.png")
plt.close(fig)

# ── Poincaré section: linear vs nonlinear ────────────────────────────────────

N_TURNS = 512
N_PART = 6          # amplitude scan: 6 different amplitudes

# monitor index (at OBS marker)
obs_idx = [i for i, e in enumerate(LAT_LIN.elements) if e.name == "OBS"][0]

t0 = LAT_LIN.twiss_at_start()
beta0 = t0["beta_x"]

fig2, (ax_lin, ax_nlin) = plt.subplots(1, 2, figsize=(11, 5))
fig2.suptitle(
    f"Poincaré Section  ({N_TURNS} turns)  — $Q_x={Qx:.3f}$, $Q_y={Qy:.3f}$",
    fontsize=11,
)

colors = plt.cm.plasma(np.linspace(0.05, 0.95, N_PART))

for panel, (ax, lat, title) in enumerate([
    (ax_lin,  LAT_LIN,          "Linear (no sextupoles)"),
    (ax_nlin, build_ring(k2=8.0), "Nonlinear (k₂ = 8.0 m⁻³)"),
]):
    for i_p, color in enumerate(colors):
        # Launch at different amplitudes from 0.5 to 3 mm
        amp = (i_p + 1) * 0.5e-3  # [m]
        pts = np.array([[amp, 0.0, 0.0, 0.0, 0.0, 0.0]])
        hist = track(pts, lat, n_turns=N_TURNS, monitor_indices=[obs_idx])
        # hist: (N_TURNS+1, 1, 1, 6)
        xh  = hist[:, 0, 0, 0] * 1e3   # [mm]
        pxh = hist[:, 0, 0, 1] * 1e3   # [mrad]
        ax.scatter(xh, pxh, s=2, color=color, alpha=0.8, linewidths=0,
                   label=f"x₀={amp*1e3:.1f} mm" if i_p < 3 else "_")

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("px [mrad]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")

ax_lin.legend(fontsize=8, markerscale=3)
plt.tight_layout()
fig2.savefig("ring_poincare.png", dpi=150)
print(f"  Saved → ring_poincare.png")
plt.close(fig2)

# ── Tune diagram ─────────────────────────────────────────────────────────────

fig3, ax = plt.subplots(figsize=(6, 6))
ax.set_title("Tune Diagram (resonances up to 4th order)", fontsize=11)
ax.set_xlabel("$Q_x$")
ax.set_ylabel("$Q_y$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Draw resonance lines n*Qx + m*Qy = p up to order |n|+|m|=4
for order in range(1, 5):
    for n in range(-order, order + 1):
        for m in range(-order, order + 1):
            if abs(n) + abs(m) != order:
                continue
            if m == 0 and n == 0:
                continue
            for p in range(-order, order + 1):
                # n*Qx + m*Qy = p  → Qy = (p - n*Qx)/m  if m != 0
                # or Qx = p/n  if m == 0
                lw = 0.5 if order > 2 else 0.9
                alpha_r = 0.25 if order > 2 else 0.5
                color_r = {1: "#e74c3c", 2: "#e67e22", 3: "#95a5a6", 4: "#bdc3c7"}[order]
                if m != 0:
                    q_arr = np.linspace(0, 1, 200)
                    qy_arr = (p - n * q_arr) / m
                    mask = (qy_arr >= 0) & (qy_arr <= 1)
                    if mask.any():
                        ax.plot(q_arr[mask], qy_arr[mask], color=color_r,
                                lw=lw, alpha=alpha_r)
                else:
                    if n != 0 and 0 <= p / n <= 1:
                        ax.axvline(p / n, color=color_r, lw=lw, alpha=alpha_r)

# Operating point
ax.plot(Qx % 1, Qy % 1, "ro", ms=8, zorder=10, label=f"($Q_x$, $Q_y$) = ({Qx:.3f}, {Qy:.3f})")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

# Legend for resonance orders
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=c, lw=1.5, label=f"Order {o}")
           for o, c in zip([1,2,3,4], ["#e74c3c","#e67e22","#95a5a6","#bdc3c7"])]
ax.legend(handles=handles + [ax.lines[-1]], fontsize=8, loc="upper right")

plt.tight_layout()
fig3.savefig("ring_tune.png", dpi=150)
print(f"  Saved → ring_tune.png")
plt.close(fig3)

# ── Dynamic aperture ──────────────────────────────────────────────────────────

print("\n  Computing dynamic aperture scan (may take ~10 s)…")

LAT_NL = build_ring(k2=8.0)
N_ANGLE = 24
N_TURNS_DA = 256
obs_nlin = [i for i, e in enumerate(LAT_NL.elements) if e.name == "OBS"][0]

angles = np.linspace(0, 2 * np.pi, N_ANGLE, endpoint=False)
apertures = []

for angle in angles:
    # Binary search for maximum stable amplitude along this angle
    lo, hi = 0.0, 20e-3  # search range [m]
    for _ in range(14):   # ~4 significant figures
        mid = (lo + hi) / 2.0
        x0 = mid * np.cos(angle)
        px0 = mid * np.sin(angle) / np.sqrt(beta0)   # normalise by sqrt(beta)
        pts = np.array([[x0, px0, 0.0, 0.0, 0.0, 0.0]])
        hist = track(pts, LAT_NL, n_turns=N_TURNS_DA, monitor_indices=[obs_nlin])
        # Check for particle loss (amplitude growth > 10× initial)
        max_amp = np.max(np.abs(hist[:, 0, 0, :4]))
        if max_amp < 100e-3:  # still confined
            lo = mid
        else:
            hi = mid
    apertures.append(lo)

apertures = np.array(apertures)

fig4, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
ax.plot(np.append(angles, angles[0]),
        np.append(apertures, apertures[0]) * 1e3,
        "o-", color="#e74c3c", lw=2, ms=5)
ax.fill(np.append(angles, angles[0]),
        np.append(apertures, apertures[0]) * 1e3,
        color="#e74c3c", alpha=0.2)
ax.set_title(f"Dynamic Aperture — $k_2 = 8.0$ m⁻³\n({N_TURNS_DA} turns)", fontsize=10)
ax.set_rlabel_position(45)
ax.yaxis.set_tick_params(labelsize=8)
fig4.text(0.5, 0.02, "Amplitude at OBS [mm]", ha="center", fontsize=9)

plt.tight_layout()
fig4.savefig("ring_aperture.png", dpi=150)
print(f"  Saved → ring_aperture.png")
plt.close(fig4)

print("\n  Done.")
