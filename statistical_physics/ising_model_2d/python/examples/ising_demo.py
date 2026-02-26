"""
2D Ising Model — Temperature Sweep Demo
========================================

Runs a Metropolis MC simulation across temperatures T = 1.0 … 4.0
and plots the phase transition (E, |M|, Cv, χ vs T) together with
lattice snapshots at three characteristic temperatures.

Output: ising2d.png (saved next to this script)
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mathphys.ising_model import IsingModel2D, T_CRITICAL

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N = 32          # lattice size (N×N)
N_THERM = 5_000  # thermalisation sweeps per temperature
N_MEAS = 10_000  # measurement sweeps per temperature
TEMPS = np.linspace(1.0, 4.0, 40)  # temperature grid

SNAPSHOT_TEMPS = [1.5, T_CRITICAL, 3.5]  # cold / critical / hot

# ---------------------------------------------------------------------------
# Temperature sweep
# ---------------------------------------------------------------------------
print(f"2D Ising Model  N={N}  T ∈ [{TEMPS[0]:.1f}, {TEMPS[-1]:.1f}]")
print(f"Onsager T_c = {T_CRITICAL:.4f}")
print(f"Running {len(TEMPS)} temperature points …")

results: list[dict] = []
for idx, T in enumerate(TEMPS):
    model = IsingModel2D(n=N, seed=idx)
    # Start ordered below Tc, disordered above
    model.lattice[:] = 1 if T < T_CRITICAL else model.lattice
    res = model.simulate(T, n_therm=N_THERM, n_measure=N_MEAS)
    res["T"] = T
    results.append(res)
    if (idx + 1) % 10 == 0:
        print(f"  {idx + 1}/{len(TEMPS)}  T={T:.2f}  |M|={res['M_mean']:.3f}  Cv={res['Cv']:.3f}")

T_arr = np.array([r["T"] for r in results])
E_arr = np.array([r["E_mean"] for r in results])
M_arr = np.array([r["M_mean"] for r in results])
Cv_arr = np.array([r["Cv"] for r in results])
chi_arr = np.array([r["chi"] for r in results])

# ---------------------------------------------------------------------------
# Lattice snapshots
# ---------------------------------------------------------------------------
print("Generating lattice snapshots …")
snapshots: list[tuple[float, np.ndarray]] = []
for T in SNAPSHOT_TEMPS:
    model = IsingModel2D(n=N, seed=99)
    model.lattice[:] = 1 if T < T_CRITICAL else model.lattice
    for _ in range(N_THERM):
        model.metropolis_step(T)
    snapshots.append((T, model.lattice.copy()))

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    f"2D Ising Model  (N={N}, J=1, PBC)\n"
    r"$H = -J\!\sum_{\langle i,j\rangle} s_i s_j$"
    f"   Onsager $T_c = {T_CRITICAL:.4f}$",
    fontsize=13,
)

# --- Observables (left column) ---
ax_e = fig.add_subplot(3, 3, 1)
ax_e.plot(T_arr, E_arr, "o-", ms=4, color="steelblue")
ax_e.axvline(T_CRITICAL, color="red", ls="--", lw=1, label=r"$T_c$")
ax_e.set_ylabel(r"$\langle E \rangle / N^2$")
ax_e.set_xlabel("Temperature $T$")
ax_e.legend(fontsize=9)
ax_e.set_title("Energy per site")

ax_m = fig.add_subplot(3, 3, 4)
ax_m.plot(T_arr, M_arr, "o-", ms=4, color="firebrick")
ax_m.axvline(T_CRITICAL, color="red", ls="--", lw=1, label=r"$T_c$")
ax_m.set_ylabel(r"$\langle |M| \rangle / N^2$")
ax_m.set_xlabel("Temperature $T$")
ax_m.legend(fontsize=9)
ax_m.set_title("Magnetisation per site (order parameter)")

ax_cv = fig.add_subplot(3, 3, 2)
ax_cv.plot(T_arr, Cv_arr, "o-", ms=4, color="darkorange")
ax_cv.axvline(T_CRITICAL, color="red", ls="--", lw=1, label=r"$T_c$")
ax_cv.set_ylabel(r"$C_v / N^2$")
ax_cv.set_xlabel("Temperature $T$")
ax_cv.legend(fontsize=9)
ax_cv.set_title("Specific heat")

ax_chi = fig.add_subplot(3, 3, 5)
ax_chi.plot(T_arr, chi_arr, "o-", ms=4, color="purple")
ax_chi.axvline(T_CRITICAL, color="red", ls="--", lw=1, label=r"$T_c$")
ax_chi.set_ylabel(r"$\chi / N^2$")
ax_chi.set_xlabel("Temperature $T$")
ax_chi.legend(fontsize=9)
ax_chi.set_title("Susceptibility")

# --- Snapshots (right column) ---
labels = [
    f"Low T = {SNAPSHOT_TEMPS[0]:.1f}  (ordered)",
    f"Critical T ≈ {T_CRITICAL:.2f}",
    f"High T = {SNAPSHOT_TEMPS[2]:.1f}  (disordered)",
]
for col, (T_snap, lat) in enumerate(snapshots):
    ax = fig.add_subplot(3, 3, col + 7)
    ax.imshow(lat, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(labels[col], fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
out_path = Path(__file__).parent / "ising2d.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
