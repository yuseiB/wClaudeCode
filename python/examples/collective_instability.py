"""Monte Carlo collective-instability demonstration.

Three scenarios (left = |x̄(turn)| log scale, right = phase-space scatter):

  Scenario 1 — Rigid-bunch BELOW Landau-damping threshold
    κ = 0.001 rad/m  (threshold ≈ 0.005–0.008 rad/m)
    Centroid excited to 2 mm, then damped by Landau damping (tune spread).

  Scenario 2 — Rigid-bunch ABOVE threshold
    κ = 0.05 rad/m
    Coherent kick overcomes Landau damping; centroid oscillation is sustained.
    With tight aperture (10 mm), more particles are lost than without kick.

  Scenario 3 — Head-tail: ξ = 0 (unstable) vs ξ = −2 (Landau-stable)
    Sliced wake drives centroid; chromaticity provides Landau damping.

Notes
-----
  x0_offset = 2 mm >> σ_x/√N ≈ 58 μm so the coherent signal dominates
  finite-ensemble noise.  Collective instabilities are NOT modelled; each
  particle follows its own single-particle map except for the coherent kick.

Output: collective_instability.png (saved at repo root).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mathphys.collective import CollectiveParams, CollectiveRing
from mathphys.storage_ring import BeamParams, RingParams

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_TURNS = 800
N_PART = 3000

RING = RingParams(tune=0.28, beta=10.0, alpha=0.0,
                  chromaticity=-2.0, sextupole_strength=0.0, aperture=0.2)
BEAM = BeamParams(emittance=1e-6, momentum_spread=1e-3,
                  n_particles=N_PART, seed=42)

# σ_Q = |ξ| σ_δ = 0.002  →  κ_th ≈ σ_Q / (β sin²(2πQ)) ≈ 0.005–0.008 rad/m
KTH = abs(RING.chromaticity) * BEAM.momentum_spread / (
    RING.beta * np.sin(2 * np.pi * RING.tune) ** 2
)
print(f"Estimated Landau threshold  κ_th ≈ {KTH:.4f} rad/m")

scenarios = [
    {
        "label": f"Rigid-bunch  BELOW threshold\n"
                 f"(κ = 0.001 rad/m,  κ_th ≈ {KTH:.4f} rad/m)",
        "ring": RING,
        "beam": BEAM,
        "coll": CollectiveParams(mode="transverse", kappa=0.001, x0_offset=2e-3),
    },
    {
        "label": f"Rigid-bunch  ABOVE threshold\n"
                 f"(κ = 0.05 rad/m,  aperture = 10 mm)",
        "ring": RingParams(tune=0.28, beta=10.0, alpha=0.0,
                           chromaticity=-2.0, aperture=0.010),
        "beam": BEAM,
        "coll": CollectiveParams(mode="transverse", kappa=0.05, x0_offset=2e-3),
    },
]

HT_RING_NO = RingParams(tune=0.28, beta=10.0, alpha=0.0,
                        chromaticity=0.0, aperture=0.2)
HT_RING_XI = RingParams(tune=0.28, beta=10.0, alpha=0.0,
                        chromaticity=-2.0, aperture=0.2)
HT_BEAM = BeamParams(emittance=1e-6, momentum_spread=1e-3,
                     n_particles=N_PART, seed=42)
HT_COLL = CollectiveParams(
    mode="headtail", wake_strength=0.005,
    wake_range=0.008, n_slices=15, sigma_z=0.01, x0_offset=2e-3,
)
ht_scenarios = [("ξ = 0  (no chromaticity)", HT_RING_NO, HT_BEAM, HT_COLL),
                ("ξ = −2  (chromaticity)", HT_RING_XI, HT_BEAM, HT_COLL)]

# ---------------------------------------------------------------------------
# Run tracking
# ---------------------------------------------------------------------------

results = []
for sc in scenarios:
    lbl = sc["label"].replace("\n", " ")
    print(f"Tracking: {lbl} ...", flush=True)
    tracker = CollectiveRing(sc["ring"], sc["beam"], sc["coll"])
    res = tracker.track(n_turns=N_TURNS)
    results.append(res)
    cx = np.abs(res.centroid_x)
    print(f"  survival={res.survival[-1]*100:.1f}%  "
          f"early|x̄|={np.mean(cx[10:100])*1e3:.3f}mm  "
          f"late|x̄|={np.mean(cx[600:800])*1e3:.3f}mm")

print("Tracking: Head-tail pair ...", flush=True)
ht_results = []
for lbl, ring, bm, coll in ht_scenarios:
    tracker = CollectiveRing(ring, bm, coll)
    res = tracker.track(n_turns=N_TURNS)
    ht_results.append((lbl, res))
    cx = np.abs(res.centroid_x)
    print(f"  [{lbl}]  early|x̄|={np.mean(cx[10:100])*1e3:.3f}mm  "
          f"late|x̄|={np.mean(cx[600:800])*1e3:.3f}mm")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle(
    "Monte Carlo Collective Instabilities  (rigid-bunch wake + sliced head-tail)",
    fontsize=13, fontweight="bold",
)
colors = ["#1f77b4", "#ff7f0e"]
turns = np.arange(N_TURNS + 1)

for row, (sc, res) in enumerate(zip(scenarios, results)):
    ax_c, ax_ps = axes[row]
    cx = np.abs(res.centroid_x)
    cx_plot = np.where(cx > 1e-8, cx, 1e-8)

    ax_c.semilogy(turns, cx_plot * 1e3, color=colors[row], lw=0.9, label="|x̄(n)|")
    ax_c.axhline(sc["coll"].x0_offset * 1e3, color="grey",
                 lw=0.7, ls=":", label="initial offset")
    ax_c.set_xlim(0, N_TURNS)
    ax_c.set_xlabel("Turn", fontsize=8)
    ax_c.set_ylabel("|x̄|  [mm]", fontsize=8)
    ax_c.tick_params(labelsize=7)
    ax_c.set_title(sc["label"], fontsize=9)

    ax_sv = ax_c.twinx()
    ax_sv.plot(turns, res.survival * 100, color=colors[row],
               lw=0.8, ls="--", alpha=0.5, label="Survival %")
    ax_sv.set_ylim(0, 105)
    ax_sv.set_ylabel("Survival [%]", fontsize=8, color="grey")
    ax_sv.tick_params(axis="y", labelcolor="grey", labelsize=7)
    l1, lb1 = ax_c.get_legend_handles_labels()
    l2, lb2 = ax_sv.get_legend_handles_labels()
    ax_c.legend(l1 + l2, lb1 + lb2, fontsize=7)

    snap_turns = sorted(res.snapshots.keys())
    x0, xp0 = res.snapshots[snap_turns[0]]
    xf, xpf = res.snapshots[snap_turns[-1]]
    ax_ps.scatter(x0 * 1e3, xp0 * 1e3, s=1, alpha=0.2, color="steelblue",
                  label=f"Turn {snap_turns[0]}")
    ax_ps.scatter(xf * 1e3, xpf * 1e3, s=1, alpha=0.2, color="tomato",
                  label=f"Turn {snap_turns[-1]}")
    ax_ps.set_xlabel("x  [mm]", fontsize=8)
    ax_ps.set_ylabel("x′  [mrad]", fontsize=8)
    ax_ps.tick_params(labelsize=7)
    ax_ps.legend(fontsize=7, markerscale=5)
    ax_ps.set_title("Phase space (x, x′)", fontsize=9)

# Row 3: head-tail comparison
ax_c3, ax_ps3 = axes[2]
ht_colors = ["#d62728", "#17becf"]
for (lbl, ht_res), col in zip(ht_results, ht_colors):
    cx = np.abs(ht_res.centroid_x)
    cx_plot = np.where(cx > 1e-8, cx, 1e-8)
    ax_c3.semilogy(turns, cx_plot * 1e3, color=col, lw=0.9, label=lbl)

ax_c3.axhline(HT_COLL.x0_offset * 1e3, color="grey", lw=0.7, ls=":", label="initial offset")
ax_c3.set_xlim(0, N_TURNS)
ax_c3.set_xlabel("Turn", fontsize=8)
ax_c3.set_ylabel("|x̄|  [mm]", fontsize=8)
ax_c3.tick_params(labelsize=7)
ax_c3.set_title("Head-tail: chromaticity stabilization", fontsize=9)
ax_c3.legend(fontsize=7)

_, ht_unstable = ht_results[0]
snap_turns = sorted(ht_unstable.snapshots.keys())
x0, xp0 = ht_unstable.snapshots[snap_turns[0]]
xf, xpf = ht_unstable.snapshots[snap_turns[-1]]
ax_ps3.scatter(x0 * 1e3, xp0 * 1e3, s=1, alpha=0.2, color="steelblue",
               label=f"Turn {snap_turns[0]}")
ax_ps3.scatter(xf * 1e3, xpf * 1e3, s=1, alpha=0.2, color="tomato",
               label=f"Turn {snap_turns[-1]}")
ax_ps3.set_xlabel("x  [mm]", fontsize=8)
ax_ps3.set_ylabel("x′  [mrad]", fontsize=8)
ax_ps3.tick_params(labelsize=7)
ax_ps3.legend(fontsize=7, markerscale=5)
ax_ps3.set_title("Phase space  (ξ = 0, head-tail)", fontsize=9)

plt.tight_layout()
out = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "collective_instability.png")
)
plt.savefig(out, dpi=150)
print(f"\nSaved → {out}")
plt.show()
