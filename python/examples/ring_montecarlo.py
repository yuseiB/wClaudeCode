"""Monte Carlo storage-ring simulation — three scenarios.

Scenarios
---------
1. **Linear, no chromaticity** — emittance conserved exactly; phase-space
   ellipse rotates without distortion.
2. **Chromatic tune spread** (ξ = −2, σ_δ = 1×10⁻³) — particles with
   different momenta precess at different rates; the ensemble emittance
   grows as the distribution decohere.
3. **Sextupole + aperture** (k₂L/2 = 5 m⁻², half-aperture = 50 mm) —
   nonlinear kicks feed-down to large-amplitude particles, driving them
   beyond the aperture; survival rate falls over turns.

Output: ring_montecarlo.png (3 × 2 panel figure, saved at repo root).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow running directly without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mathphys.storage_ring import BeamParams, RingParams, StorageRing

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

N_TURNS = 2000
N_PART = 2000

scenarios = [
    {
        "label": "Linear optics\n(no chromaticity, no sextupole)",
        "ring": RingParams(
            tune=0.28,
            beta=10.0,
            alpha=0.5,
            chromaticity=0.0,
            sextupole_strength=0.0,
            aperture=np.inf,
        ),
        "beam": BeamParams(
            emittance=1e-6,
            momentum_spread=1e-3,
            n_particles=N_PART,
            seed=0,
        ),
    },
    {
        "label": "Chromatic tune spread\n(ξ = −2,  σ_δ = 1×10⁻³)",
        "ring": RingParams(
            tune=0.28,
            beta=10.0,
            alpha=0.5,
            chromaticity=-2.0,
            sextupole_strength=0.0,
            aperture=np.inf,
        ),
        "beam": BeamParams(
            emittance=1e-6,
            momentum_spread=1e-3,
            n_particles=N_PART,
            seed=0,
        ),
    },
    {
        "label": "Sextupole + aperture\n(k₂L/2 = 20 m⁻²,  A = 20 mm,  ε = 3×10⁻⁵ m·rad)",
        "ring": RingParams(
            tune=0.28,
            beta=10.0,
            alpha=0.0,
            chromaticity=-2.0,
            sextupole_strength=20.0,
            aperture=0.02,  # 20 mm half-aperture
        ),
        "beam": BeamParams(
            emittance=3e-5,   # large halo — σ_x ≈ 17 mm, 3σ reaches aperture
            momentum_spread=1e-3,
            n_particles=N_PART,
            seed=0,
        ),
    },
]

# ---------------------------------------------------------------------------
# Run tracking
# ---------------------------------------------------------------------------

results = []
for sc in scenarios:
    label_oneline = sc["label"].replace("\n", " ")
    print(f"Tracking: {label_oneline} ...", flush=True)
    tracker = StorageRing(sc["ring"], sc["beam"])
    res = tracker.track(n_turns=N_TURNS)
    results.append(res)
    eps_ratio = res.emittance[-1] / res.emittance[0]
    print(
        f"  survival={res.survival[-1]*100:.1f}%  "
        f"ε/ε₀={eps_ratio:.4f}"
    )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle(
    "Monte Carlo Storage-Ring Tracking  (no collective instabilities)",
    fontsize=13,
    fontweight="bold",
)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, (sc, res) in enumerate(zip(scenarios, results)):
    ax_ev, ax_ps = axes[i]

    # ── Left panel: emittance & survival vs turn ────────────────────────────
    turns = res.turns
    eps_norm = res.emittance / res.emittance[0]
    ax_ev.plot(turns, eps_norm, color=colors[i], lw=1.2, label="ε / ε₀")
    ax_ev.set_xlim(0, N_TURNS)
    ax_ev.set_xlabel("Turn", fontsize=8)
    ax_ev.set_ylabel("ε / ε₀", fontsize=8)
    ax_ev.tick_params(labelsize=7)
    ax_ev.set_title(sc["label"], fontsize=9)

    ax_surv = ax_ev.twinx()
    ax_surv.plot(
        turns,
        res.survival * 100,
        color=colors[i],
        lw=0.9,
        ls="--",
        alpha=0.55,
        label="Survival %",
    )
    ax_surv.set_ylim(0, 105)
    ax_surv.set_ylabel("Survival [%]", fontsize=8, color="grey")
    ax_surv.tick_params(axis="y", labelcolor="grey", labelsize=7)

    lines1, lbl1 = ax_ev.get_legend_handles_labels()
    lines2, lbl2 = ax_surv.get_legend_handles_labels()
    ax_ev.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7, loc="upper left")

    # ── Right panel: phase-space scatter (turn 0 vs final) ─────────────────
    snap_turns = sorted(res.snapshots.keys())
    t0, tf = snap_turns[0], snap_turns[-1]
    x0, xp0 = res.snapshots[t0]
    xf, xpf = res.snapshots[tf]

    ax_ps.scatter(
        x0 * 1e3, xp0 * 1e3, s=1, alpha=0.25, color="steelblue",
        label=f"Turn {t0}  (N={len(x0)})",
    )
    ax_ps.scatter(
        xf * 1e3, xpf * 1e3, s=1, alpha=0.25, color="tomato",
        label=f"Turn {tf}  (N={len(xf)})",
    )
    ax_ps.set_xlabel("x  [mm]", fontsize=8)
    ax_ps.set_ylabel("x′  [mrad]", fontsize=8)
    ax_ps.tick_params(labelsize=7)
    ax_ps.legend(fontsize=7, markerscale=5)
    ax_ps.set_title("Phase space (x, x′)", fontsize=9)

plt.tight_layout()

out = os.path.join(
    os.path.dirname(__file__), "..", "..", "ring_montecarlo.png"
)
out = os.path.abspath(out)
plt.savefig(out, dpi=150)
print(f"\nSaved → {out}")
plt.show()
