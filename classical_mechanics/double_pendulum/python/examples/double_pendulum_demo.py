"""Double-pendulum demo: time evolution, trajectory, phase portrait, energy check.

Run from the project root:
    python python/examples/double_pendulum_demo.py

Output: double_pendulum.png  (four-panel figure)
"""

import numpy as np
import matplotlib.pyplot as plt

from mathphys.double_pendulum import DoublePendulum


def main() -> None:
    dp = DoublePendulum(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81)

    # Initial conditions: both bobs at rest, large angles → chaotic regime
    theta1_0 = np.radians(120.0)
    theta2_0 = np.radians(-30.0)

    print(f"Simulating double pendulum  θ₁₀={np.degrees(theta1_0):.0f}°  θ₂₀={np.degrees(theta2_0):.0f}°")
    sol = dp.simulate(theta1_0, theta2_0, t_end=30.0, dt=0.005)
    print(f"  steps: {len(sol['t'])}  |  E₀ = {sol['energy'][0]:.4f} J")

    t   = sol["t"]
    th1 = np.degrees(sol["theta1"])
    th2 = np.degrees(sol["theta2"])
    E   = sol["energy"]
    dE  = (E - E[0]) / abs(E[0]) * 100   # relative energy drift [%]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Double Pendulum — nonlinear dynamics\n"
        f"m₁=m₂=1 kg, L₁=L₂=1 m, θ₁₀={np.degrees(theta1_0):.0f}°, θ₂₀={np.degrees(theta2_0):.0f}°",
        fontsize=12,
    )

    # ── 1. Angular evolution ─────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t, th1, lw=0.8, color="steelblue", label="θ₁")
    ax.plot(t, th2, lw=0.8, color="tomato",    label="θ₂", alpha=0.85)
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("angle  [°]")
    ax.set_title("Angular evolution")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── 2. Cartesian trajectories ────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(sol["x1"], sol["y1"], lw=0.5, color="steelblue", alpha=0.5, label="bob-1")
    ax.plot(sol["x2"], sol["y2"], lw=0.5, color="tomato",    alpha=0.5, label="bob-2")
    # mark the final position
    ax.scatter(sol["x2"][-1], sol["y2"][-1], color="tomato", s=25, zorder=5)
    ax.scatter(sol["x1"][-1], sol["y1"][-1], color="steelblue", s=25, zorder=5)
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title("Trajectory (Cartesian)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 3. Phase portrait of bob-2 ───────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(sol["theta2"], sol["omega2"], lw=0.4, color="tomato", alpha=0.6)
    ax.set_xlabel("θ₂  [rad]")
    ax.set_ylabel("ω₂  [rad s⁻¹]")
    ax.set_title("Phase portrait — bob-2")
    ax.grid(True, alpha=0.3)

    # ── 4. Energy conservation (numerical error) ─────────────────────────
    ax = axes[1, 1]
    ax.plot(t, dE, lw=0.7, color="purple")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("ΔE / E₀  [%]")
    ax.set_title("Energy conservation (integrator error)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "double_pendulum.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
