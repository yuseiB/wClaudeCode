"""Double pendulum dynamics — exact nonlinear Lagrangian equations.

Physical setup
--------------
The pivot is fixed at the origin.  Bob-1 hangs on a massless rigid rod of
length L₁ at angle θ₁ measured from the downward vertical.  Bob-2 hangs from
bob-1 on rod L₂ at angle θ₂ (same convention).

Lagrangian  L = T − V
~~~~~~~~~~~~~~~~~~~~~
Positions (y positive upward):

    x₁ =  L₁ sinθ₁               y₁ = −L₁ cosθ₁
    x₂ =  L₁ sinθ₁ + L₂ sinθ₂   y₂ = −L₁ cosθ₁ − L₂ cosθ₂

Kinetic energy:

    T = ½m₁L₁²ω₁² + ½m₂[L₁²ω₁² + L₂²ω₂² + 2L₁L₂ω₁ω₂ cos(θ₁−θ₂)]

Potential energy:

    V = −(m₁+m₂)gL₁ cosθ₁ − m₂gL₂ cosθ₂

Euler–Lagrange equations (solved for the accelerations):

    Δ  ≡ θ₁ − θ₂
    D  ≡ 2m₁ + m₂ − m₂ cos 2Δ          (common denominator)

    θ̈₁ = [−g(2m₁+m₂)sinθ₁  − m₂g sin(θ₁−2θ₂)
            − 2sinΔ · m₂(ω₂²L₂ + ω₁²L₁ cosΔ)] / (L₁ D)

    θ̈₂ = [2sinΔ · (ω₁²L₁(m₁+m₂) + g(m₁+m₂)cosθ₁
            + ω₂²L₂ m₂ cosΔ)] / (L₂ D)

No small-angle approximation is used.  The integrator is scipy's adaptive
RK45 (Dormand–Prince) with tight tolerances so that energy drift stays below
a few parts in 10⁶ over typical run times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class DoublePendulum:
    """Exact nonlinear double-pendulum model.

    Parameters
    ----------
    m1, m2 : float
        Masses of the two bobs [kg].
    L1, L2 : float
        Rod lengths [m].
    g : float
        Gravitational acceleration [m s⁻²].
    """

    m1: float = 1.0
    m2: float = 1.0
    L1: float = 1.0
    L2: float = 1.0
    g: float = 9.81

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------

    def eom(self, _t: float, state: np.ndarray) -> np.ndarray:
        """RHS of the first-order ODE system.

        State vector: [θ₁, ω₁, θ₂, ω₂]
        Returns:      [ω₁, α₁, ω₂, α₂]   (α = angular acceleration)
        """
        th1, om1, th2, om2 = state
        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g

        delta = th1 - th2
        sin_d = np.sin(delta)
        cos_d = np.cos(delta)
        D = 2 * m1 + m2 - m2 * np.cos(2 * delta)   # denominator (never zero)

        alpha1 = (
            -g * (2 * m1 + m2) * np.sin(th1)
            - m2 * g * np.sin(th1 - 2 * th2)
            - 2 * sin_d * m2 * (om2 ** 2 * L2 + om1 ** 2 * L1 * cos_d)
        ) / (L1 * D)

        alpha2 = (
            2 * sin_d * (
                om1 ** 2 * L1 * (m1 + m2)
                + g * (m1 + m2) * np.cos(th1)
                + om2 ** 2 * L2 * m2 * cos_d
            )
        ) / (L2 * D)

        return np.array([om1, alpha1, om2, alpha2])

    def energy(self, state: np.ndarray) -> float:
        """Total mechanical energy  E = T + V  (conserved quantity).

        Useful as a numerical sanity check: drift in E indicates
        integrator error.
        """
        th1, om1, th2, om2 = state
        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g

        T = (
            0.5 * m1 * L1 ** 2 * om1 ** 2
            + 0.5 * m2 * (
                L1 ** 2 * om1 ** 2
                + L2 ** 2 * om2 ** 2
                + 2 * L1 * L2 * om1 * om2 * np.cos(th1 - th2)
            )
        )
        V = -(m1 + m2) * g * L1 * np.cos(th1) - m2 * g * L2 * np.cos(th2)
        return float(T + V)

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def simulate(
        self,
        theta1_0: float,
        theta2_0: float,
        omega1_0: float = 0.0,
        omega2_0: float = 0.0,
        t_end: float = 20.0,
        dt: float = 0.01,
        method: str = "RK45",
    ) -> dict:
        """Integrate the equations of motion and return a solution dict.

        Parameters
        ----------
        theta1_0, theta2_0 : float
            Initial angles [radians].
        omega1_0, omega2_0 : float
            Initial angular velocities [rad s⁻¹].
        t_end : float
            Simulation end time [s].
        dt : float
            Output time step [s].  (The integrator uses adaptive steps
            internally; this only controls the sampling of the output.)
        method : str
            scipy solve_ivp integrator.  "RK45" (Dormand–Prince, default)
            is a good all-round choice.  "DOP853" gives higher-order
            accuracy at the cost of more function evaluations.

        Returns
        -------
        dict with keys
            t                 : (N,) time array [s]
            theta1, omega1    : (N,) angle [rad] and angular velocity [rad/s] of bob-1
            theta2, omega2    : (N,) same for bob-2
            x1, y1            : (N,) Cartesian position of bob-1 [m]
            x2, y2            : (N,) Cartesian position of bob-2 [m]
            energy            : (N,) total mechanical energy [J]
        """
        t_eval = np.arange(0.0, t_end, dt)
        state0 = np.array([theta1_0, omega1_0, theta2_0, omega2_0])

        sol = solve_ivp(
            self.eom,
            (0.0, t_end),
            state0,
            method=method,
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-11,
        )

        th1, om1, th2, om2 = sol.y

        x1 = self.L1 * np.sin(th1)
        y1 = -self.L1 * np.cos(th1)
        x2 = x1 + self.L2 * np.sin(th2)
        y2 = y1 - self.L2 * np.cos(th2)

        E = np.array([self.energy(sol.y[:, k]) for k in range(sol.y.shape[1])])

        return dict(
            t=sol.t,
            theta1=th1, omega1=om1,
            theta2=th2, omega2=om2,
            x1=x1, y1=y1,
            x2=x2, y2=y2,
            energy=E,
        )
