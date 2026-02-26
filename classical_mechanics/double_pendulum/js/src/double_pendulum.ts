/**
 * Exact nonlinear double pendulum (Lagrangian formulation).
 *
 * State vector: [θ₁, ω₁, θ₂, ω₂].
 * Angles measured from the downward vertical.
 * Integrator: fixed-step RK4.
 */

export interface State {
  theta1: number; // rad
  omega1: number; // rad/s
  theta2: number; // rad
  omega2: number; // rad/s
}

// ── Vector helpers for RK4 ────────────────────────────────────────────────────

type Vec4 = readonly [number, number, number, number];

function stateToVec(s: State): Vec4 {
  return [s.theta1, s.omega1, s.theta2, s.omega2];
}

function vecToState([theta1, omega1, theta2, omega2]: Vec4): State {
  return { theta1, omega1, theta2, omega2 };
}

function addV(a: Vec4, b: Vec4): Vec4 {
  return [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]];
}

function scaleV(a: Vec4, k: number): Vec4 {
  return [a[0]*k, a[1]*k, a[2]*k, a[3]*k];
}

// ── Main class ────────────────────────────────────────────────────────────────

export class DoublePendulum {
  constructor(
    public readonly m1 = 1.0,
    public readonly m2 = 1.0,
    public readonly L1 = 1.0,
    public readonly L2 = 1.0,
    public readonly g  = 9.81,
  ) {}

  /** RHS of the first-order ODE system. */
  private eomVec([th1, w1, th2, w2]: Vec4): Vec4 {
    const dlt = th1 - th2;
    const D   = 2*this.m1 + this.m2 - this.m2 * Math.cos(2*dlt);

    const a1 = (
      -this.g * (2*this.m1 + this.m2) * Math.sin(th1)
      - this.m2 * this.g * Math.sin(th1 - 2*th2)
      - 2 * Math.sin(dlt) * this.m2 * (w2*w2*this.L2 + w1*w1*this.L1*Math.cos(dlt))
    ) / (this.L1 * D);

    const a2 = (
      2 * Math.sin(dlt) * (
        w1*w1*this.L1*(this.m1 + this.m2)
        + this.g*(this.m1 + this.m2)*Math.cos(th1)
        + w2*w2*this.L2*this.m2*Math.cos(dlt)
      )
    ) / (this.L2 * D);

    return [w1, a1, w2, a2];
  }

  /** Single fixed-step RK4 advance. */
  step(s: State, dt: number): State {
    const y  = stateToVec(s);
    const k1 = this.eomVec(y);
    const k2 = this.eomVec(addV(y, scaleV(k1, dt/2)));
    const k3 = this.eomVec(addV(y, scaleV(k2, dt/2)));
    const k4 = this.eomVec(addV(y, scaleV(k3, dt)));
    return vecToState(addV(y, scaleV(
      [k1[0]+2*k2[0]+2*k3[0]+k4[0],
       k1[1]+2*k2[1]+2*k3[1]+k4[1],
       k1[2]+2*k2[2]+2*k3[2]+k4[2],
       k1[3]+2*k2[3]+2*k3[3]+k4[3]],
      dt / 6,
    )));
  }

  /** Kinetic energy T. */
  kineticEnergy(s: State): number {
    const { theta1: th1, omega1: w1, theta2: th2, omega2: w2 } = s;
    const dlt = th1 - th2;
    return (
      0.5*this.m1*this.L1**2*w1**2 +
      0.5*this.m2*(this.L1**2*w1**2 + this.L2**2*w2**2
                   + 2*this.L1*this.L2*w1*w2*Math.cos(dlt))
    );
  }

  /** Potential energy V. */
  potentialEnergy(s: State): number {
    return -this.g * (
      (this.m1 + this.m2)*this.L1*Math.cos(s.theta1)
      + this.m2*this.L2*Math.cos(s.theta2)
    );
  }

  /** Total mechanical energy E = T + V. */
  energy(s: State): number {
    return this.kineticEnergy(s) + this.potentialEnergy(s);
  }

  /** Cartesian position of bob-1. */
  bob1(s: State): { x: number; y: number } {
    return {
      x:  this.L1 * Math.sin(s.theta1),
      y: -this.L1 * Math.cos(s.theta1),
    };
  }

  /** Cartesian position of bob-2. */
  bob2(s: State): { x: number; y: number } {
    const b = this.bob1(s);
    return {
      x: b.x + this.L2 * Math.sin(s.theta2),
      y: b.y - this.L2 * Math.cos(s.theta2),
    };
  }
}
