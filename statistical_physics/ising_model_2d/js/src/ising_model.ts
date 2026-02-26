/**
 * 2D Ising Model — Metropolis-Hastings Monte Carlo
 *
 * Square N×N lattice with periodic boundary conditions.
 * Hamiltonian:  H = -J Σ_{<i,j>} s_i s_j   (nearest neighbours)
 * Spins:        s_i ∈ {-1, +1}
 *
 * Onsager's exact critical temperature (J = k_B = 1):
 *   T_c = 2 / ln(1 + √2) ≈ 2.2692
 */

export const T_CRITICAL = 2.0 / Math.log(1.0 + Math.sqrt(2.0)); // ≈ 2.2692

/** Running statistics (over recent sweeps). */
export interface Stats {
  eMean: number;   // ⟨E⟩ / N²
  mMean: number;   // ⟨|M|⟩ / N²
  sweeps: number;  // total sweeps performed
}

export class IsingModel2D {
  readonly n: number;
  readonly j: number;
  /** Flat Int8Array of ±1 spins, index = row * n + col. */
  lattice: Int8Array;

  private temp: number;
  private exp4: number;  // exp(-4J/T)
  private exp8: number;  // exp(-8J/T)

  // Running accumulators
  private eAcc = 0.0;
  private mAcc = 0.0;
  private count = 0;
  sweeps = 0;

  constructor(n: number, j = 1.0, seed = 42) {
    this.n = n;
    this.j = j;
    this.temp = 2.0;
    this.exp4 = Math.exp(-j * 4.0 / this.temp);
    this.exp8 = Math.exp(-j * 8.0 / this.temp);

    // Xorshift32 PRNG for reproducible initialisation
    let r = seed | 0 || 42;
    const xr = () => {
      r ^= r << 13; r ^= r >>> 17; r ^= r << 5;
      return (r >>> 0) / 0x1_0000_0000;
    };

    this.lattice = new Int8Array(n * n);
    for (let i = 0; i < n * n; i++) {
      this.lattice[i] = xr() < 0.5 ? 1 : -1;
    }
  }

  // ---------------------------------------------------------------------------
  // Temperature control
  // ---------------------------------------------------------------------------

  setTemperature(t: number): void {
    this.temp = t;
    this.exp4 = Math.exp(-this.j * 4.0 / t);
    this.exp8 = Math.exp(-this.j * 8.0 / t);
    // Reset accumulators when T changes
    this.eAcc = 0; this.mAcc = 0; this.count = 0;
  }

  getTemperature(): number { return this.temp; }

  // ---------------------------------------------------------------------------
  // Observables (instantaneous)
  // ---------------------------------------------------------------------------

  energy(): number {
    const n = this.n;
    const s = this.lattice;
    let e = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const spin = s[i * n + j];
        e += spin * (s[i * n + (j + 1) % n] + s[((i + 1) % n) * n + j]);
      }
    }
    return -this.j * e;
  }

  magnetization(): number {
    let m = 0;
    for (let i = 0; i < this.lattice.length; i++) m += this.lattice[i];
    return m;
  }

  // ---------------------------------------------------------------------------
  // Monte Carlo sweep
  // ---------------------------------------------------------------------------

  /**
   * One full MC sweep (N² single-spin flip attempts).
   * Uses a simple linear congruential generator for speed.
   */
  step(): void {
    const n = this.n;
    const n2 = n * n;
    const s = this.lattice;
    const exp4 = this.exp4;
    const exp8 = this.exp8;

    // LCG PRNG (fast, not cryptographic)
    let seed = (Math.random() * 0x1_0000_0000) | 0;
    const next = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
      return (seed >>> 0) / 0x1_0000_0000;
    };

    for (let k = 0; k < n2; k++) {
      const i = (next() * n) | 0;
      const j = (next() * n) | 0;
      const spin = s[i * n + j];

      const nb = s[((i - 1 + n) % n) * n + j]
               + s[((i + 1) % n) * n + j]
               + s[i * n + (j - 1 + n) % n]
               + s[i * n + (j + 1) % n];

      const dEoverJ = 2 * spin * nb;

      let accept: boolean;
      if (dEoverJ <= 0) {
        accept = true;
      } else if (dEoverJ === 4) {
        accept = next() < exp4;
      } else if (dEoverJ === 8) {
        accept = next() < exp8;
      } else {
        accept = false;
      }
      if (accept) s[i * n + j] = -spin as -1 | 1;
    }

    this.sweeps++;
    const n2f = n2;
    const e = this.energy() / n2f;
    const m = Math.abs(this.magnetization()) / n2f;
    this.eAcc += e;
    this.mAcc += m;
    this.count++;
  }

  stats(): Stats {
    const c = Math.max(this.count, 1);
    return {
      eMean:  this.eAcc / c,
      mMean:  this.mAcc / c,
      sweeps: this.sweeps,
    };
  }

  resetStats(): void {
    this.eAcc = 0; this.mAcc = 0; this.count = 0;
  }

  // ---------------------------------------------------------------------------
  // Preset initialisations
  // ---------------------------------------------------------------------------

  setAllUp(): void { this.lattice.fill(1); this.resetStats(); }
  setAllDown(): void { this.lattice.fill(-1); this.resetStats(); }
  randomise(): void {
    for (let i = 0; i < this.lattice.length; i++) {
      this.lattice[i] = Math.random() < 0.5 ? 1 : -1;
    }
    this.resetStats();
  }
}
