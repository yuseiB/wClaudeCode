/**
 * em_physics.ts — EM cavity and waveguide analytical solutions (TypeScript)
 *
 * Physical constants (SI):
 *   c  = 2.99792458e8 m/s
 *   μ₀ = 4π×10⁻⁷ H/m
 *   ε₀ = 1/(μ₀c²) F/m
 *   η₀ ≈ 376.73 Ω
 */

// ── Physical constants ───────────────────────────────────────────────────────

export const C_LIGHT = 2.99792458e8;
export const MU0     = 4e-7 * Math.PI;
export const EPS0    = 1.0 / (MU0 * C_LIGHT * C_LIGHT);
export const ETA0    = MU0 * C_LIGHT;

// ── Bessel function table (zeros) ────────────────────────────────────────────

/** χ_mn: n-th zero of J_m (TM cutoff) */
const TM_ZEROS: number[][] = [
  [2.4048, 5.5201, 8.6537],  // m=0
  [3.8317, 7.0156, 10.1735], // m=1
  [5.1356, 8.4172, 11.6198], // m=2
  [6.3802, 9.7610, 13.0152], // m=3
  [7.5883, 11.0647, 14.3725],// m=4
];

/** χ'_mn: n-th zero of J_m' (TE cutoff) */
const TE_ZEROS: number[][] = [
  [3.8317, 7.0156, 10.1735],  // m=0
  [1.8412, 5.3314, 8.5363],   // m=1  ← lowest: χ'₁₁
  [3.0542, 6.7061, 9.9695],   // m=2
  [4.2012, 8.0152, 11.3459],  // m=3
  [5.3175, 9.2824, 12.6819],  // m=4
];

/** J₀(x) — series expansion */
function j0(x: number): number {
  let s = 1, term = 1;
  const x2 = x * x / 4;
  for (let k = 1; k <= 30; k++) {
    term *= -x2 / (k * k);
    s += term;
    if (Math.abs(term) < 1e-15 * Math.abs(s)) break;
  }
  return s;
}

/** J₁(x) — series expansion */
function j1(x: number): number {
  if (x === 0) return 0;
  let s = 0.5 * x, term = 0.5 * x;
  const x2 = x * x / 4;
  for (let k = 1; k <= 30; k++) {
    term *= -x2 / (k * (k + 1));
    s += term;
    if (Math.abs(term) < 1e-15 * Math.abs(s)) break;
  }
  return s;
}

/** Jₘ(x) — forward recurrence for m > 1 */
export function jm(m: number, x: number): number {
  if (m === 0) return j0(x);
  if (m === 1) return j1(x);
  if (Math.abs(x) < 1e-30) return 0;
  let prev = j0(x), curr = j1(x);
  for (let k = 1; k < m; k++) {
    const next = (2 * k / x) * curr - prev;
    prev = curr; curr = next;
  }
  return curr;
}

/** J_m'(x) = (J_{m-1}(x) − J_{m+1}(x)) / 2 */
export function djm(m: number, x: number): number {
  if (m === 0) return -j1(x);
  return 0.5 * (jm(m - 1, x) - jm(m + 1, x));
}

// ── EMField at a point ────────────────────────────────────────────────────────

export interface EMField {
  ex: number; ey: number; ez: number;
  hx: number; hy: number; hz: number;
}

export const eMag = (f: EMField) => Math.sqrt(f.ex**2 + f.ey**2 + f.ez**2);
export const hMag = (f: EMField) => Math.sqrt(f.hx**2 + f.hy**2 + f.hz**2);

const ZERO_FIELD: EMField = { ex:0, ey:0, ez:0, hx:0, hy:0, hz:0 };

// ── Rectangular Cavity ────────────────────────────────────────────────────────

export type ModeType = 'TE' | 'TM';

export interface RectCavityMode {
  type: 'rect-cavity';
  a: number; b: number; d: number;
  m: number; n: number; p: number;
  modeType: ModeType;
  kx: number; ky: number; kz: number;
  kc2: number; omega: number;
}

export function makeRectCavityMode(
  a: number, b: number, d: number,
  m: number, n: number, p: number,
  modeType: ModeType = 'TE'
): RectCavityMode {
  if (modeType === 'TE' && m === 0 && n === 0) throw new Error('TE: m and n cannot both be 0');
  if (modeType === 'TE' && p < 1) throw new Error('TE: p must be >= 1');
  if (modeType === 'TM' && (m < 1 || n < 1)) throw new Error('TM: m and n must be >= 1');

  const kx = m * Math.PI / a;
  const ky = n * Math.PI / b;
  const kz = p * Math.PI / d;
  const kc2 = kx**2 + ky**2;
  const omega = C_LIGHT * Math.sqrt(kx**2 + ky**2 + kz**2);
  return { type: 'rect-cavity', a, b, d, m, n, p, modeType, kx, ky, kz, kc2, omega };
}

export function rectCavityFreq(mo: RectCavityMode): number {
  return mo.omega / (2 * Math.PI);
}

export function rectCavityFields(
  mo: RectCavityMode,
  x: number, y: number, z: number,
  phase: number
): EMField {
  const { kx, ky, kz, kc2, omega } = mo;
  const ct = Math.cos(phase), st = Math.sin(phase);
  const cx = Math.cos(kx*x), sx = Math.sin(kx*x);
  const cy = Math.cos(ky*y), sy = Math.sin(ky*y);
  const cz = Math.cos(kz*z), sz = Math.sin(kz*z);

  if (mo.modeType === 'TE') {
    const hz = cx * cy * sz * st;
    if (kc2 < 1e-30) return { ...ZERO_FIELD, hz };
    return {
      ex:  (omega * MU0 * ky / kc2) * cx * sy * sz * ct,
      ey: -(omega * MU0 * kx / kc2) * sx * cy * sz * ct,
      ez: 0,
      hx: -(kx * kz / kc2) * sx * cy * cz * st,
      hy: -(ky * kz / kc2) * cx * sy * cz * st,
      hz,
    };
  } else { // TM
    const ez = sx * sy * cz * ct;
    if (kc2 < 1e-30) return { ...ZERO_FIELD, ez };
    return {
      ex:  (kx * kz / kc2) * cx * sy * sz * ct,
      ey:  (ky * kz / kc2) * sx * cy * sz * ct,
      ez,
      hx:  (omega * EPS0 * ky / kc2) * sx * cy * cz * st,
      hy: -(omega * EPS0 * kx / kc2) * cx * sy * cz * st,
      hz: 0,
    };
  }
}

export function rectCavityLabel(mo: RectCavityMode): string {
  return `${mo.modeType}_${mo.m}${mo.n}${mo.p}`;
}

// ── Rectangular Waveguide ─────────────────────────────────────────────────────

export interface RectWaveguideMode {
  type: 'rect-wg';
  a: number; b: number;
  m: number; n: number;
  modeType: ModeType;
  kx: number; ky: number; kc2: number; fc: number;
}

export function makeRectWaveguideMode(
  a: number, b: number,
  m: number, n: number,
  modeType: ModeType = 'TE'
): RectWaveguideMode {
  if (modeType === 'TE' && m === 0 && n === 0) throw new Error('TE: m and n cannot both be 0');
  if (modeType === 'TM' && (m < 1 || n < 1)) throw new Error('TM: m and n must be >= 1');
  const kx = m * Math.PI / a;
  const ky = n * Math.PI / b;
  const kc2 = kx**2 + ky**2;
  const fc = C_LIGHT * Math.sqrt(kc2) / (2 * Math.PI);
  return { type: 'rect-wg', a, b, m, n, modeType, kx, ky, kc2, fc };
}

export function rectWgBeta(mo: RectWaveguideMode, freq: number): number {
  const k = 2 * Math.PI * freq / C_LIGHT;
  const d = k**2 - mo.kc2;
  return d >= 0 ? Math.sqrt(d) : 0;
}

export function rectWgFields(
  mo: RectWaveguideMode,
  x: number, y: number, z: number,
  freq: number, phase: number
): EMField {
  const beta  = rectWgBeta(mo, freq);
  const omega = 2 * Math.PI * freq;
  const psi   = Math.cos(phase - beta * z);
  const { kx, ky, kc2 } = mo;
  if (kc2 < 1e-30) return ZERO_FIELD;

  const cx = Math.cos(kx*x), sx = Math.sin(kx*x);
  const cy = Math.cos(ky*y), sy = Math.sin(ky*y);

  if (mo.modeType === 'TE') {
    return {
      ex: -(omega * MU0 * ky / kc2) * cx * sy * psi,
      ey:  (omega * MU0 * kx / kc2) * sx * cy * psi,
      ez: 0,
      hx:  (beta * kx / kc2) * sx * cy * psi,
      hy:  (beta * ky / kc2) * cx * sy * psi,
      hz:  cx * cy * psi,
    };
  } else { // TM
    return {
      ex: -(beta * kx / kc2) * cx * sy * psi,
      ey: -(beta * ky / kc2) * sx * cy * psi,
      ez:  sx * sy * psi,
      hx:  (omega * EPS0 * ky / kc2) * sx * cy * psi,
      hy: -(omega * EPS0 * kx / kc2) * cx * sy * psi,
      hz: 0,
    };
  }
}

export function rectWgLabel(mo: RectWaveguideMode): string {
  return `${mo.modeType}_${mo.m}${mo.n}`;
}

// ── Circular Waveguide ────────────────────────────────────────────────────────

export interface CircWaveguideMode {
  type: 'circ-wg';
  r: number;
  m: number; n: number;
  modeType: ModeType;
  chi: number; kc: number; kc2: number; fc: number;
}

export function makeCircWaveguideMode(
  r: number, m: number, n: number,
  modeType: ModeType = 'TE'
): CircWaveguideMode {
  if (n < 1) throw new Error('n must be >= 1');
  if (m > 4 || n > 3) throw new Error('m/n out of table range');
  const chi = modeType === 'TM' ? TM_ZEROS[m][n-1] : TE_ZEROS[m][n-1];
  const kc  = chi / r;
  const fc  = C_LIGHT * kc / (2 * Math.PI);
  return { type: 'circ-wg', r, m, n, modeType, chi, kc, kc2: kc**2, fc };
}

export function circWgBeta(mo: CircWaveguideMode, freq: number): number {
  const k = 2 * Math.PI * freq / C_LIGHT;
  const d = k**2 - mo.kc2;
  return d >= 0 ? Math.sqrt(d) : 0;
}

export function circWgFields(
  mo: CircWaveguideMode,
  rho: number, phi: number, z: number,
  freq: number, phase: number
): EMField {
  const beta  = circWgBeta(mo, freq);
  const omega = 2 * Math.PI * freq;
  const psi   = Math.cos(phase - beta * z);
  const { kc, kc2, m } = mo;
  const kcRho = kc * rho;
  const jmv   = jm(m, kcRho);
  const djmv  = djm(m, kcRho) * kc;
  const cmphi = Math.cos(m * phi), smphi = Math.sin(m * phi);
  const rhoS  = Math.max(rho, 1e-30);

  if (mo.modeType === 'TM') {
    return {
      ex: -(beta * kc / kc2) * djmv * cmphi * psi,
      ey:  (beta * m / (kc2 * rhoS)) * jmv * smphi * psi,
      ez:  jmv * cmphi * psi,
      hx: -(omega * EPS0 * m / (kc2 * rhoS)) * jmv * smphi * psi,
      hy:  (omega * EPS0 * kc / kc2) * djmv * cmphi * psi,
      hz: 0,
    };
  } else { // TE
    return {
      ex:  (omega * MU0 * m / (kc2 * rhoS)) * jmv * smphi * psi,
      ey: -(omega * MU0 * kc / kc2) * djmv * cmphi * psi,
      ez: 0,
      hx: -(beta * kc / kc2) * djmv * cmphi * psi,
      hy:  (beta * m / (kc2 * rhoS)) * jmv * smphi * psi,
      hz:  jmv * cmphi * psi,
    };
  }
}

export function circWgLabel(mo: CircWaveguideMode): string {
  return `${mo.modeType}_${mo.m}${mo.n}`;
}

// ── Dispersion data helpers ───────────────────────────────────────────────────

export function rectWgDispersion(
  mo: RectWaveguideMode,
  fMin: number, fMax: number, nPts = 200
): { freqs: number[]; betas: number[] } {
  const freqs: number[] = [], betas: number[] = [];
  for (let i = 0; i < nPts; i++) {
    const f = fMin + (fMax - fMin) * i / (nPts - 1);
    freqs.push(f);
    betas.push(rectWgBeta(mo, f));
  }
  return { freqs, betas };
}
