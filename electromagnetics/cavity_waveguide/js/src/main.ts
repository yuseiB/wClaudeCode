/**
 * main.ts — Interactive EM Cavity & Waveguide Field Visualizer
 *
 * Features:
 *   - Real-time field animation (phase ωt evolves)
 *   - Mode selection: cavity (rectangular/cylindrical) + waveguide (rectangular/circular)
 *   - Visualisation: E-field heatmap + H-field quiver arrows
 *   - Dispersion chart (ω vs β for waveguide modes)
 *   - Live frequency and β readout
 */

import {
  C_LIGHT, MU0, EPS0,
  makeRectCavityMode, rectCavityFields, rectCavityFreq, rectCavityLabel,
  makeRectWaveguideMode, rectWgFields, rectWgBeta, rectWgLabel,
  makeCircWaveguideMode, circWgFields, circWgBeta, circWgLabel,
  eMag, hMag,
  rectWgDispersion,
  type EMField, type ModeType,
} from './em_physics';

// ── Canvas setup ─────────────────────────────────────────────────────────────

const CANVAS_W = 400, CANVAS_H = 400;
const DISP_W   = 400, DISP_H   = 250;

// ── Colormap (RdBu: -1→blue, 0→white, +1→red) ────────────────────────────────

function colormap(v: number): [number, number, number] {
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    const r = 200 + Math.round(55 * t);
    const g = Math.round(200 * (1 - t));
    const b = Math.round(200 * (1 - t));
    return [r, g, b];
  } else {
    const s = -t;
    const r = Math.round(200 * (1 - s));
    const g = Math.round(200 * (1 - s));
    const b = 200 + Math.round(55 * s);
    return [r, g, b];
  }
}

function heatColormap(v: number): [number, number, number] {
  // hot_r: 0→white, 1→black (inverted hot)
  const t = Math.max(0, Math.min(1, v));
  if (t < 0.33) {
    const s = t / 0.33;
    return [255, Math.round(255 * (1 - s)), Math.round(255 * (1 - s))];
  } else if (t < 0.67) {
    const s = (t - 0.33) / 0.34;
    return [255, Math.round(s * 100), 0];
  } else {
    const s = (t - 0.67) / 0.33;
    return [Math.round(255 * (1 - s * 0.5)), 0, 0];
  }
}

// ── App state ─────────────────────────────────────────────────────────────────

type ViewMode = 'rect-cavity' | 'circ-wg' | 'rect-wg';

interface AppState {
  viewMode: ViewMode;
  phase: number;
  playing: boolean;
  speed: number;            // phase increment per frame
  // Rectangular cavity
  rcA: number; rcB: number; rcD: number;
  rcM: number; rcN: number; rcP: number;
  rcModeType: ModeType;
  // Rectangular waveguide
  rwA: number; rwB: number;
  rwM: number; rwN: number;
  rwModeType: ModeType;
  rwFreqMult: number;        // f / fc  (operating frequency factor)
  // Circular waveguide
  cwR: number;
  cwM: number; cwN: number;
  cwModeType: ModeType;
  cwFreqMult: number;
}

const state: AppState = {
  viewMode: 'rect-cavity',
  phase: 0,
  playing: true,
  speed: 0.08,
  rcA: 0.04, rcB: 0.02, rcD: 0.03,
  rcM: 1, rcN: 0, rcP: 1,
  rcModeType: 'TE',
  rwA: 0.04, rwB: 0.02,
  rwM: 1, rwN: 0,
  rwModeType: 'TE',
  rwFreqMult: 1.5,
  cwR: 0.015,
  cwM: 1, cwN: 1,
  cwModeType: 'TE',
  cwFreqMult: 1.5,
};

// ── DOM elements ──────────────────────────────────────────────────────────────

const fieldCanvas  = document.getElementById('fieldCanvas')  as HTMLCanvasElement;
const dispCanvas   = document.getElementById('dispCanvas')   as HTMLCanvasElement;
const infoDiv      = document.getElementById('info')         as HTMLDivElement;
const viewSelect   = document.getElementById('viewMode')     as HTMLSelectElement;
const playBtn      = document.getElementById('playBtn')      as HTMLButtonElement;
const speedSlider  = document.getElementById('speedSlider')  as HTMLInputElement;
const modeTypeBtn  = document.getElementById('modeTypeBtn')  as HTMLButtonElement;
const mPanel       = document.getElementById('mPanel')       as HTMLDivElement;
const nPanel       = document.getElementById('nPanel')       as HTMLDivElement;
const pPanel       = document.getElementById('pPanel')       as HTMLDivElement;
const freqPanel    = document.getElementById('freqPanel')    as HTMLDivElement;

const fCtx = fieldCanvas.getContext('2d')!;
const dCtx = dispCanvas.getContext('2d')!;

fieldCanvas.width  = CANVAS_W;
fieldCanvas.height = CANVAS_H;
dispCanvas.width   = DISP_W;
dispCanvas.height  = DISP_H;

// ── Render field canvas ───────────────────────────────────────────────────────

const N = 80;   // grid resolution

function renderRectCavity(): void {
  let mode;
  try {
    mode = makeRectCavityMode(
      state.rcA, state.rcB, state.rcD,
      state.rcM, state.rcN, state.rcP,
      state.rcModeType
    );
  } catch {
    drawError(fCtx, 'Invalid mode indices');
    return;
  }

  const { a, d } = mode;
  const dxCell = CANVAS_W / N, dzCell = CANVAS_H / N;

  // Build grid: x=col, z=row, y=b/2
  const eGrid: number[][] = [];
  let eMax = 1e-30;
  for (let row = 0; row < N; row++) {
    eGrid.push([]);
    for (let col = 0; col < N; col++) {
      const x = (col + 0.5) * a / N;
      const z = (row + 0.5) * d / N;
      const f = rectCavityFields(mode, x, state.rcB / 2, z, state.phase);
      const e = eMag(f);
      eGrid[row].push(e);
      if (e > eMax) eMax = e;
    }
  }

  const imgData = fCtx.createImageData(CANVAS_W, CANVAS_H);
  for (let row = 0; row < N; row++) {
    for (let col = 0; col < N; col++) {
      const t = eGrid[row][col] / eMax;
      const [r, g, b] = heatColormap(t);
      for (let dr = 0; dr < Math.ceil(dzCell); dr++) {
        for (let dc = 0; dc < Math.ceil(dxCell); dc++) {
          const px = Math.floor(row * dzCell) + dr;
          const py = Math.floor(col * dxCell) + dc;
          if (px >= CANVAS_H || py >= CANVAS_W) continue;
          const idx = (px * CANVAS_W + py) * 4;
          imgData.data[idx]   = r;
          imgData.data[idx+1] = g;
          imgData.data[idx+2] = b;
          imgData.data[idx+3] = 255;
        }
      }
    }
  }
  fCtx.putImageData(imgData, 0, 0);

  // Quiver: E-field arrows (xz plane)
  const STEP = 8;
  fCtx.strokeStyle = 'rgba(255,255,0,0.8)';
  fCtx.lineWidth = 1.5;
  for (let row = STEP/2; row < N; row += STEP) {
    for (let col = STEP/2; col < N; col += STEP) {
      const x = (col + 0.5) * a / N;
      const z = (row + 0.5) * d / N;
      const f = rectCavityFields(mode, x, state.rcB / 2, z, state.phase);
      const mag = Math.sqrt(f.ex**2 + f.ez**2);
      if (mag < 0.01 * eMax) continue;
      const scale = (STEP * dxCell * 0.45) / eMax;
      const cx2 = (col + 0.5) * dxCell;
      const cy2 = (row + 0.5) * dzCell;
      drawArrow(fCtx, cx2, cy2, f.ex * scale, f.ez * scale);
    }
  }

  // Label
  fCtx.fillStyle = 'white';
  fCtx.font = 'bold 13px monospace';
  fCtx.fillText(
    `${rectCavityLabel(mode)}  f=${(rectCavityFreq(mode)/1e9).toFixed(3)} GHz`,
    8, 20
  );
  fCtx.font = '11px monospace';
  fCtx.fillText(`|E| — xz cross-section (y=b/2)   ωt=${(state.phase % (2*Math.PI)).toFixed(2)}`, 8, 36);

  updateInfoCavity(mode);
}

function renderRectWaveguide(): void {
  let mode;
  try {
    mode = makeRectWaveguideMode(state.rwA, state.rwB, state.rwM, state.rwN, state.rwModeType);
  } catch {
    drawError(fCtx, 'Invalid mode indices');
    return;
  }

  const freq = mode.fc * state.rwFreqMult;
  const beta = rectWgBeta(mode, freq);
  const { a, b } = mode;
  const dxCell = CANVAS_W / N, dyCell = CANVAS_H / N;

  const eGrid: number[][] = [];
  let eMax = 1e-30;
  for (let row = 0; row < N; row++) {
    eGrid.push([]);
    for (let col = 0; col < N; col++) {
      const x = (col + 0.5) * a / N;
      const y = (row + 0.5) * b / N;
      const f = rectWgFields(mode, x, y, 0, freq, state.phase);
      const e = eMag(f);
      eGrid[row].push(e);
      if (e > eMax) eMax = e;
    }
  }

  const imgData = fCtx.createImageData(CANVAS_W, CANVAS_H);
  for (let row = 0; row < N; row++) {
    for (let col = 0; col < N; col++) {
      const t = eGrid[row][col] / eMax;
      const [r, g, b2] = heatColormap(t);
      for (let dr = 0; dr < Math.ceil(dyCell); dr++) {
        for (let dc = 0; dc < Math.ceil(dxCell); dc++) {
          const px = Math.floor(row * dyCell) + dr;
          const py = Math.floor(col * dxCell) + dc;
          if (px >= CANVAS_H || py >= CANVAS_W) continue;
          const idx = (px * CANVAS_W + py) * 4;
          imgData.data[idx]   = r;
          imgData.data[idx+1] = g;
          imgData.data[idx+2] = b2;
          imgData.data[idx+3] = 255;
        }
      }
    }
  }
  fCtx.putImageData(imgData, 0, 0);

  // Quiver: E-field
  const STEP = 8;
  fCtx.strokeStyle = 'rgba(0,255,255,0.8)';
  fCtx.lineWidth = 1.5;
  for (let row = STEP/2; row < N; row += STEP) {
    for (let col = STEP/2; col < N; col += STEP) {
      const x = (col + 0.5) * a / N;
      const y = (row + 0.5) * b / N;
      const f = rectWgFields(mode, x, y, 0, freq, state.phase);
      const mag = Math.sqrt(f.ex**2 + f.ey**2);
      if (mag < 0.02 * eMax) continue;
      const scale = (STEP * dxCell * 0.45) / eMax;
      const cx2 = (col + 0.5) * dxCell;
      const cy2 = (row + 0.5) * dyCell;
      drawArrow(fCtx, cx2, cy2, f.ex * scale, f.ey * scale);
    }
  }

  fCtx.fillStyle = 'white';
  fCtx.font = 'bold 13px monospace';
  fCtx.fillText(`${rectWgLabel(mode)}  fc=${(mode.fc/1e9).toFixed(3)} GHz`, 8, 20);
  fCtx.font = '11px monospace';
  fCtx.fillText(
    `f=${(freq/1e9).toFixed(3)} GHz  β=${beta.toFixed(1)} rad/m  ωt=${(state.phase%(2*Math.PI)).toFixed(2)}`,
    8, 36
  );

  drawDispersionChart(mode, freq);
  updateInfoWg(mode, freq, beta);
}

function renderCircWaveguide(): void {
  let mode;
  try {
    mode = makeCircWaveguideMode(state.cwR, state.cwM, state.cwN, state.cwModeType);
  } catch {
    drawError(fCtx, 'Invalid mode indices');
    return;
  }

  const freq  = mode.fc * state.cwFreqMult;
  const beta  = circWgBeta(mode, freq);
  const R = mode.r;
  const dxCell = CANVAS_W / N, dyCell = CANVAS_H / N;

  const eGrid: number[][] = [];
  let eMax = 1e-30;
  for (let row = 0; row < N; row++) {
    eGrid.push([]);
    for (let col = 0; col < N; col++) {
      const x = (col / N - 0.5) * 2 * R;
      const y = (row / N - 0.5) * 2 * R;
      const rho = Math.sqrt(x**2 + y**2);
      const phi = Math.atan2(y, x);
      if (rho > R) { eGrid[row].push(NaN); continue; }
      const f = circWgFields(mode, rho, phi, 0, freq, state.phase);
      const e = eMag(f);
      eGrid[row].push(e);
      if (e > eMax) eMax = e;
    }
  }

  const imgData = fCtx.createImageData(CANVAS_W, CANVAS_H);
  for (let row = 0; row < N; row++) {
    for (let col = 0; col < N; col++) {
      const t = eGrid[row][col];
      let r2 = 30, g2 = 30, b2 = 30;
      if (!isNaN(t)) {
        [r2, g2, b2] = heatColormap(t / eMax);
      }
      for (let dr = 0; dr < Math.ceil(dyCell); dr++) {
        for (let dc = 0; dc < Math.ceil(dxCell); dc++) {
          const px = Math.floor(row * dyCell) + dr;
          const py = Math.floor(col * dxCell) + dc;
          if (px >= CANVAS_H || py >= CANVAS_W) continue;
          const idx = (px * CANVAS_W + py) * 4;
          imgData.data[idx]   = r2;
          imgData.data[idx+1] = g2;
          imgData.data[idx+2] = b2;
          imgData.data[idx+3] = 255;
        }
      }
    }
  }
  fCtx.putImageData(imgData, 0, 0);

  // Draw circle boundary
  fCtx.strokeStyle = 'white';
  fCtx.lineWidth = 2;
  fCtx.beginPath();
  fCtx.arc(CANVAS_W/2, CANVAS_H/2, R / (2*R) * CANVAS_W, 0, 2*Math.PI);
  fCtx.stroke();

  fCtx.fillStyle = 'white';
  fCtx.font = 'bold 13px monospace';
  fCtx.fillText(`${circWgLabel(mode)}  fc=${(mode.fc/1e9).toFixed(3)} GHz`, 8, 20);
  fCtx.font = '11px monospace';
  fCtx.fillText(
    `f=${(freq/1e9).toFixed(3)} GHz  β=${beta.toFixed(1)} rad/m  ωt=${(state.phase%(2*Math.PI)).toFixed(2)}`,
    8, 36
  );

  drawDispersionChartCirc(mode, freq);
  updateInfoCircWg(mode, freq, beta);
}

// ── Arrow drawing ─────────────────────────────────────────────────────────────

function drawArrow(ctx: CanvasRenderingContext2D, x: number, y: number, dx: number, dy: number): void {
  const len = Math.sqrt(dx**2 + dy**2);
  if (len < 0.5) return;
  ctx.beginPath();
  ctx.moveTo(x - dx/2, y - dy/2);
  ctx.lineTo(x + dx/2, y + dy/2);
  // arrowhead
  const ang = Math.atan2(dy, dx);
  const aLen = Math.min(len * 0.4, 5);
  ctx.moveTo(x + dx/2, y + dy/2);
  ctx.lineTo(x + dx/2 - aLen * Math.cos(ang - 0.5), y + dy/2 - aLen * Math.sin(ang - 0.5));
  ctx.moveTo(x + dx/2, y + dy/2);
  ctx.lineTo(x + dx/2 - aLen * Math.cos(ang + 0.5), y + dy/2 - aLen * Math.sin(ang + 0.5));
  ctx.stroke();
}

function drawError(ctx: CanvasRenderingContext2D, msg: string): void {
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
  ctx.fillStyle = '#ff6b6b';
  ctx.font = '14px monospace';
  ctx.fillText(msg, 20, CANVAS_H / 2);
}

// ── Dispersion chart ──────────────────────────────────────────────────────────

function drawDispersionChart(mode: ReturnType<typeof makeRectWaveguideMode>, fOp: number): void {
  dCtx.fillStyle = '#0d1117';
  dCtx.fillRect(0, 0, DISP_W, DISP_H);

  const fc = mode.fc;
  const fMax = fc * 3;
  const { freqs, betas } = rectWgDispersion(mode, 0, fMax);
  const betaMax = betas[betas.length - 1] * 1.05;

  const mx = 45, my = 15, mw = DISP_W - mx - 20, mh = DISP_H - my - 35;
  const scX = (f: number) => mx + (f / fMax) * mw;
  const scY = (b: number) => my + mh - (b / betaMax) * mh;

  // Grid
  dCtx.strokeStyle = '#333';
  dCtx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {
    const y = my + i * mh / 5;
    dCtx.beginPath(); dCtx.moveTo(mx, y); dCtx.lineTo(mx + mw, y); dCtx.stroke();
    dCtx.fillStyle = '#aaa'; dCtx.font = '10px monospace';
    dCtx.fillText((betaMax * (5 - i) / 5).toFixed(0), 2, y + 4);
  }

  // β curve
  dCtx.strokeStyle = '#4fc3f7';
  dCtx.lineWidth = 2;
  dCtx.beginPath();
  freqs.forEach((f, i) => {
    const px = scX(f), py = scY(betas[i]);
    i === 0 ? dCtx.moveTo(px, py) : dCtx.lineTo(px, py);
  });
  dCtx.stroke();

  // Light line ω=βc
  dCtx.strokeStyle = '#888';
  dCtx.setLineDash([4, 4]);
  dCtx.lineWidth = 1;
  dCtx.beginPath();
  dCtx.moveTo(scX(0), scY(0));
  dCtx.lineTo(scX(fMax), scY(fMax / C_LIGHT));
  dCtx.stroke();
  dCtx.setLineDash([]);

  // Cutoff line
  dCtx.strokeStyle = '#ff9800';
  dCtx.lineWidth = 1.5;
  dCtx.setLineDash([4, 2]);
  const xfc = scX(fc);
  dCtx.beginPath(); dCtx.moveTo(xfc, my); dCtx.lineTo(xfc, my + mh); dCtx.stroke();
  dCtx.setLineDash([]);

  // Operating point
  const betaOp = rectWgBeta(mode, fOp);
  dCtx.fillStyle = '#ff5252';
  dCtx.beginPath();
  dCtx.arc(scX(fOp), scY(betaOp), 5, 0, 2 * Math.PI);
  dCtx.fill();

  // Axes labels
  dCtx.fillStyle = '#ccc'; dCtx.font = '11px monospace';
  dCtx.fillText(`fc=${(fc/1e9).toFixed(2)}G`, xfc - 28, my + mh + 14);
  dCtx.fillText('f →', mx + mw - 20, my + mh + 28);
  dCtx.fillText('β', 2, my + 6);

  // Title
  dCtx.fillStyle = '#4fc3f7'; dCtx.font = 'bold 11px monospace';
  dCtx.fillText(`Dispersion: ${rectWgLabel(mode)}  (ω vs β)`, mx, my - 2);
}

function drawDispersionChartCirc(mode: ReturnType<typeof makeCircWaveguideMode>, fOp: number): void {
  dCtx.fillStyle = '#0d1117';
  dCtx.fillRect(0, 0, DISP_W, DISP_H);

  const fc = mode.fc;
  const fMax = fc * 3;
  const nPts = 200;
  const freqs: number[] = [], betas: number[] = [];
  for (let i = 0; i < nPts; i++) {
    const f = i * fMax / (nPts - 1);
    freqs.push(f);
    betas.push(circWgBeta(mode, f));
  }
  const betaMax = Math.max(betas[betas.length - 1] * 1.05, 1);

  const mx = 45, my = 15, mw = DISP_W - mx - 20, mh = DISP_H - my - 35;
  const scX = (f: number) => mx + (f / fMax) * mw;
  const scY = (b: number) => my + mh - (b / betaMax) * mh;

  dCtx.strokeStyle = '#333'; dCtx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {
    const y = my + i * mh / 5;
    dCtx.beginPath(); dCtx.moveTo(mx, y); dCtx.lineTo(mx + mw, y); dCtx.stroke();
    dCtx.fillStyle = '#aaa'; dCtx.font = '10px monospace';
    dCtx.fillText((betaMax * (5 - i) / 5).toFixed(0), 2, y + 4);
  }

  dCtx.strokeStyle = '#80cbc4'; dCtx.lineWidth = 2;
  dCtx.beginPath();
  freqs.forEach((f, i) => {
    const px = scX(f), py = scY(betas[i]);
    i === 0 ? dCtx.moveTo(px, py) : dCtx.lineTo(px, py);
  });
  dCtx.stroke();

  dCtx.strokeStyle = '#888'; dCtx.setLineDash([4,4]); dCtx.lineWidth = 1;
  dCtx.beginPath();
  dCtx.moveTo(scX(0), scY(0));
  dCtx.lineTo(scX(fMax), scY(fMax / C_LIGHT));
  dCtx.stroke(); dCtx.setLineDash([]);

  dCtx.strokeStyle = '#ff9800'; dCtx.lineWidth = 1.5; dCtx.setLineDash([4,2]);
  const xfc = scX(fc);
  dCtx.beginPath(); dCtx.moveTo(xfc, my); dCtx.lineTo(xfc, my+mh); dCtx.stroke();
  dCtx.setLineDash([]);

  const betaOp = circWgBeta(mode, fOp);
  dCtx.fillStyle = '#ff5252';
  dCtx.beginPath(); dCtx.arc(scX(fOp), scY(betaOp), 5, 0, 2*Math.PI); dCtx.fill();

  dCtx.fillStyle = '#ccc'; dCtx.font = '11px monospace';
  dCtx.fillText(`fc=${(fc/1e9).toFixed(2)}G`, xfc-28, my+mh+14);
  dCtx.fillStyle = '#80cbc4'; dCtx.font = 'bold 11px monospace';
  dCtx.fillText(`Dispersion: ${circWgLabel(mode)}  (ω vs β)`, mx, my-2);
}

// ── Info panel ────────────────────────────────────────────────────────────────

function updateInfoCavity(mode: ReturnType<typeof makeRectCavityMode>): void {
  const f = rectCavityFreq(mode);
  infoDiv.innerHTML = `
    <b>${rectCavityLabel(mode)}</b>  f<sub>res</sub> = ${(f/1e9).toFixed(4)} GHz<br>
    a=${(mode.a*100).toFixed(0)} cm, b=${(mode.b*100).toFixed(0)} cm, d=${(mode.d*100).toFixed(0)} cm<br>
    λ = ${(C_LIGHT/f*100).toFixed(2)} cm<br>
    <small>E~cos(ωt) (hot=max),  H~sin(ωt)  [90° phase shift]</small>
  `;
}

function updateInfoWg(mode: ReturnType<typeof makeRectWaveguideMode>, freq: number, beta: number): void {
  const lam = beta > 0 ? (2*Math.PI/beta*100) : Infinity;
  const vg  = beta > 0 ? ((2*Math.PI*freq/beta) / C_LIGHT * 100) : 0;
  infoDiv.innerHTML = `
    <b>${rectWgLabel(mode)}</b>  f<sub>c</sub> = ${(mode.fc/1e9).toFixed(4)} GHz<br>
    f<sub>op</sub> = ${(freq/1e9).toFixed(3)} GHz  (${state.rwFreqMult.toFixed(1)}×f<sub>c</sub>)<br>
    β = ${beta.toFixed(1)} rad/m   λ<sub>g</sub> = ${isFinite(lam)?lam.toFixed(2):'∞'} cm<br>
    v<sub>ph</sub>/c = ${beta>0?(freq/(beta*C_LIGHT/(2*Math.PI)*C_LIGHT)).toFixed(2):'—'}
    <small style="float:right">▶ yellow arrows = E-field</small>
  `;
}

function updateInfoCircWg(mode: ReturnType<typeof makeCircWaveguideMode>, freq: number, beta: number): void {
  const lam = beta > 0 ? (2*Math.PI/beta*100) : Infinity;
  infoDiv.innerHTML = `
    <b>${circWgLabel(mode)}</b>  f<sub>c</sub> = ${(mode.fc/1e9).toFixed(4)} GHz<br>
    R = ${(mode.r*100).toFixed(1)} cm  χ = ${mode.chi.toFixed(4)}<br>
    f<sub>op</sub> = ${(freq/1e9).toFixed(3)} GHz  β = ${beta.toFixed(1)} rad/m<br>
    λ<sub>g</sub> = ${isFinite(lam)?lam.toFixed(2):'∞'} cm
    <small style="float:right">▶ cyan = E in (ρ,φ) plane</small>
  `;
}

// ── Render loop ───────────────────────────────────────────────────────────────

function render(): void {
  fCtx.fillStyle = '#0d1117';
  fCtx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  switch (state.viewMode) {
    case 'rect-cavity': renderRectCavity(); break;
    case 'rect-wg':     renderRectWaveguide(); break;
    case 'circ-wg':     renderCircWaveguide(); break;
  }

  if (state.playing) {
    state.phase += state.speed;
  }
  requestAnimationFrame(render);
}

// ── UI event wiring ───────────────────────────────────────────────────────────

function updatePanelVisibility(): void {
  pPanel.style.display     = state.viewMode === 'rect-cavity' ? 'block' : 'none';
  freqPanel.style.display  = state.viewMode !== 'rect-cavity' ? 'block' : 'none';
  dispCanvas.style.display = state.viewMode !== 'rect-cavity' ? 'block' : 'none';
}

viewSelect.addEventListener('change', () => {
  state.viewMode = viewSelect.value as ViewMode;
  updatePanelVisibility();
});

playBtn.addEventListener('click', () => {
  state.playing = !state.playing;
  playBtn.textContent = state.playing ? '⏸ Pause' : '▶ Play';
});

speedSlider.addEventListener('input', () => {
  state.speed = parseFloat(speedSlider.value);
});

modeTypeBtn.addEventListener('click', () => {
  if (state.viewMode === 'rect-cavity') {
    state.rcModeType = state.rcModeType === 'TE' ? 'TM' : 'TE';
    modeTypeBtn.textContent = `Mode: ${state.rcModeType}`;
    if (state.rcModeType === 'TM') {
      if (state.rcM < 1) state.rcM = 1;
      if (state.rcN < 1) state.rcN = 1;
    }
  } else if (state.viewMode === 'rect-wg') {
    state.rwModeType = state.rwModeType === 'TE' ? 'TM' : 'TE';
    modeTypeBtn.textContent = `Mode: ${state.rwModeType}`;
  } else {
    state.cwModeType = state.cwModeType === 'TE' ? 'TM' : 'TE';
    modeTypeBtn.textContent = `Mode: ${state.cwModeType}`;
  }
});

// Preset buttons
document.querySelectorAll<HTMLButtonElement>('[data-preset]').forEach(btn => {
  btn.addEventListener('click', () => {
    const preset = btn.dataset['preset']!;
    switch (preset) {
      case 'te101':
        state.viewMode = 'rect-cavity'; state.rcModeType = 'TE';
        state.rcM = 1; state.rcN = 0; state.rcP = 1;
        viewSelect.value = 'rect-cavity';
        modeTypeBtn.textContent = 'Mode: TE';
        break;
      case 'tm110':
        state.viewMode = 'rect-cavity'; state.rcModeType = 'TM';
        state.rcM = 1; state.rcN = 1; state.rcP = 0;
        viewSelect.value = 'rect-cavity';
        modeTypeBtn.textContent = 'Mode: TM';
        break;
      case 'te10':
        state.viewMode = 'rect-wg'; state.rwModeType = 'TE';
        state.rwM = 1; state.rwN = 0;
        viewSelect.value = 'rect-wg';
        modeTypeBtn.textContent = 'Mode: TE';
        break;
      case 'te11c':
        state.viewMode = 'circ-wg'; state.cwModeType = 'TE';
        state.cwM = 1; state.cwN = 1;
        viewSelect.value = 'circ-wg';
        modeTypeBtn.textContent = 'Mode: TE';
        break;
    }
    updatePanelVisibility();
  });
});

// m, n, p sliders / inputs
document.getElementById('mSlider')!.addEventListener('input', (e) => {
  const v = parseInt((e.target as HTMLInputElement).value);
  if (state.viewMode === 'rect-cavity') state.rcM = v;
  else if (state.viewMode === 'rect-wg') state.rwM = v;
  else state.cwM = v;
  document.getElementById('mVal')!.textContent = String(v);
});

document.getElementById('nSlider')!.addEventListener('input', (e) => {
  const v = parseInt((e.target as HTMLInputElement).value);
  if (state.viewMode === 'rect-cavity') state.rcN = v;
  else if (state.viewMode === 'rect-wg') state.rwN = v;
  else state.cwN = v;
  document.getElementById('nVal')!.textContent = String(v);
});

document.getElementById('pSlider')!.addEventListener('input', (e) => {
  const v = parseInt((e.target as HTMLInputElement).value);
  if (state.viewMode === 'rect-cavity') state.rcP = v;
  document.getElementById('pVal')!.textContent = String(v);
});

document.getElementById('freqSlider')!.addEventListener('input', (e) => {
  const v = parseFloat((e.target as HTMLInputElement).value);
  if (state.viewMode === 'rect-wg') state.rwFreqMult = v;
  else state.cwFreqMult = v;
  document.getElementById('freqVal')!.textContent = `${v.toFixed(1)}×fc`;
});

// Initial setup
updatePanelVisibility();
render();
