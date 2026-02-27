/**
 * 2D Ising Model — Interactive Browser App
 *
 * Features
 * --------
 * - Real-time N×N lattice rendering (red = +1, blue = −1)
 * - Temperature slider + preset buttons (Low T / Critical / High T)
 * - Play / Pause / Reset controls
 * - Speed multiplier (1 – 32 sweeps per frame)
 * - Live display of ⟨E⟩/N², ⟨|M|⟩/N², T, and MC sweeps
 */

import './style.css';
import { IsingModel2D, T_CRITICAL } from './ising_model';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const N = 64;    // lattice size
const CELL = 7;  // pixels per spin (canvas = N * CELL)

const PRESETS = [
  { label: 'Low T (ordered)',    T: 1.5,         init: 'up'     },
  { label: `Critical (T≈${T_CRITICAL.toFixed(2)})`, T: T_CRITICAL, init: 'random' },
  { label: 'High T (disordered)', T: 3.5,        init: 'random' },
] as const;

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------
document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
<h1>2D Ising Model — Statistical Physics</h1>
<p class="subtitle">H = −J Σ s<sub>i</sub>s<sub>j</sub> &nbsp;|&nbsp;
   Metropolis-Hastings MC &nbsp;|&nbsp;
   Onsager T<sub>c</sub> = ${T_CRITICAL.toFixed(4)}</p>

<div class="layout">

  <div class="canvas-wrap">
    <canvas id="canvas" width="${N * CELL}" height="${N * CELL}"></canvas>
    <div class="tc-line">Red = spin +1 &nbsp; Blue = spin −1</div>
  </div>

  <div class="controls">

    <!-- Stats -->
    <div class="panel">
      <div class="panel-title">Observables (running average)</div>
      <div class="stats-grid">
        <div class="stat">
          <span class="stat-label">⟨E⟩ / N²</span>
          <span class="stat-value" id="stat-e">—</span>
        </div>
        <div class="stat">
          <span class="stat-label">⟨|M|⟩ / N²</span>
          <span class="stat-value mag" id="stat-m">—</span>
        </div>
        <div class="stat">
          <span class="stat-label">Temperature T</span>
          <span class="stat-value temp" id="stat-t">—</span>
        </div>
        <div class="stat">
          <span class="stat-label">MC sweeps</span>
          <span class="stat-value" id="stat-sweeps" style="font-size:.85rem">0</span>
        </div>
      </div>
    </div>

    <!-- Temperature -->
    <div class="panel">
      <div class="panel-title">Temperature</div>
      <div class="slider-row">
        <label>T</label>
        <input id="slider-t" type="range" min="0.5" max="5.0" step="0.02" value="2.269">
        <span class="slider-val" id="val-t">2.269</span>
      </div>
      <div style="margin-top:.5rem">
        <span class="speed-label">T<sub>c</sub> = ${T_CRITICAL.toFixed(4)}</span>
      </div>
    </div>

    <!-- Presets -->
    <div class="panel">
      <div class="panel-title">Presets</div>
      <div class="btn-row">
        ${PRESETS.map((p, i) => `
          <button class="preset-btn" data-preset="${i}">${p.label}</button>
        `).join('')}
      </div>
    </div>

    <!-- Controls -->
    <div class="panel">
      <div class="panel-title">Simulation</div>
      <div class="btn-row" style="margin-bottom:.5rem">
        <button id="btn-play" class="primary">Pause</button>
        <button id="btn-reset">Reset</button>
        <button id="btn-up">All +1</button>
        <button id="btn-down" class="danger">All −1</button>
      </div>
      <div class="slider-row">
        <label style="min-width:4.5rem;font-size:.7rem">Sweeps/frame</label>
        <input id="slider-speed" type="range" min="1" max="32" step="1" value="4">
        <span class="slider-val" id="val-speed">×4</span>
      </div>
    </div>

  </div>
</div>
`;

// ---------------------------------------------------------------------------
// Canvas setup
// ---------------------------------------------------------------------------
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const imageData = ctx.createImageData(N * CELL, N * CELL);
const pixels = imageData.data;

function renderLattice(model: IsingModel2D): void {
  const lattice = model.lattice;
  const n = model.n;
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const spin = lattice[row * n + col];
      // +1 → red (220, 60, 60), −1 → blue (60, 100, 220)
      const r = spin > 0 ? 220 : 60;
      const g = spin > 0 ? 60  : 100;
      const b = spin > 0 ? 60  : 220;
      // Fill CELL×CELL block
      for (let dy = 0; dy < CELL; dy++) {
        for (let dx = 0; dx < CELL; dx++) {
          const px = ((row * CELL + dy) * (n * CELL) + (col * CELL + dx)) * 4;
          pixels[px]     = r;
          pixels[px + 1] = g;
          pixels[px + 2] = b;
          pixels[px + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

// ---------------------------------------------------------------------------
// Model + state
// ---------------------------------------------------------------------------
let model = new IsingModel2D(N, 1.0, 42);
model.setTemperature(T_CRITICAL);

let running = true;
let sweepsPerFrame = 4;

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------
const sliderT     = document.getElementById('slider-t')     as HTMLInputElement;
const valT        = document.getElementById('val-t')        as HTMLSpanElement;
const sliderSpeed = document.getElementById('slider-speed') as HTMLInputElement;
const valSpeed    = document.getElementById('val-speed')    as HTMLSpanElement;
const btnPlay     = document.getElementById('btn-play')     as HTMLButtonElement;
const btnReset    = document.getElementById('btn-reset')    as HTMLButtonElement;
const btnUp       = document.getElementById('btn-up')       as HTMLButtonElement;
const btnDown     = document.getElementById('btn-down')     as HTMLButtonElement;
const statE       = document.getElementById('stat-e')       as HTMLSpanElement;
const statM       = document.getElementById('stat-m')       as HTMLSpanElement;
const statTel     = document.getElementById('stat-t')       as HTMLSpanElement;
const statSweeps  = document.getElementById('stat-sweeps')  as HTMLSpanElement;

sliderT.addEventListener('input', () => {
  const t = parseFloat(sliderT.value);
  valT.textContent = t.toFixed(3);
  model.setTemperature(t);
});

sliderSpeed.addEventListener('input', () => {
  sweepsPerFrame = parseInt(sliderSpeed.value);
  valSpeed.textContent = `×${sweepsPerFrame}`;
});

btnPlay.addEventListener('click', () => {
  running = !running;
  btnPlay.textContent = running ? 'Pause' : 'Play';
  if (running) loop();
});

btnReset.addEventListener('click', () => {
  model = new IsingModel2D(N, 1.0, Date.now());
  model.setTemperature(parseFloat(sliderT.value));
  renderLattice(model);
});

btnUp.addEventListener('click', () => { model.setAllUp(); });
btnDown.addEventListener('click', () => { model.setAllDown(); });

document.querySelectorAll('[data-preset]').forEach(btn => {
  btn.addEventListener('click', () => {
    const i = parseInt((btn as HTMLElement).dataset['preset']!);
    const p = PRESETS[i];
    model = new IsingModel2D(N, 1.0, Date.now());
    model.setTemperature(p.T);
    if (p.init === 'up') model.setAllUp();
    else model.randomise();
    sliderT.value = String(p.T.toFixed(3));
    valT.textContent = p.T.toFixed(3);
    renderLattice(model);
  });
});

// ---------------------------------------------------------------------------
// Animation loop
// ---------------------------------------------------------------------------
function loop(): void {
  if (!running) return;

  for (let k = 0; k < sweepsPerFrame; k++) model.step();
  renderLattice(model);

  const st = model.stats();
  const t  = model.getTemperature();
  statE.textContent      = st.eMean.toFixed(4);
  statM.textContent      = st.mMean.toFixed(4);
  statTel.textContent    = t.toFixed(3);
  statSweeps.textContent = st.sweeps.toLocaleString();

  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------
// Initial render
// ---------------------------------------------------------------------------
renderLattice(model);
loop();
