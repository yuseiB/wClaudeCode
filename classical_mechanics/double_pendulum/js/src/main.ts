import './style.css';
import { DoublePendulum, type State } from './double_pendulum';

// ── Constants ──────────────────────────────────────────────────────────────

const PHYSICS_DT   = 0.001;   // physics time step [s]
const MAX_TRAIL    = 600;     // pendulum bob-2 trail length (frames)
const MAX_PHASE    = 6000;    // phase portrait history (frames)
const MAX_TRAJ     = 4000;    // trajectory history (frames)
const MAX_ENERGY_F = 1800;    // energy plot history (~30 s at 60 fps)

const PRESETS: Record<string, State> = {
  'Near-linear':  { theta1:  10*Math.PI/180, omega1: 0, theta2:  10*Math.PI/180, omega2: 0 },
  'Intermediate': { theta1:  90*Math.PI/180, omega1: 0, theta2:   0,              omega2: 0 },
  'Chaotic':      { theta1: 120*Math.PI/180, omega1: 0, theta2: -30*Math.PI/180, omega2: 0 },
};

// ── DOM references ─────────────────────────────────────────────────────────

const pendulumCanvas = document.getElementById('pendulum-canvas') as HTMLCanvasElement;
const phaseCanvas    = document.getElementById('phase-canvas')    as HTMLCanvasElement;
const trajCanvas     = document.getElementById('traj-canvas')     as HTMLCanvasElement;
const energyCanvas   = document.getElementById('energy-canvas')   as HTMLCanvasElement;

const pCtx  = pendulumCanvas.getContext('2d')!;
const phCtx = phaseCanvas.getContext('2d')!;
const trCtx = trajCanvas.getContext('2d')!;
const enCtx = energyCanvas.getContext('2d')!;

const playPauseBtn  = document.getElementById('play-pause')    as HTMLButtonElement;
const resetBtn      = document.getElementById('reset')         as HTMLButtonElement;
const speedSelect   = document.getElementById('speed')         as HTMLSelectElement;
const theta1Input   = document.getElementById('theta1')        as HTMLInputElement;
const theta2Input   = document.getElementById('theta2')        as HTMLInputElement;
const omega1Input   = document.getElementById('omega1')        as HTMLInputElement;
const omega2Input   = document.getElementById('omega2')        as HTMLInputElement;
const theta1Label   = document.getElementById('theta1-label')  as HTMLSpanElement;
const theta2Label   = document.getElementById('theta2-label')  as HTMLSpanElement;
const omega1Label   = document.getElementById('omega1-label')  as HTMLSpanElement;
const omega2Label   = document.getElementById('omega2-label')  as HTMLSpanElement;
const timeDisplay   = document.getElementById('time-display')  as HTMLSpanElement;
const driftDisplay  = document.getElementById('drift-display') as HTMLSpanElement;
const presetBtns    = document.querySelectorAll<HTMLButtonElement>('.preset-btn');

// ── Offscreen canvases (accumulate phase / trajectory without clearing) ────

const phOff    = document.createElement('canvas');
phOff.width  = phaseCanvas.width;
phOff.height = phaseCanvas.height;
const phOffCtx = phOff.getContext('2d')!;

const trOff    = document.createElement('canvas');
trOff.width  = trajCanvas.width;
trOff.height = trajCanvas.height;
const trOffCtx = trOff.getContext('2d')!;

// ── App state ──────────────────────────────────────────────────────────────

const dp = new DoublePendulum();

let initState: State = { ...PRESETS['Near-linear'] };
let state: State     = { ...initState };
let simTime    = 0;
let energy0    = dp.energy(state);
let isRunning  = false;
let speed      = 1;
let lastTimestamp = 0;

type Pt = { x: number; y: number };
type PhasePt = { theta2: number; omega2: number };
type EnergyPt = { t: number; T: number; V: number; E: number };

const trail:         Pt[]       = [];
const phaseHistory:  PhasePt[]  = [];
const trajHistory:   Pt[]       = [];
const energyHistory: EnergyPt[] = [];

let prevPhasePt: PhasePt | null = null;
let prevTrajPt:  Pt      | null = null;

// ── Coordinate helpers ─────────────────────────────────────────────────────

/** Map a value from [inMin,inMax] → [outMin,outMax]. */
function mapRange(v: number, inMin: number, inMax: number, outMin: number, outMax: number) {
  return outMin + (v - inMin) / (inMax - inMin) * (outMax - outMin);
}

// Pendulum canvas: physical origin → canvas pixel
const PC_W = pendulumCanvas.width;
const PC_H = pendulumCanvas.height;
const PC_CX = PC_W / 2;
const PC_CY = PC_H / 2;
const PC_SCALE = (Math.min(PC_W, PC_H) / 2 - 24) / (dp.L1 + dp.L2);

function toP(x: number, y: number): [number, number] {
  return [PC_CX + x * PC_SCALE, PC_CY - y * PC_SCALE];
}

// Phase portrait canvas: (theta2, omega2) → pixel
const PH_W = phaseCanvas.width;
const PH_H = phaseCanvas.height;
const PH_MX = 18; // margin x
const PH_MY = 10; // margin y

function toPhase(theta2: number, omega2: number): [number, number] {
  const x = mapRange(theta2, -Math.PI, Math.PI, PH_MX, PH_W - PH_MX);
  const y = mapRange(omega2, 10, -10,             PH_MY, PH_H - PH_MY);
  return [x, y];
}

// Trajectory canvas: (x2, y2) physical → pixel
const TR_W  = trajCanvas.width;
const TR_H  = trajCanvas.height;
const TR_MX = 12;
const TR_MY = 10;
const TR_MAX = dp.L1 + dp.L2;

function toTraj(x: number, y: number): [number, number] {
  const px = mapRange(x, -TR_MAX, TR_MAX, TR_MX, TR_W - TR_MX);
  const py = mapRange(y, -TR_MAX, TR_MAX, TR_MY, TR_H - TR_MY);
  return [px, py];
}

// ── Offscreen canvas initialisation ───────────────────────────────────────

function initOffscreenBg(ctx: CanvasRenderingContext2D, W: number, H: number) {
  ctx.fillStyle = '#0d0d1a';
  ctx.fillRect(0, 0, W, H);
  // Centre cross-hairs
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(W/2, 0); ctx.lineTo(W/2, H);
  ctx.moveTo(0, H/2); ctx.lineTo(W, H/2);
  ctx.stroke();
}

function resetOffscreens() {
  initOffscreenBg(phOffCtx, PH_W, PH_H);
  initOffscreenBg(trOffCtx, TR_W, TR_H);
  prevPhasePt = null;
  prevTrajPt  = null;
}

// ── Incremental offscreen drawing ─────────────────────────────────────────

function addPhasePoint(s: State) {
  const curr: PhasePt = { theta2: s.theta2, omega2: s.omega2 };
  const [cx, cy] = toPhase(curr.theta2, curr.omega2);

  if (prevPhasePt) {
    const [px, py] = toPhase(prevPhasePt.theta2, prevPhasePt.omega2);
    phOffCtx.beginPath();
    phOffCtx.moveTo(px, py);
    phOffCtx.lineTo(cx, cy);
    phOffCtx.strokeStyle = 'rgba(179,100,255,0.75)';
    phOffCtx.lineWidth = 1;
    phOffCtx.stroke();
  } else {
    phOffCtx.fillStyle = 'rgba(179,100,255,0.9)';
    phOffCtx.fillRect(cx - 1, cy - 1, 2, 2);
  }
  prevPhasePt = curr;
}

function addTrajPoint(b2: Pt) {
  const [cx, cy] = toTraj(b2.x, b2.y);

  if (prevTrajPt) {
    const [px, py] = toTraj(prevTrajPt.x, prevTrajPt.y);
    trOffCtx.beginPath();
    trOffCtx.moveTo(px, py);
    trOffCtx.lineTo(cx, cy);
    trOffCtx.strokeStyle = 'rgba(255,140,0,0.65)';
    trOffCtx.lineWidth = 1;
    trOffCtx.stroke();
  }
  prevTrajPt = { ...b2 };
}

// ── Pendulum canvas ────────────────────────────────────────────────────────

function drawPendulum() {
  pCtx.fillStyle = '#0d0d1a';
  pCtx.fillRect(0, 0, PC_W, PC_H);

  const reach = (dp.L1 + dp.L2) * PC_SCALE;

  // Reach circle
  pCtx.beginPath();
  pCtx.arc(PC_CX, PC_CY, reach, 0, 2 * Math.PI);
  pCtx.strokeStyle = 'rgba(255,255,255,0.05)';
  pCtx.lineWidth = 1;
  pCtx.stroke();

  // Pivot marker
  pCtx.strokeStyle = 'rgba(255,255,255,0.35)';
  pCtx.lineWidth = 1;
  pCtx.beginPath();
  pCtx.moveTo(PC_CX - 7, PC_CY); pCtx.lineTo(PC_CX + 7, PC_CY);
  pCtx.moveTo(PC_CX, PC_CY - 7); pCtx.lineTo(PC_CX, PC_CY + 7);
  pCtx.stroke();

  // Bob-2 trail
  if (trail.length > 1) {
    pCtx.beginPath();
    const [x0, y0] = toP(trail[0].x, trail[0].y);
    pCtx.moveTo(x0, y0);
    for (let i = 1; i < trail.length; i++) {
      const [xi, yi] = toP(trail[i].x, trail[i].y);
      pCtx.lineTo(xi, yi);
    }
    pCtx.strokeStyle = 'rgba(255,140,0,0.28)';
    pCtx.lineWidth = 1.5;
    pCtx.stroke();
  }

  // Positions
  const b1 = dp.bob1(state);
  const b2 = dp.bob2(state);
  const [p1x, p1y] = toP(b1.x, b1.y);
  const [p2x, p2y] = toP(b2.x, b2.y);

  // Rods
  pCtx.strokeStyle = 'rgba(200,200,230,0.8)';
  pCtx.lineWidth = 3;
  pCtx.lineCap = 'round';
  pCtx.beginPath();
  pCtx.moveTo(PC_CX, PC_CY); pCtx.lineTo(p1x, p1y);
  pCtx.moveTo(p1x,   p1y);   pCtx.lineTo(p2x, p2y);
  pCtx.stroke();

  // Bob-1 (cyan)
  pCtx.beginPath();
  pCtx.arc(p1x, p1y, 9, 0, 2 * Math.PI);
  pCtx.fillStyle = '#00c8ff';
  pCtx.fill();

  // Bob-2 (orange)
  pCtx.beginPath();
  pCtx.arc(p2x, p2y, 11, 0, 2 * Math.PI);
  pCtx.fillStyle = '#ff8c00';
  pCtx.fill();

  // Angle labels
  pCtx.fillStyle = 'rgba(255,255,255,0.4)';
  pCtx.font = '11px monospace';
  pCtx.textAlign = 'left';
  const deg = (r: number) => `${(r * 180 / Math.PI).toFixed(1)}°`;
  pCtx.fillText(`θ₁=${deg(state.theta1)}  ω₁=${state.omega1.toFixed(2)}`, 8, PC_H - 18);
  pCtx.fillText(`θ₂=${deg(state.theta2)}  ω₂=${state.omega2.toFixed(2)}`, 8, PC_H - 4);
}

// ── Phase-portrait canvas ──────────────────────────────────────────────────

function drawPhaseCanvas() {
  // Draw background + grid (stored in offscreen already), then blit
  phCtx.drawImage(phOff, 0, 0);

  // Axis tick labels
  phCtx.fillStyle = 'rgba(255,255,255,0.28)';
  phCtx.font = '9px monospace';
  phCtx.textAlign = 'center';
  phCtx.fillText('-π', ...toPhase(-Math.PI, 0).map((v,i) => i===1 ? v + 12 : v) as [number,number]);
  phCtx.fillText(' 0', ...toPhase(0,        0).map((v,i) => i===1 ? v + 12 : v) as [number,number]);
  phCtx.fillText('+π', ...toPhase( Math.PI, 0).map((v,i) => i===1 ? v + 12 : v) as [number,number]);
  phCtx.textAlign = 'right';
  phCtx.fillText('-10', ...toPhase(0, -10).map((v,i) => i===0 ? v - 2 : v) as [number,number]);
  phCtx.fillText('+10', ...toPhase(0,  10).map((v,i) => i===0 ? v - 2 : v) as [number,number]);

  // Current point highlight
  const [cx, cy] = toPhase(state.theta2, state.omega2);
  phCtx.beginPath();
  phCtx.arc(cx, cy, 3, 0, 2 * Math.PI);
  phCtx.fillStyle = '#b364ff';
  phCtx.fill();
}

// ── Trajectory canvas ──────────────────────────────────────────────────────

function drawTrajCanvas() {
  trCtx.drawImage(trOff, 0, 0);

  // Axis tick labels
  trCtx.fillStyle = 'rgba(255,255,255,0.28)';
  trCtx.font = '9px monospace';
  trCtx.textAlign = 'center';
  trCtx.fillText(`-${TR_MAX}`, ...toTraj(-TR_MAX, 0).map((v,i) => i===1 ? v - 4 : v) as [number,number]);
  trCtx.fillText(`+${TR_MAX}`, ...toTraj( TR_MAX, 0).map((v,i) => i===1 ? v - 4 : v) as [number,number]);
  trCtx.textAlign = 'right';
  trCtx.fillText(`-${TR_MAX}`, ...toTraj(0, -TR_MAX).map((v,i) => i===0 ? v - 2 : v) as [number,number]);
  trCtx.fillText(`+${TR_MAX}`, ...toTraj(0,  TR_MAX).map((v,i) => i===0 ? v - 2 : v) as [number,number]);

  // Current bob-2 position
  const b2 = dp.bob2(state);
  const [cx, cy] = toTraj(b2.x, b2.y);
  trCtx.beginPath();
  trCtx.arc(cx, cy, 3, 0, 2 * Math.PI);
  trCtx.fillStyle = '#ff8c00';
  trCtx.fill();
}

// ── Energy canvas ──────────────────────────────────────────────────────────

function drawEnergyCanvas() {
  const W = energyCanvas.width;
  const H = energyCanvas.height;
  const mL = 38, mR = 8, mT = 8, mB = 18;
  const plotW = W - mL - mR;
  const plotH = H - mT - mB;

  enCtx.fillStyle = '#0d0d1a';
  enCtx.fillRect(0, 0, W, H);

  if (energyHistory.length < 2) return;

  const tMin = energyHistory[0].t;
  const tMax = energyHistory[energyHistory.length - 1].t;
  const tSpan = Math.max(tMax - tMin, 1);

  // Auto-scale y to include T, V, E
  let eMin = Infinity, eMax = -Infinity;
  for (const { T, V, E } of energyHistory) {
    eMin = Math.min(eMin, T, V, E);
    eMax = Math.max(eMax, T, V, E);
  }
  const ePad = Math.max((eMax - eMin) * 0.06, 0.01);
  eMin -= ePad; eMax += ePad;
  if (eMin === eMax) { eMin -= 1; eMax += 1; }

  function tx(t: number) { return mL + (t - tMin) / tSpan * plotW; }
  function ty(e: number) { return mT + (1 - (e - eMin) / (eMax - eMin)) * plotH; }

  // Zero-energy reference if in range
  if (eMin <= 0 && 0 <= eMax) {
    enCtx.strokeStyle = 'rgba(255,255,255,0.07)';
    enCtx.lineWidth = 1;
    enCtx.beginPath();
    enCtx.moveTo(mL, ty(0)); enCtx.lineTo(mL + plotW, ty(0));
    enCtx.stroke();
  }

  // Draw T, V, E lines
  const lines: [keyof EnergyPt, string][] = [['T', '#00e676'], ['V', '#4fc3f7'], ['E', '#ffffff']];
  for (const [key, color] of lines) {
    enCtx.beginPath();
    enCtx.strokeStyle = color;
    enCtx.lineWidth = key === 'E' ? 1.8 : 1.2;
    enCtx.globalAlpha = key === 'E' ? 1.0 : 0.75;
    for (let i = 0; i < energyHistory.length; i++) {
      const pt = energyHistory[i];
      const ex = tx(pt.t);
      const ey = ty(pt[key] as number);
      if (i === 0) enCtx.moveTo(ex, ey);
      else          enCtx.lineTo(ex, ey);
    }
    enCtx.stroke();
    enCtx.globalAlpha = 1;
  }

  // Y-axis labels
  enCtx.fillStyle = 'rgba(255,255,255,0.32)';
  enCtx.font = '9px monospace';
  enCtx.textAlign = 'right';
  enCtx.fillText(eMax.toFixed(1), mL - 2, mT + 8);
  enCtx.fillText(eMin.toFixed(1), mL - 2, mT + plotH);

  // X-axis labels
  enCtx.textAlign = 'left';
  enCtx.fillStyle = 'rgba(255,255,255,0.28)';
  enCtx.fillText(`${tMin.toFixed(0)}s`, mL, H - 3);
  enCtx.textAlign = 'right';
  enCtx.fillText(`${tMax.toFixed(0)}s`, mL + plotW, H - 3);
}

// ── Render frame ───────────────────────────────────────────────────────────

function render() {
  drawPendulum();
  drawPhaseCanvas();
  drawTrajCanvas();
  drawEnergyCanvas();

  // Update HUD
  timeDisplay.textContent  = simTime.toFixed(2);
  const driftPct = energy0 !== 0
    ? Math.abs((dp.energy(state) - energy0) / energy0) * 100
    : 0;
  driftDisplay.textContent = driftPct.toExponential(2);
}

// ── Record per-frame data ──────────────────────────────────────────────────

function recordFrame() {
  const b2 = dp.bob2(state);

  // Pendulum trail
  trail.push({ ...b2 });
  if (trail.length > MAX_TRAIL) trail.shift();

  // Phase portrait (offscreen + history)
  addPhasePoint(state);
  phaseHistory.push({ theta2: state.theta2, omega2: state.omega2 });
  if (phaseHistory.length > MAX_PHASE) phaseHistory.shift();

  // Trajectory (offscreen + history)
  addTrajPoint(b2);
  trajHistory.push({ ...b2 });
  if (trajHistory.length > MAX_TRAJ) trajHistory.shift();

  // Energy
  energyHistory.push({ t: simTime, T: dp.kineticEnergy(state), V: dp.potentialEnergy(state), E: dp.energy(state) });
  if (energyHistory.length > MAX_ENERGY_F) energyHistory.shift();
}

// ── Animation loop ─────────────────────────────────────────────────────────

function animate(timestamp: number) {
  if (isRunning) {
    const wallDt = lastTimestamp > 0 ? Math.min((timestamp - lastTimestamp) / 1000, 0.1) : 0;
    const simDt  = wallDt * speed;
    const steps  = Math.max(1, Math.round(simDt / PHYSICS_DT));
    const dt     = simDt / steps;

    for (let i = 0; i < steps; i++) {
      state    = dp.step(state, dt);
      simTime += dt;
    }
    recordFrame();
  }

  lastTimestamp = timestamp;
  render();
  requestAnimationFrame(animate);
}

// ── Reset ──────────────────────────────────────────────────────────────────

function reset() {
  state   = { ...initState };
  simTime = 0;
  energy0 = dp.energy(state);
  trail.length         = 0;
  phaseHistory.length  = 0;
  trajHistory.length   = 0;
  energyHistory.length = 0;
  resetOffscreens();
  render();
}

// ── Sync controls → initState ──────────────────────────────────────────────

function syncControls() {
  const th1 = parseFloat(theta1Input.value) * Math.PI / 180;
  const th2 = parseFloat(theta2Input.value) * Math.PI / 180;
  const w1  = parseFloat(omega1Input.value);
  const w2  = parseFloat(omega2Input.value);

  theta1Label.textContent = `${(+theta1Input.value).toFixed(0)}°`;
  theta2Label.textContent = `${(+theta2Input.value).toFixed(0)}°`;
  omega1Label.textContent = `${w1.toFixed(1)}`;
  omega2Label.textContent = `${w2.toFixed(1)}`;

  initState = { theta1: th1, omega1: w1, theta2: th2, omega2: w2 };
}

function applyPreset(name: string) {
  const p = PRESETS[name];
  if (!p) return;

  theta1Input.value = `${(p.theta1 * 180 / Math.PI).toFixed(0)}`;
  theta2Input.value = `${(p.theta2 * 180 / Math.PI).toFixed(0)}`;
  omega1Input.value = `${p.omega1}`;
  omega2Input.value = `${p.omega2}`;
  syncControls();
  reset();

  presetBtns.forEach(btn => btn.classList.toggle('active', btn.dataset['preset'] === name));
}

// ── Event handlers ─────────────────────────────────────────────────────────

playPauseBtn.addEventListener('click', () => {
  isRunning = !isRunning;
  playPauseBtn.textContent = isRunning ? '⏸ Pause' : '▶ Play';
  if (isRunning) lastTimestamp = 0;
});

resetBtn.addEventListener('click', () => {
  syncControls();
  reset();
});

speedSelect.addEventListener('change', () => {
  speed = parseFloat(speedSelect.value);
});

[theta1Input, theta2Input, omega1Input, omega2Input].forEach(input => {
  input.addEventListener('input', () => {
    syncControls();
    // Deselect all presets when manually changed
    presetBtns.forEach(btn => btn.classList.remove('active'));
  });
});

presetBtns.forEach(btn => {
  btn.addEventListener('click', () => applyPreset(btn.dataset['preset'] ?? ''));
});

// ── Initialise ─────────────────────────────────────────────────────────────

applyPreset('Near-linear');
requestAnimationFrame(animate);
