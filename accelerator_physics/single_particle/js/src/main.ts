/**
 * main.ts — Accelerator Physics Browser Visualizer
 *
 * Architecture: TypeScript frontend → fetch → Python FastAPI (localhost:8000)
 *
 * Panels:
 *   1. Lattice layout bar (colored by element type)
 *   2. Twiss functions β_x, β_y, D_x vs s
 *   3. Phase space x–px (Poincaré section from tracking)
 *   4. Tune diagram with resonance lines
 */

import {
  API_BASE,
  type ElementSpec,
  type LatticeRequest,
  type TrackResponse,
  type TwissResponse,
  checkHealth,
  computeTwiss,
  getFodoPreset,
  getRingPreset,
  trackBeam,
} from "./api_client.js";

// ── Canvas elements ──────────────────────────────────────────────────────────

const cnvLattice = document.getElementById("latticeCanvas") as HTMLCanvasElement;
const cnvTwiss   = document.getElementById("twissCanvas")   as HTMLCanvasElement;
const cnvPhase   = document.getElementById("phaseCanvas")   as HTMLCanvasElement;
const cnvTune    = document.getElementById("tuneCanvas")    as HTMLCanvasElement;

const ctxLat   = cnvLattice.getContext("2d")!;
const ctxTwiss = cnvTwiss.getContext("2d")!;
const ctxPhase = cnvPhase.getContext("2d")!;
const ctxTune  = cnvTune.getContext("2d")!;

// ── DOM refs ─────────────────────────────────────────────────────────────────

const apiStatusEl   = document.getElementById("apiStatus")     as HTMLElement;
const latticeJsonEl = document.getElementById("latticeJson")    as HTMLTextAreaElement;
const infoPanelEl   = document.getElementById("infoPanel")      as HTMLElement;
const nPartSlider   = document.getElementById("nPartSlider")    as HTMLInputElement;
const nPartVal      = document.getElementById("nPartVal")       as HTMLElement;
const nTurnSlider   = document.getElementById("nTurnSlider")    as HTMLInputElement;
const nTurnVal      = document.getElementById("nTurnVal")       as HTMLElement;

// ── State ─────────────────────────────────────────────────────────────────────

let lastTwiss: TwissResponse | null = null;
let lastLattice: LatticeRequest | null = null;

// ── Colour scheme ─────────────────────────────────────────────────────────────

const ELEM_COLORS: Record<string, string> = {
  drift:      "#2a2f3a",
  quadrupole: "",   // handled by k1 sign
  qf:         "#c0392b",
  qd:         "#2980b9",
  dipole:     "#d4a017",
  sextupole:  "#8e44ad",
  rf:         "#27ae60",
  marker:     "#555",
};

const PALETTE = [
  "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
  "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
];

// ── Helpers ───────────────────────────────────────────────────────────────────

function setStatus(ok: boolean | null): void {
  apiStatusEl.className = "status-badge";
  if (ok === null) {
    apiStatusEl.classList.add("status-unknown");
    apiStatusEl.textContent = "確認中…";
  } else if (ok) {
    apiStatusEl.classList.add("status-ok");
    apiStatusEl.textContent = `接続済 (${API_BASE})`;
  } else {
    apiStatusEl.classList.add("status-error");
    apiStatusEl.textContent = "未接続";
  }
}

function setInfo(html: string): void {
  infoPanelEl.innerHTML = html;
}

function showError(msg: string): void {
  setInfo(`<span style="color:#f85149">エラー: ${msg}</span>`);
}

function elemColor(spec: ElementSpec): string {
  if (spec.type === "quadrupole") {
    const k1 = spec.params.k1 ?? 0;
    return k1 >= 0 ? ELEM_COLORS.qf : ELEM_COLORS.qd;
  }
  return ELEM_COLORS[spec.type] ?? "#444";
}

function currentLatticeRequest(): LatticeRequest | null {
  try {
    return JSON.parse(latticeJsonEl.value) as LatticeRequest;
  } catch {
    showError("ラティス JSON の解析に失敗しました。");
    return null;
  }
}

// ── Clear canvases ─────────────────────────────────────────────────────────────

function clearCanvas(ctx: CanvasRenderingContext2D, bg = "#0d1117"): void {
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

// ── Draw lattice layout bar ───────────────────────────────────────────────────

function drawLatticeBar(lat: LatticeRequest): void {
  clearCanvas(ctxLat);
  const W = cnvLattice.width;
  const H = cnvLattice.height;
  const PAD = 8;

  // Flatten elements * n_repeats
  const elems: ElementSpec[] = [];
  for (let r = 0; r < lat.n_repeats; r++) {
    for (const e of lat.elements) elems.push(e);
  }

  const totalLen = elems.reduce((s, e) => s + e.length, 0);
  if (totalLen <= 0) return;

  let x = PAD;
  const barW = W - 2 * PAD;
  const barH = H - 2 * PAD;
  const y = PAD;

  for (const e of elems) {
    const w = Math.max(1, (e.length / totalLen) * barW);
    ctxLat.fillStyle = elemColor(e);
    ctxLat.fillRect(x, y, w, barH);
    // Thin border
    ctxLat.strokeStyle = "#0d1117";
    ctxLat.lineWidth = 0.5;
    ctxLat.strokeRect(x, y, w, barH);
    x += w;
  }

  // Legend
  const legendItems: [string, string][] = [
    ["QF", ELEM_COLORS.qf],
    ["QD", ELEM_COLORS.qd],
    ["Dipole", ELEM_COLORS.dipole],
    ["Drift", ELEM_COLORS.drift],
    ["Sext", ELEM_COLORS.sextupole],
    ["RF", ELEM_COLORS.rf],
  ];
  let lx = PAD;
  ctxLat.font = "10px system-ui";
  for (const [label, color] of legendItems) {
    ctxLat.fillStyle = color;
    ctxLat.fillRect(lx, H - 3, 14, 3);
    ctxLat.fillStyle = "#8b949e";
    ctxLat.fillText(label, lx, H - 6);
    lx += ctxLat.measureText(label).width + 18;
  }
}

// ── Draw Twiss functions ──────────────────────────────────────────────────────

function drawTwiss(tw: TwissResponse): void {
  clearCanvas(ctxTwiss);
  const W = cnvTwiss.width;
  const H = cnvTwiss.height;
  const PL = 48, PR = 16, PT = 16, PB = 28;  // margins
  const iW = W - PL - PR;
  const iH = H - PT - PB;

  const s = tw.s;
  const sMax = s[s.length - 1];

  // Y-axis for beta (left): 0..max_beta
  const maxBeta = Math.max(...tw.beta_x, ...tw.beta_y) * 1.1;
  // Y-axis for Dx (right): scaled separately
  const maxDx = Math.max(...tw.Dx.map(Math.abs)) * 1.2 || 1.0;

  function sx(sVal: number): number { return PL + (sVal / sMax) * iW; }
  function syBeta(v: number): number { return PT + iH - (v / maxBeta) * iH; }
  function syDx(v: number): number {
    return PT + iH / 2 - (v / maxDx) * (iH / 2);
  }

  // Grid
  ctxTwiss.strokeStyle = "#21262d";
  ctxTwiss.lineWidth = 0.8;
  for (let i = 0; i <= 4; i++) {
    const gy = PT + (i / 4) * iH;
    ctxTwiss.beginPath();
    ctxTwiss.moveTo(PL, gy); ctxTwiss.lineTo(PL + iW, gy);
    ctxTwiss.stroke();
  }

  // Shade element regions (quads = faint red/blue, dipoles = faint yellow)
  if (lastLattice) {
    const elems = [...Array(lastLattice.n_repeats)].flatMap(() => lastLattice!.elements);
    let ss = 0;
    for (const e of elems) {
      const s1 = ss + e.length;
      if (e.type === "quadrupole") {
        const k1 = e.params.k1 ?? 0;
        ctxTwiss.fillStyle = k1 >= 0 ? "rgba(192,57,43,0.08)" : "rgba(41,128,185,0.08)";
        ctxTwiss.fillRect(sx(ss), PT, sx(s1) - sx(ss), iH);
      } else if (e.type === "dipole") {
        ctxTwiss.fillStyle = "rgba(212,160,23,0.07)";
        ctxTwiss.fillRect(sx(ss), PT, sx(s1) - sx(ss), iH);
      }
      ss = s1;
    }
  }

  // Plot lines
  function drawLine(
    vals: number[],
    toY: (v: number) => number,
    color: string,
    dash: number[] = [],
  ): void {
    ctxTwiss.beginPath();
    ctxTwiss.strokeStyle = color;
    ctxTwiss.lineWidth = 1.8;
    ctxTwiss.setLineDash(dash);
    for (let i = 0; i < s.length; i++) {
      const x = sx(s[i]), y = toY(vals[i]);
      i === 0 ? ctxTwiss.moveTo(x, y) : ctxTwiss.lineTo(x, y);
    }
    ctxTwiss.stroke();
    ctxTwiss.setLineDash([]);
  }

  drawLine(tw.beta_x, syBeta, "#e74c3c");
  drawLine(tw.beta_y, syBeta, "#3498db");
  drawLine(tw.Dx.map(v => v * 100), syDx, "#2ecc71", [4, 3]);  // D_x in cm

  // Axes
  ctxTwiss.strokeStyle = "#30363d";
  ctxTwiss.lineWidth = 1;
  ctxTwiss.beginPath();
  ctxTwiss.moveTo(PL, PT); ctxTwiss.lineTo(PL, PT + iH); ctxTwiss.lineTo(PL + iW, PT + iH);
  ctxTwiss.stroke();

  // Axis labels
  ctxTwiss.fillStyle = "#8b949e";
  ctxTwiss.font = "11px system-ui";
  ctxTwiss.fillText(`0`, PL - 18, PT + iH + 4);
  ctxTwiss.fillText(`${sMax.toFixed(1)} m`, PL + iW - 20, PT + iH + 14);
  ctxTwiss.fillText(`β_max=${maxBeta.toFixed(1)} m`, PL + 4, PT + 12);
  ctxTwiss.fillText(`D_x [cm]`, PL + 4, PT + 24);

  // Legend
  const lx0 = PL + iW - 130;
  ctxTwiss.font = "11px system-ui";
  ctxTwiss.fillStyle = "#e74c3c"; ctxTwiss.fillText("─ β_x",  lx0,       PT + 12);
  ctxTwiss.fillStyle = "#3498db"; ctxTwiss.fillText("─ β_y",  lx0 + 44,  PT + 12);
  ctxTwiss.fillStyle = "#2ecc71"; ctxTwiss.fillText("-- D_x", lx0 + 88,  PT + 12);

  // Zero line for dispersion
  ctxTwiss.strokeStyle = "#2ecc71";
  ctxTwiss.lineWidth = 0.5;
  ctxTwiss.setLineDash([2, 2]);
  ctxTwiss.beginPath();
  ctxTwiss.moveTo(PL, syDx(0)); ctxTwiss.lineTo(PL + iW, syDx(0));
  ctxTwiss.stroke();
  ctxTwiss.setLineDash([]);
}

// ── Draw tune diagram ─────────────────────────────────────────────────────────

function drawTuneDiagram(Qx: number, Qy: number): void {
  clearCanvas(ctxTune);
  const W = cnvTune.width;
  const H = cnvTune.height;
  const PAD = 32;
  const iW = W - 2 * PAD, iH = H - 2 * PAD;

  function tx(q: number): number { return PAD + q * iW; }
  function ty(q: number): number { return PAD + (1 - q) * iH; }

  // Grid
  ctxTune.strokeStyle = "#21262d";
  ctxTune.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    ctxTune.beginPath();
    ctxTune.moveTo(PAD + (i / 4) * iW, PAD);
    ctxTune.lineTo(PAD + (i / 4) * iW, PAD + iH);
    ctxTune.stroke();
    ctxTune.beginPath();
    ctxTune.moveTo(PAD, PAD + (i / 4) * iH);
    ctxTune.lineTo(PAD + iW, PAD + (i / 4) * iH);
    ctxTune.stroke();
  }

  // Resonance lines n*Qx + m*Qy = p
  const orderColors: Record<number, string> = {
    1: "rgba(231,76,60,0.55)",
    2: "rgba(230,126,34,0.45)",
    3: "rgba(149,165,166,0.35)",
    4: "rgba(189,195,199,0.25)",
  };
  const orderWidths: Record<number, number> = { 1: 1.4, 2: 1.0, 3: 0.7, 4: 0.5 };

  for (let order = 1; order <= 4; order++) {
    ctxTune.strokeStyle = orderColors[order];
    ctxTune.lineWidth = orderWidths[order];
    for (let n = -order; n <= order; n++) {
      for (let m = -order; m <= order; m++) {
        if (Math.abs(n) + Math.abs(m) !== order) continue;
        for (let p = -order; p <= order; p++) {
          ctxTune.beginPath();
          if (m !== 0) {
            const q0 = 0, q1 = 1;
            const qy0 = (p - n * q0) / m;
            const qy1 = (p - n * q1) / m;
            // Clip to [0,1]
            const pts: [number, number][] = [];
            for (const [qx, qy] of [[q0, qy0], [q1, qy1]] as [number, number][]) {
              if (qy >= 0 && qy <= 1) pts.push([qx, qy]);
            }
            // Find intersections with qy=0 and qy=1
            if (m !== 0) {
              const qx_at0 = (p - m * 0) / n;
              const qx_at1 = (p - m * 1) / n;
              if (n !== 0 && qx_at0 >= 0 && qx_at0 <= 1) pts.push([qx_at0, 0]);
              if (n !== 0 && qx_at1 >= 0 && qx_at1 <= 1) pts.push([qx_at1, 1]);
            }
            if (pts.length >= 2) {
              ctxTune.moveTo(tx(pts[0][0]), ty(pts[0][1]));
              ctxTune.lineTo(tx(pts[pts.length - 1][0]), ty(pts[pts.length - 1][1]));
            }
          } else if (n !== 0) {
            const qx = p / n;
            if (qx >= 0 && qx <= 1) {
              ctxTune.moveTo(tx(qx), ty(0));
              ctxTune.lineTo(tx(qx), ty(1));
            }
          }
          ctxTune.stroke();
        }
      }
    }
  }

  // Axes
  ctxTune.strokeStyle = "#30363d";
  ctxTune.lineWidth = 1;
  ctxTune.strokeRect(PAD, PAD, iW, iH);

  // Axis labels
  ctxTune.fillStyle = "#8b949e";
  ctxTune.font = "11px system-ui";
  ctxTune.fillText("0", PAD - 14, ty(0) + 4);
  ctxTune.fillText("1", PAD - 14, ty(1) + 4);
  ctxTune.fillText("Q_x", tx(0.5) - 10, ty(0) + 20);
  ctxTune.save();
  ctxTune.translate(PAD - 20, ty(0.5));
  ctxTune.rotate(-Math.PI / 2);
  ctxTune.fillText("Q_y", -12, 0);
  ctxTune.restore();
  ctxTune.fillText("0", tx(0) - 4, ty(0) + 20);
  ctxTune.fillText("1", tx(1) - 4, ty(0) + 20);

  // Operating point
  const qxFrac = ((Qx % 1) + 1) % 1;
  const qyFrac = ((Qy % 1) + 1) % 1;
  ctxTune.beginPath();
  ctxTune.arc(tx(qxFrac), ty(qyFrac), 6, 0, 2 * Math.PI);
  ctxTune.fillStyle = "#f85149";
  ctxTune.fill();
  ctxTune.strokeStyle = "#fff";
  ctxTune.lineWidth = 1.5;
  ctxTune.stroke();

  // Label
  ctxTune.fillStyle = "#f85149";
  ctxTune.font = "bold 11px system-ui";
  ctxTune.fillText(
    `(${qxFrac.toFixed(3)}, ${qyFrac.toFixed(3)})`,
    tx(qxFrac) + 8, ty(qyFrac) - 6,
  );
}

// ── Draw phase space ──────────────────────────────────────────────────────────

function drawPhaseSpace(hist: TrackResponse): void {
  clearCanvas(ctxPhase);
  const W = cnvPhase.width;
  const H = cnvPhase.height;
  const PAD = 36;
  const iW = W - 2 * PAD, iH = H - 2 * PAD;

  // Collect all x, px values across turns
  const allX: number[] = [], allPx: number[] = [];
  for (const turn of hist.data) {
    for (const p of turn) {
      allX.push(p[0] * 1e3);    // mm
      allPx.push(p[1] * 1e3);   // mrad
    }
  }
  const xmax = Math.max(...allX.map(Math.abs)) * 1.15 || 1;
  const pmax = Math.max(...allPx.map(Math.abs)) * 1.15 || 1;

  function tx(v: number): number { return PAD + ((v + xmax) / (2 * xmax)) * iW; }
  function ty(v: number): number { return PAD + ((pmax - v) / (2 * pmax)) * iH; }

  // Grid + axes
  ctxPhase.strokeStyle = "#21262d";
  ctxPhase.lineWidth = 0.7;
  ctxPhase.beginPath();
  ctxPhase.moveTo(tx(0), PAD); ctxPhase.lineTo(tx(0), PAD + iH);
  ctxPhase.moveTo(PAD, ty(0)); ctxPhase.lineTo(PAD + iW, ty(0));
  ctxPhase.stroke();

  // Points: colour by particle index, fade by turn
  const nTurns = hist.n_turns;
  const nPart  = hist.n_particles;

  for (let ip = 0; ip < nPart; ip++) {
    const color = PALETTE[ip % PALETTE.length];
    for (let t = 0; t <= nTurns; t++) {
      const p = hist.data[t][ip];
      const alpha = 0.3 + 0.7 * (t / nTurns);
      ctxPhase.fillStyle = color + Math.round(alpha * 255).toString(16).padStart(2, "0");
      ctxPhase.fillRect(tx(p[0] * 1e3) - 1, ty(p[1] * 1e3) - 1, 2, 2);
    }
  }

  // Axes labels
  ctxPhase.fillStyle = "#8b949e";
  ctxPhase.font = "11px system-ui";
  ctxPhase.fillText(`x [mm]`, PAD + iW - 30, PAD + iH + 14);
  ctxPhase.save();
  ctxPhase.translate(14, PAD + iH / 2 + 20);
  ctxPhase.rotate(-Math.PI / 2);
  ctxPhase.fillText("px [mrad]", 0, 0);
  ctxPhase.restore();
  ctxPhase.fillText(`±${xmax.toFixed(2)}mm`, PAD, PAD - 6);
}

// ── Info panel ─────────────────────────────────────────────────────────────────

function showTwissInfo(tw: TwissResponse): void {
  setInfo(`
    <b>Q_x</b> = ${tw.tune_x.toFixed(4)}<br>
    <b>Q_y</b> = ${tw.tune_y.toFixed(4)}<br>
    <b>ξ_x</b> = ${tw.chromaticity_x.toFixed(3)}<br>
    <b>ξ_y</b> = ${tw.chromaticity_y.toFixed(3)}<br>
    <b>α_c</b> = ${tw.momentum_compaction.toExponential(3)}<br>
    <b>C</b>   = ${tw.circumference.toFixed(2)} m<br>
    <b>β_x,max</b> = ${Math.max(...tw.beta_x).toFixed(2)} m<br>
    <b>β_y,max</b> = ${Math.max(...tw.beta_y).toFixed(2)} m<br>
    <b>D_x,max</b> = ${(Math.max(...tw.Dx.map(Math.abs)) * 100).toFixed(2)} cm
  `);
}

// ── Event handlers ────────────────────────────────────────────────────────────

document.getElementById("presetFodo")!.addEventListener("click", async () => {
  try {
    const req = await getFodoPreset();
    lastLattice = req;
    latticeJsonEl.value = JSON.stringify(req, null, 2);
    drawLatticeBar(req);
    setInfo("FODO プリセットを読み込みました。「Twiss 計算」を押してください。");
  } catch (e) {
    showError(String(e));
  }
});

document.getElementById("presetRing")!.addEventListener("click", async () => {
  try {
    const req = await getRingPreset();
    lastLattice = req;
    latticeJsonEl.value = JSON.stringify(req, null, 2);
    drawLatticeBar(req);
    setInfo("AG リングプリセットを読み込みました。「Twiss 計算」を押してください。");
  } catch (e) {
    showError(String(e));
  }
});

document.getElementById("computeTwiss")!.addEventListener("click", async () => {
  const req = currentLatticeRequest();
  if (!req) return;
  lastLattice = req;
  setInfo("計算中…");
  try {
    const tw = await computeTwiss(req);
    lastTwiss = tw;
    drawLatticeBar(req);
    drawTwiss(tw);
    drawTuneDiagram(tw.tune_x, tw.tune_y);
    showTwissInfo(tw);
  } catch (e) {
    showError(String(e));
  }
});

document.getElementById("computeTrack")!.addEventListener("click", async () => {
  const req = currentLatticeRequest();
  if (!req) return;
  lastLattice = req;

  if (!lastTwiss) {
    showError("まず「Twiss 計算」を実行してください。");
    return;
  }

  const nPart  = parseInt(nPartSlider.value);
  const nTurns = parseInt(nTurnSlider.value);

  // Generate matched Gaussian beam
  const eps = 1e-6;  // 1 μm·rad emittance
  const bx  = lastTwiss.beta_x[0];
  const ax  = lastTwiss.alpha_x[0];
  const gx  = lastTwiss.gamma_x[0];
  const by  = lastTwiss.beta_y[0];
  const ay  = lastTwiss.alpha_y[0];
  const gy  = lastTwiss.gamma_y[0];

  // Cholesky decomposition of sigma matrix Σ = ε [[β,-α],[-α,γ]]
  function choleskySigma(b: number, a: number, g: number, em: number): [number, number, number] {
    // Σ = [[em*b, -em*a], [-em*a, em*g]]
    // L = [[l11, 0], [l21, l22]] where L L^T = Σ
    const l11 = Math.sqrt(em * b);
    const l21 = -em * a / l11;
    const l22 = Math.sqrt(em * g - l21 * l21);
    return [l11, l21, l22];
  }

  function randn(): number {
    // Box-Muller
    const u1 = Math.random(), u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1 + 1e-300)) * Math.cos(2 * Math.PI * u2);
  }

  const [lx11, lx21, lx22] = choleskySigma(bx, ax, gx, eps);
  const [ly11, ly21, ly22] = choleskySigma(by, ay, gy, eps);

  const particles: number[][] = [];
  for (let i = 0; i < nPart; i++) {
    const zx1 = randn(), zx2 = randn();
    const zy1 = randn(), zy2 = randn();
    particles.push([
      lx11 * zx1,               // x
      lx21 * zx1 + lx22 * zx2,  // px
      ly11 * zy1,                // y
      ly21 * zy1 + ly22 * zy2,  // py
      0, 0,                      // delta, l
    ]);
  }

  setInfo(`追跡中: ${nPart} 粒子 × ${nTurns} ターン…`);
  try {
    const resp = await trackBeam({ lattice: req, particles, n_turns: nTurns });
    drawPhaseSpace(resp);
    setInfo(
      `追跡完了: ${nPart} 粒子 × ${nTurns} ターン<br>` +
      (lastTwiss
        ? `Q_x = ${lastTwiss.tune_x.toFixed(4)}, Q_y = ${lastTwiss.tune_y.toFixed(4)}`
        : ""),
    );
  } catch (e) {
    showError(String(e));
  }
});

// ── Sliders ───────────────────────────────────────────────────────────────────

nPartSlider.addEventListener("input", () => { nPartVal.textContent = nPartSlider.value; });
nTurnSlider.addEventListener("input", () => { nTurnVal.textContent = nTurnSlider.value; });

// ── Initialise ────────────────────────────────────────────────────────────────

async function init(): Promise<void> {
  setStatus(null);
  clearCanvas(ctxLat);
  clearCanvas(ctxTwiss);
  clearCanvas(ctxPhase);
  clearCanvas(ctxTune);

  const ok = await checkHealth();
  setStatus(ok);

  if (!ok) {
    setInfo(
      "Python API サーバーに接続できません。<br>" +
      "以下のコマンドでサーバーを起動してください:<br><br>" +
      "<code>uvicorn app:app --reload --port 8000<br>" +
      "  --app-dir accelerator_physics/single_particle/python/api</code>",
    );
  } else {
    setInfo("接続済。プリセットを選択するか、JSON を入力して「Twiss 計算」を押してください。");
  }
}

init();
