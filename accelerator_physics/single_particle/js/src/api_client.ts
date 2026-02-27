/**
 * api_client.ts — REST client for the Accelerator Physics FastAPI backend.
 *
 * The Python server must be running at API_BASE before calling these functions.
 * If the server is unreachable, every function rejects with a descriptive error.
 */

export const API_BASE = "http://localhost:8000";

// ── JSON types (mirrors schemas.py) ─────────────────────────────────────────

export interface ElementSpec {
  type: "drift" | "quadrupole" | "dipole" | "sextupole" | "rf" | "marker";
  name: string;
  length: number;
  params: Record<string, number>;
}

export interface LatticeRequest {
  elements: ElementSpec[];
  beam_energy_gev: number;
  n_repeats: number;
}

export interface TwissResponse {
  s: number[];
  beta_x: number[];
  alpha_x: number[];
  gamma_x: number[];
  beta_y: number[];
  alpha_y: number[];
  gamma_y: number[];
  Dx: number[];
  Dpx: number[];
  tune_x: number;
  tune_y: number;
  chromaticity_x: number;
  chromaticity_y: number;
  momentum_compaction: number;
  circumference: number;
}

export interface TrackRequest {
  lattice: LatticeRequest;
  particles: number[][];   // (n_particles, 6)
  n_turns: number;
}

export interface TrackResponse {
  data: number[][][];      // (n_turns+1, n_particles, 6)
  n_turns: number;
  n_particles: number;
}

// ── Generic fetch helper ────────────────────────────────────────────────────

async function apiFetch<T>(
  method: "GET" | "POST",
  path: string,
  body?: unknown,
): Promise<T> {
  const url = API_BASE + path;
  const opts: RequestInit = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body !== undefined) opts.body = JSON.stringify(body);

  let res: Response;
  try {
    res = await fetch(url, opts);
  } catch {
    throw new Error(
      `API サーバーに接続できません (${url})。\n` +
      `uvicorn を起動してください:\n` +
      `  uvicorn app:app --port 8000 --app-dir …/api`,
    );
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j.detail) detail += `: ${typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)}`;
    } catch {/* ignore */}
    throw new Error(detail);
  }

  return res.json() as Promise<T>;
}

// ── Public API ───────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
  try {
    await apiFetch<{ status: string }>("GET", "/");
    return true;
  } catch {
    return false;
  }
}

export async function getFodoPreset(): Promise<LatticeRequest> {
  return apiFetch<LatticeRequest>("GET", "/lattice/fodo");
}

export async function getRingPreset(): Promise<LatticeRequest> {
  return apiFetch<LatticeRequest>("GET", "/lattice/ring");
}

export async function computeTwiss(req: LatticeRequest): Promise<TwissResponse> {
  return apiFetch<TwissResponse>("POST", "/twiss", req);
}

export async function trackBeam(req: TrackRequest): Promise<TrackResponse> {
  return apiFetch<TrackResponse>("POST", "/track", req);
}
