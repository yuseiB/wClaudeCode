# CLAUDE.md — MathPhys Repository Guide

> Guidance for AI assistants working in this codebase.

## Project Overview

**MathPhys** is a multi-language educational toolkit implementing exact nonlinear
double-pendulum dynamics. The same physics model is coded in five languages to
compare performance, idioms, and numerical approaches.

| Language | Integrator | Output |
|---|---|---|
| Python | Adaptive RK45 (SciPy DOP853, rtol=1e-9) | matplotlib figure, CSV |
| C++ | Fixed-step RK4 | CSV files |
| Rust | Fixed-step RK4 | CSV files |
| Julia | Fixed-step RK4 | CSV files |
| JavaScript/TypeScript | Fixed-step RK4 | Live browser canvas |

---

## Repository Layout

```
wClaudeCode/
├── pyproject.toml              # Python project metadata + tool config
├── README.md
├── double_pendulum.png         # Demo output (committed artifact)
├── .claude/
│   ├── settings.json           # Claude Code hook configuration
│   └── hooks/session-start.sh  # Auto-setup script for remote sessions
├── python/
│   ├── src/mathphys/
│   │   ├── __init__.py
│   │   ├── double_pendulum.py  # DoublePendulum class, adaptive RK45
│   │   └── numerics.py         # integrate_trapezoid, finite_difference
│   ├── tests/test_numerics.py
│   └── examples/double_pendulum_demo.py
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── double_pendulum.hpp
│   │   └── numerics.hpp
│   ├── src/
│   │   ├── double_pendulum.cpp
│   │   └── numerics.cpp
│   ├── tests/test_numerics.cpp
│   ├── examples/dp_sim.cpp
│   └── build/                  # CMake build artifacts (not committed)
├── rust/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   ├── main.rs
│   │   ├── double_pendulum.rs
│   │   ├── numerics.rs
│   │   └── bin/dp_sim.rs
│   └── target/                 # Cargo build artifacts (not committed)
├── julia/
│   ├── Project.toml
│   ├── src/
│   │   ├── MathPhys.jl
│   │   └── DoublePendulum.jl
│   ├── tests/runtests.jl
│   └── examples/dp_sim.jl
└── js/
    ├── package.json
    ├── tsconfig.json
    ├── index.html
    ├── src/
    │   ├── main.ts             # 500+ line browser UI, Canvas rendering
    │   ├── double_pendulum.ts
    │   └── style.css
    └── dist/                   # Pre-built bundle (committed, open directly)
```

---

## Build Systems

### Python

```bash
# Install package in editable mode with dev extras
pip install -e ".[dev]"

# The package source lives under python/src/; set PYTHONPATH if needed:
export PYTHONPATH="$PWD/python/src"
```

Tool config lives in `pyproject.toml`:
- **Linter:** `ruff` (line length 100, rules E/F/W/I/N/UP)
- **Tests:** `pytest`, test paths = `python/tests/`
- **Build backend:** `setuptools>=68`

### C++

```bash
# Configure (from repo root)
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build cpp/build --parallel

# Targets produced:
#   cpp/build/test_numerics   — test executable
#   cpp/build/dp_sim_exec     — simulation example
```

Requirements: CMake ≥ 3.20, GCC 12+ or Clang 15+ (C++20).

### Rust

```bash
cd rust
cargo build           # debug build
cargo build --release # release build

# Binaries:
#   target/debug/mathphys or target/release/mathphys (main.rs)
#   target/debug/dp_sim   or target/release/dp_sim   (bin/dp_sim.rs)
```

Requirements: Rust 1.75+ stable. Key dependency: `nalgebra = "0.33"`.

### Julia

```bash
julia --project=julia -e "using Pkg; Pkg.instantiate()"
```

Requirements: Julia 1.10+. Julia is **optional** — skip if not installed.

### JavaScript / TypeScript

```bash
cd js
npm install          # first time only
npm run dev          # dev server with HMR → http://localhost:5173
npm run build        # rebuild dist/
npx tsc --noEmit     # full type-check (Vite itself only transpiles)
```

Requirements: Node.js 18+.

---

## Running Tests

| Language | Command | From |
|---|---|---|
| Python | `pytest python/tests/` | repo root |
| Python (with coverage) | `pytest --cov=mathphys python/tests/` | repo root |
| C++ | `ctest --output-on-failure` | `cpp/build/` |
| Rust | `cargo test` | `rust/` |
| Julia | `julia --project=julia julia/tests/runtests.jl` | repo root |

**Test coverage:** All numerics tests use known analytic solutions (constant,
linear, quadratic, sinusoidal) to validate `integrate_trapezoid` and
`finite_difference`.

---

## Session-Start Hook

`.claude/hooks/session-start.sh` runs automatically on Claude Code remote
sessions (`CLAUDE_CODE_REMOTE=true`). It:

1. Installs Python deps via `pip install -e ".[dev]"`
2. Runs `cargo fetch && cargo build` (Rust)
3. Configures and builds C++ via CMake
4. Instantiates Julia environment (skipped if Julia not in PATH)

You do **not** need to manually build before working — the hook handles it.

---

## Physics Domain

The double pendulum consists of two point masses on massless rigid rods at a
fixed pivot. State vector: `[θ₁, ω₁, θ₂, ω₂]` (angles from downward vertical,
angular velocities).

**Default parameters:** m₁ = m₂ = 1 kg, L₁ = L₂ = 1 m, g = 9.81 m s⁻²

**Euler–Lagrange equations** (identical across all implementations):

```
Δ = θ₁ − θ₂
D = 2m₁ + m₂ − m₂·cos(2Δ)

θ̈₁ = [−g(2m₁+m₂)sinθ₁ − m₂g·sin(θ₁−2θ₂) − 2sinΔ·m₂(ω₂²L₂ + ω₁²L₁cosΔ)] / (L₁·D)
θ̈₂ = [2sinΔ·(ω₁²L₁(m₁+m₂) + g(m₁+m₂)cosθ₁ + ω₂²L₂m₂cosΔ)] / (L₂·D)
```

**Energy conservation** is used as a correctness metric:
- RK4 at dt = 1 ms: |ΔE/E₀| < 1×10⁻⁶ for 30 s chaotic run
- Python adaptive solver: |ΔE/E₀| < 1×10⁻⁸

**Canonical presets:**

| Preset | θ₁ | θ₂ | Behaviour |
|---|---|---|---|
| Near-linear | 10° | 10° | Quasi-periodic |
| Intermediate | 90° | 0° | Mixed regular/chaotic |
| Chaotic | 120° | −30° | Fully chaotic |

---

## Code Conventions

### Cross-language consistency

- **State order** is always `[θ₁, ω₁, θ₂, ω₂]` — never rearrange this.
- **Physics equations** must be identical across all languages; verify against
  the Python implementation as the reference (uses SciPy's validated solver).
- **Energy tracking** is mandatory in every implementation.
- **CSV output format** for CLI tools: `t, theta1, omega1, theta2, omega2, x2, y2, energy`.

### Python

- Source lives in `python/src/mathphys/` (not the repo root).
- Use `numpy` arrays; avoid Python lists for numeric state.
- Linting: `ruff check .` — fix before committing.
- Follow existing docstring style (NumPy/Google format).

### C++

- Standard: C++20. Use `std::span`, `std::array`, range-based for.
- Headers in `cpp/include/`, implementations in `cpp/src/`.
- No raw owning pointers; prefer `std::vector` / value types.
- Compile commands exported to `cpp/build/compile_commands.json` (useful for
  clangd).

### Rust

- Edition 2021. Immutable by default; add `mut` only when necessary.
- Use `nalgebra` vector types for linear algebra, not raw arrays.
- `#[allow(dead_code)]` is acceptable for library functions not yet called from
  binaries, but prefer cleaning up unused exports.
- All numeric helpers use iterators/windows — keep that style.

### Julia

- Use Unicode math symbols (`θ₁`, `ω₂`, `Δ`, etc.) where they improve
  readability — this is idiomatic Julia for physics code.
- Prefer in-place mutations (`update!` style) for hot simulation loops.
- Module is `MathPhys`; sub-module `DoublePendulum` re-exported at top level.

### TypeScript / JavaScript

- TypeScript strict mode (`tsconfig.json`). Run `npx tsc --noEmit` to verify.
- Canvas rendering in `main.ts`; physics only in `double_pendulum.ts`.
- No production dependencies — keep it that way (pure browser, Canvas API only).
- The `dist/` bundle is committed so users can open `js/dist/index.html` without
  a build step. Rebuild and commit `dist/` when changing `js/src/`.

---

## Common Workflows

### Adding a new physics quantity

1. Add computation to Python `double_pendulum.py` first (validate with tests).
2. Mirror the same logic in C++, Rust, Julia, TypeScript.
3. If it's a CSV column, add it to all CSV writers and update the column header
   comment in README.md.
4. Update tests in all languages.

### Changing the integrator

- Python: modify `solve_ivp` call in `DoublePendulum.simulate()`.
- C++/Rust/Julia/TS: the RK4 step is inlined — replace the 4-stage kernel.
- Verify energy drift improves or stays within spec after any change.

### Running the demo

```bash
# Python — saves double_pendulum.png
python python/examples/double_pendulum_demo.py

# C++ — writes case_*.csv, sensitivity_*.csv, mass_ratio_*.csv
./cpp/build/dp_sim_exec

# Rust — same CSV set as C++
cargo run --bin dp_sim --release --manifest-path rust/Cargo.toml

# Julia
julia --project=julia julia/examples/dp_sim.jl

# Browser (no build needed)
open js/dist/index.html
```

---

## What Not to Do

- Do **not** commit `cpp/build/` or `rust/target/` — they are in `.gitignore`.
- Do **not** add production npm dependencies to `js/` without a strong reason.
- Do **not** change the state vector order `[θ₁, ω₁, θ₂, ω₂]` — it is a
  cross-language contract.
- Do **not** skip energy validation when modifying integrators.
- Do **not** approximate `sin`/`cos` (e.g., small-angle) — exact nonlinear
  dynamics are the project's core requirement.
