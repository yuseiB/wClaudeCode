# CLAUDE.md — MathPhys Repository Guide

> Guidance for AI assistants working in this codebase.

## Project Scope

**MathPhys** is a multi-language educational toolkit for mathematical physics
simulations. The same physical model is coded in **five languages** (Python,
C++, Rust, Julia, TypeScript) to compare performance, idioms, and numerical
approaches.

**In scope:** Physics simulation topics listed in the roadmap below.
**Out of scope:** General tools (typing games, OCR formatters, text helpers),
machine-learning roadmaps, and any task unrelated to mathematical physics
simulation. If asked to do something outside this scope, politely decline and
ask the user to open a separate repository for that work.

---

## Repository Layout

```
wClaudeCode/
├── CLAUDE.md                          ← this file
├── ROADMAP.md                         ← planned topics and status
├── README.md
├── pyproject.toml                     ← Python project metadata + tool config
├── .gitignore
├── .claude/
│   ├── settings.json
│   └── hooks/session-start.sh        ← auto-setup for remote sessions
│
├── src/mathphys/                      ← shared Python package (pip install -e .)
│   ├── __init__.py
│   ├── double_pendulum.py
│   ├── ising_model.py
│   ├── cavity.py
│   ├── waveguide.py
│   ├── accelerator.py
│   └── numerics.py
│
├── classical_mechanics/
│   └── double_pendulum/               ← exact nonlinear Lagrangian dynamics
│       ├── python/examples/ tests/
│       ├── cpp/    include/ src/ tests/ examples/
│       ├── rust/   src/ src/bin/
│       ├── julia/  src/ tests/ examples/
│       └── js/     src/ dist/
│
├── statistical_physics/
│   └── ising_model_2d/                ← 2-D Ising model (Metropolis MC)
│       ├── python/examples/ tests/
│       ├── cpp/    include/ src/ tests/ examples/
│       ├── rust/   src/ src/bin/
│       ├── julia/  src/ tests/ examples/
│       └── js/     src/ dist/
│
├── electromagnetics/
│   └── cavity_waveguide/              ← PEC cavity & waveguide analytical modes
│       ├── python/examples/ tests/
│       ├── cpp/    include/ src/ tests/ examples/
│       ├── rust/   src/ src/bin/
│       ├── julia/  src/ tests/ examples/
│       └── js/     src/ dist/
│
└── accelerator_physics/
    └── single_particle/               ← FODO lattice & storage-ring tracking
        ├── python/examples/ tests/ api/   (C++/Rust/Julia not yet added)
        └── js/     src/
```

Each topic follows the same layout. When adding a new topic, copy the structure
of an existing one.

---

## Build Systems

### Python (shared package)

```bash
# Install package in editable mode (run from repo root)
pip install -e ".[dev]"

# The shared library lives in src/mathphys/
# All topic-specific code imports from mathphys
```

Tool config in `pyproject.toml`:
- **Linter:** `ruff` (line-length 100, rules E/F/W/I/N/UP)
- **Tests:** `pytest`, paths configured for all topic test directories
- **Build backend:** `setuptools>=68`

### C++ (per topic)

Each topic has its own `cpp/` subdirectory with `CMakeLists.txt`:

```bash
cd <topic>/<subtopic>/cpp
mkdir -p build && cd build
cmake ..
cmake --build .
ctest --output-on-failure
```

Requirements: CMake ≥ 3.20, GCC 12+ or Clang 15+ (C++20).

### Rust (per topic)

```bash
cd <topic>/<subtopic>/rust
cargo test
cargo run --bin <sim_name> --release
```

Requirements: Rust 1.75+ stable.

### Julia (per topic)

```bash
julia --project=<topic>/<subtopic>/julia \
      <topic>/<subtopic>/julia/tests/runtests.jl
```

Requirements: Julia 1.10+.

### JavaScript / TypeScript (per topic)

```bash
cd <topic>/<subtopic>/js
npm install
npm run dev          # dev server → http://localhost:5173
npm run build        # rebuild dist/
```

Requirements: Node.js 18+. The `dist/` bundle is committed so users can open
`js/dist/index.html` without a build step.

---

## Running Tests

### All Python tests (from repo root)

```bash
python -m pytest
```

### Per-topic tests

| Topic | Python | C++ | Rust |
|---|---|---|---|
| Classical Mechanics | `pytest classical_mechanics/double_pendulum/python/tests/` | `ctest` in `cpp/build/` | `cargo test` |
| Statistical Physics | `pytest statistical_physics/ising_model_2d/python/tests/` | `ctest` in `cpp/build/` | `cargo test` |
| Electromagnetics | `pytest electromagnetics/cavity_waveguide/python/tests/` | `ctest` in `cpp/build/` (28 tests) | `cargo test` (29 tests) |
| Accelerator Physics | `pytest accelerator_physics/single_particle/python/tests/` | (not yet) | (not yet) |

---

## Code Conventions

### Cross-language consistency

- The **shared Python library** in `src/mathphys/` is the reference implementation.
  Verify other languages agree with it.
- **CSV output** from CLI tools: column order must match the topic's README table.
- **Energy / accuracy tracking** is mandatory in every simulation implementation.
- Do **not** approximate physics (e.g., small-angle for double pendulum) — exact
  nonlinear dynamics are a core requirement.

### Python

- Source lives in `src/mathphys/` (the shared library) and topic-specific code
  in `<topic>/<subtopic>/python/`.
- Use `numpy` arrays; avoid Python lists for numeric state.
- Linting: `ruff check .` — fix before committing.
- Follow existing docstring style (NumPy/Google format).

### C++

- Standard: C++20. Headers in `include/`, implementations in `src/`.
- No raw owning pointers; prefer `std::vector` / value types.

### Rust

- Edition 2021. Immutable by default.
- Use `nalgebra` for linear algebra where appropriate.

### Julia

- Use Unicode math symbols (`θ₁`, `ω₂`, `Δ`, etc.) — idiomatic for physics.
- Prefer in-place mutations for hot simulation loops.

### TypeScript / JavaScript

- TypeScript strict mode. Run `npx tsc --noEmit` to verify.
- Physics logic in separate module from rendering/UI.
- No production dependencies unless absolutely necessary.

---

## Session-Start Hook

`.claude/hooks/session-start.sh` runs automatically on Claude Code remote
sessions (`CLAUDE_CODE_REMOTE=true`). It installs Python deps and optionally
builds C++ and Rust for the first topic encountered. You do not need to manually
build before working.

---

## Adding a New Physics Topic

1. Create the directory `<category>/<topic>/` following the layout above.
2. Implement in Python first (add a module to `src/mathphys/`), validate with tests.
3. Mirror the physics in C++, Rust, Julia, TypeScript.
4. Add the topic's test path to `[tool.pytest.ini_options] testpaths` in `pyproject.toml`.
5. Update `README.md` and `ROADMAP.md`.
6. Work on a single branch named `feature/<topic>` — do not mix topics in one branch.

---

## Branch Naming Convention

Use descriptive, topic-focused branch names:

- `feature/<topic-name>` for new physics topics
- `fix/<description>` for bug fixes
- `docs/<description>` for documentation updates

**Do not** create branches for out-of-scope work (tools, roadmaps, non-physics
utilities). Keep each branch focused on a single physics topic.

---

## What Not to Do

- Do **not** commit `build/`, `target/`, `__pycache__/`, `*.egg-info/`, `node_modules/`.
- Do **not** create branches for tasks unrelated to mathematical physics
  simulation (ML roadmaps, utility tools, text formatters, etc.).
- Do **not** mix multiple unrelated topics in a single branch or PR.
- Do **not** change the physics equations without verifying energy conservation
  or other accuracy metrics.
- Do **not** skip tests when adding new physics modules.
- Do **not** add npm production dependencies without a strong reason.
