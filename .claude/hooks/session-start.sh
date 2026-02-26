#!/bin/bash
# Session-start hook for wClaudeCode — Mathematical Physics toolkit
# Supports: Python, Rust, C++, Julia (when available)
# Topic structure: classical_mechanics/double_pendulum, statistical_physics/ising_model_2d, ...
set -euo pipefail

# Only run in Claude Code remote sessions
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
echo "=== MathPhys Session Start Hook ==="
echo "Project: $PROJECT_DIR"

# ── Python ──────────────────────────────────────────────────────────────────
echo ""
echo "--- Installing Python dependencies ---"
cd "$PROJECT_DIR"
pip install --quiet -e ".[dev]"
# Shared Python package lives in src/
echo "export PYTHONPATH=\"$PROJECT_DIR/src\"" >> "${CLAUDE_ENV_FILE:-/dev/null}"
echo "Python: OK"

# ── Rust ────────────────────────────────────────────────────────────────────
echo ""
echo "--- Fetching Rust dependencies ---"
for RUST_DIR in \
    "$PROJECT_DIR/classical_mechanics/double_pendulum/rust" \
    "$PROJECT_DIR/statistical_physics/ising_model_2d/rust"; do
  if [ -d "$RUST_DIR" ]; then
    echo "  -> $RUST_DIR"
    (cd "$RUST_DIR" && cargo fetch --quiet && cargo build --quiet 2>&1 | tail -3)
  fi
done
echo "Rust: OK"

# ── C++ ─────────────────────────────────────────────────────────────────────
echo ""
echo "--- Building C++ (CMake) ---"
for CPP_DIR in \
    "$PROJECT_DIR/classical_mechanics/double_pendulum/cpp" \
    "$PROJECT_DIR/statistical_physics/ising_model_2d/cpp"; do
  if [ -d "$CPP_DIR" ]; then
    echo "  -> $CPP_DIR"
    BUILD_DIR="$CPP_DIR/build"
    mkdir -p "$BUILD_DIR"
    cmake -S "$CPP_DIR" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      --log-level=WARNING -Wno-dev 2>&1 | grep -v "^--" || true
    cmake --build "$BUILD_DIR" --parallel "$(nproc)" 2>&1 | tail -3
  fi
done
echo "C++: OK"

# ── Julia (optional) ────────────────────────────────────────────────────────
if command -v julia &>/dev/null; then
  echo ""
  echo "--- Instantiating Julia environments ---"
  for JL_DIR in \
      "$PROJECT_DIR/classical_mechanics/double_pendulum/julia" \
      "$PROJECT_DIR/statistical_physics/ising_model_2d/julia"; do
    if [ -d "$JL_DIR" ]; then
      echo "  -> $JL_DIR"
      julia --project="$JL_DIR" -e "using Pkg; Pkg.instantiate()" 2>&1 | tail -3
    fi
  done
  echo "Julia: OK"
else
  echo ""
  echo "Julia not found in PATH — skipping (install Julia 1.10+ to enable)"
fi

echo ""
echo "=== Session Start Hook complete ==="
