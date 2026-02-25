#!/bin/bash
# Session-start hook for wClaudeCode — Mathematical Physics toolkit
# Supports: Python, Rust, C++, Julia (when available)
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
# Ensure the package is importable from the project source layout
echo "export PYTHONPATH=\"$PROJECT_DIR/python/src\"" >> "${CLAUDE_ENV_FILE:-/dev/null}"
echo "Python: OK"

# ── Rust ────────────────────────────────────────────────────────────────────
echo ""
echo "--- Fetching Rust dependencies ---"
cd "$PROJECT_DIR/rust"
cargo fetch --quiet
cargo build --quiet 2>&1 | tail -5
echo "Rust: OK"

# ── C++ ─────────────────────────────────────────────────────────────────────
echo ""
echo "--- Building C++ (CMake) ---"
BUILD_DIR="$PROJECT_DIR/cpp/build"
mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR/cpp" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  --log-level=WARNING -Wno-dev 2>&1 | grep -v "^--" || true
cmake --build "$BUILD_DIR" --parallel "$(nproc)" 2>&1 | tail -5
echo "C++: OK"

# ── Julia (optional) ────────────────────────────────────────────────────────
if command -v julia &>/dev/null; then
  echo ""
  echo "--- Instantiating Julia environment ---"
  julia --project="$PROJECT_DIR/julia" -e "using Pkg; Pkg.instantiate()" 2>&1 | tail -5
  echo "Julia: OK"
else
  echo ""
  echo "Julia not found in PATH — skipping (install Julia 1.10+ to enable)"
fi

echo ""
echo "=== Session Start Hook complete ==="
