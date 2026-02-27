# MathPhys — Double Pendulum Simulation

> Exact nonlinear double-pendulum dynamics implemented in five languages,
> with a real-time interactive browser visualization.
>
> 5つの言語で実装された正確な非線形二重振り子動力学、リアルタイム対話型ブラウザ可視化付き。

| Language | Integrator | Output |
|---|---|---|
| **Python** | Adaptive RK45 (SciPy DOP853, rtol=1e-9) | matplotlib figure, CSV |
| **C++** | Fixed-step RK4 | CSV files |
| **Rust** | Fixed-step RK4 | CSV files |
| **Julia** | Fixed-step RK4 | CSV files |
| **JavaScript / TypeScript** | Fixed-step RK4 | Live browser canvas |

---

## Interactive Browser Demo

The fastest way to explore the system — no installation required:

```bash
open js/dist/index.html
```

Or run the development server with hot reload:

```bash
cd js && npm install && npm run dev
# → http://localhost:5173
```

The app renders four live panels:

| Panel | Description |
|---|---|
| **Pendulum** | Swinging rods with trailing bob-2 path |
| **Phase portrait** | θ₂ vs ω₂, accumulates indefinitely |
| **Trajectory** | Bob-2 Cartesian path x₂, y₂ |
| **Energy** | Kinetic T, potential V and total E vs time |

Controls: preset buttons · θ₁/θ₂/ω₁/ω₂ sliders · Play / Pause / Reset · speed ×½ ×1 ×2 ×5 · live ΔE/E₀ readout.

![Python demo output](double_pendulum.png)

---

## Physics

Two point masses on massless rigid rods at a fixed pivot. Angles are measured from the downward vertical.

**Default parameters**

| Symbol | Value | Description |
|---|---|---|
| m₁, m₂ | 1 kg | Bob masses |
| L₁, L₂ | 1 m | Rod lengths |
| g | 9.81 m s⁻² | Gravitational acceleration |

**Euler–Lagrange equations of motion**

$$\ddot{\theta}_1 = \frac{-g(2m_1+m_2)\sin\theta_1 - m_2 g\sin(\theta_1-2\theta_2) - 2\sin\Delta\cdot m_2(\dot\theta_2^2 L_2 + \dot\theta_1^2 L_1\cos\Delta)}{L_1\,D}$$

$$\ddot{\theta}_2 = \frac{2\sin\Delta\!\left[\dot\theta_1^2 L_1(m_1+m_2)+g(m_1+m_2)\cos\theta_1+\dot\theta_2^2 L_2 m_2\cos\Delta\right]}{L_2\,D}$$

where $\Delta = \theta_1 - \theta_2$ and $D = 2m_1 + m_2 - m_2\cos 2\Delta$.

Total mechanical energy $E = T + V$ is conserved; all implementations track $|\Delta E / E_0|$ as an integrator quality metric.

**Canonical initial conditions**

| Preset | θ₁ | θ₂ | Behaviour |
|---|---|---|---|
| Near-linear | 10° | 10° | Small oscillations, quasi-periodic |
| Intermediate | 90° | 0° | Mixed regular / chaotic |
| Chaotic | 120° | −30° | Sensitive to initial conditions, fully chaotic |

---

## Repository Layout

```
wClaudeCode/
├── pyproject.toml              # Python project metadata + tool config
├── CLAUDE.md                   # Guide for AI assistants
├── double_pendulum.png         # Demo output (committed artifact)
├── .claude/
│   └── hooks/session-start.sh  # Auto-build script for remote sessions
├── python/
│   ├── src/mathphys/
│   │   ├── double_pendulum.py  # DoublePendulum class, adaptive RK45
│   │   └── numerics.py         # integrate_trapezoid, finite_difference
│   ├── tests/test_numerics.py
│   └── examples/double_pendulum_demo.py
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/                # double_pendulum.hpp, numerics.hpp
│   ├── src/                    # double_pendulum.cpp, numerics.cpp
│   ├── tests/test_numerics.cpp
│   └── examples/dp_sim.cpp
├── rust/
│   ├── Cargo.toml
│   └── src/
│       ├── double_pendulum.rs
│       ├── numerics.rs
│       └── bin/dp_sim.rs
├── julia/
│   ├── Project.toml
│   ├── src/                    # MathPhys.jl, DoublePendulum.jl
│   ├── tests/runtests.jl
│   └── examples/dp_sim.jl
└── js/
    ├── src/
    │   ├── double_pendulum.ts
    │   └── main.ts             # Canvas UI, 4 panels
    ├── index.html
    └── dist/                   # Pre-built bundle (open directly)
```

---

## Setup & Usage

### Python

**Requirements:** Python 3.11+

```bash
pip install -e ".[dev]"                           # install with dev tools

python python/examples/double_pendulum_demo.py    # saves double_pendulum.png
pytest python/tests/                              # run tests
pytest --cov=mathphys python/tests/              # with coverage
ruff check .                                      # lint
```

The demo uses SciPy's `solve_ivp` with DOP853 (rtol=1e-9, atol=1e-11) and
plots θ₁/θ₂, ω₁/ω₂, phase portraits, and energy drift on a single figure.

### C++

**Requirements:** CMake ≥ 3.20, GCC 12+ or Clang 15+

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build --parallel

cd cpp/build
ctest --output-on-failure    # run tests
./dp_sim_exec                # generate 8 CSV scenarios
```

Each scenario writes a CSV with columns `t, theta1, omega1, theta2, omega2, x2, y2, energy`.

### Rust

**Requirements:** Rust 1.75+ stable

```bash
cd rust
cargo test                          # unit + integration tests
cargo run --bin dp_sim --release    # generate 8 CSV scenarios
```

Uses [nalgebra](https://nalgebra.org/) for vector types. Produces the same 8 CSV scenarios as C++.

### Julia

**Requirements:** Julia 1.10+ *(optional — rest of the project works without it)*

```bash
julia --project=julia -e "using Pkg; Pkg.instantiate()"  # first time

julia --project=julia julia/tests/runtests.jl             # run tests
julia --project=julia julia/examples/dp_sim.jl            # generate CSVs
```

### JavaScript / TypeScript

**Requirements:** Node.js 18+ *(only for dev server — `dist/` needs no build step)*

```bash
cd js
npm install           # first time only
npm run dev           # dev server with HMR → localhost:5173
npm run build         # rebuild dist/
npx tsc --noEmit      # full type-check
```

---

## Tests

| Language | Framework | Command |
|---|---|---|
| Python | pytest | `pytest python/tests/` |
| C++ | CTest | `ctest` (run from `cpp/build/`) |
| Rust | built-in + approx | `cargo test` (run from `rust/`) |
| Julia | @testset | `julia --project=julia julia/tests/runtests.jl` |

All numerics tests validate `integrate_trapezoid` and `finite_difference`
against known analytic solutions (constant, linear, quadratic, sinusoidal).

---

## Energy Conservation

A correct RK4 integrator at dt = 1 ms keeps $|\Delta E / E_0| < 10^{-6}$
for the 30-second chaotic scenario. The Python adaptive solver maintains
drift below $10^{-8}$.

The browser app displays the live drift percentage next to the simulation
clock for interactive verification.
