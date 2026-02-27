# MathPhys — Mathematical Physics Simulations

> Physics-first educational repository: the same model implemented in five
> languages (Python, C++, Rust, Julia, TypeScript), organized by physical topic.

---

## Repository layout

```
.
├── src/mathphys/                     # Shared Python package (pip install -e .)
│   ├── double_pendulum.py
│   ├── ising_model.py
│   ├── cavity.py                     # ← EM cavity modes (new)
│   ├── waveguide.py                  # ← EM waveguide modes (new)
│   └── numerics.py
│
├── classical_mechanics/
│   └── double_pendulum/              # Exact nonlinear double pendulum
│       ├── python/examples/ tests/
│       ├── cpp/    include/ src/ tests/ examples/
│       ├── rust/   src/ src/bin/
│       ├── julia/  src/ tests/ examples/
│       └── js/     src/ dist/
│
├── statistical_physics/
│   └── ising_model_2d/               # 2D Ising model (Metropolis MC)
│       ├── python/examples/ tests/
│       ├── cpp/    include/ src/ tests/ examples/
│       ├── rust/   src/ src/bin/
│       ├── julia/  src/ tests/ examples/
│       └── js/     src/ dist/
│
└── electromagnetics/                 # ← NEW
    └── cavity_waveguide/             # EM cavity & waveguide analytical solutions
        ├── python/examples/ tests/
        ├── cpp/    include/ src/ tests/ examples/
        ├── rust/   src/ src/bin/
        ├── julia/  src/ tests/ examples/
        └── js/     src/ dist/          ← interactive field visualizer
```

---

## Topics

### 古典力学 / Classical Mechanics

#### Double Pendulum (`classical_mechanics/double_pendulum/`)

Exact nonlinear Lagrangian dynamics — no small-angle approximation.

| Symbol | Value | Description |
|---|---|---|
| m₁, m₂ | 1 kg | Bob masses |
| L₁, L₂ | 1 m | Rod lengths |
| g | 9.81 m s⁻² | Gravitational acceleration |

**Euler–Lagrange equations of motion**

$$\ddot{\theta}_1 = \frac{-g(2m_1+m_2)\sin\theta_1 - m_2 g\sin(\theta_1-2\theta_2) - 2\sin\Delta\cdot m_2(\dot\theta_2^2 L_2 + \dot\theta_1^2 L_1\cos\Delta)}{L_1\,D}$$

$$\ddot{\theta}_2 = \frac{2\sin\Delta\!\left[\dot\theta_1^2 L_1(m_1+m_2)+g(m_1+m_2)\cos\theta_1+\dot\theta_2^2 L_2 m_2\cos\Delta\right]}{L_2\,D}$$

where $\Delta = \theta_1 - \theta_2$, $D = 2m_1 + m_2 - m_2\cos 2\Delta$.

Energy conservation $E = T + V$ tracked as integrator quality metric (ΔE/E₀).

| Preset | θ₁ | θ₂ | Behaviour |
|---|---|---|---|
| Near-linear | 10° | 10° | Small oscillations, quasi-periodic |
| Intermediate | 90° | 0° | Mixed regular / chaotic |
| Chaotic | 120° | −30° | Sensitive to initial conditions |

---

### 統計物理学 / Statistical Physics

#### 2D Ising Model (`statistical_physics/ising_model_2d/`)

Ferromagnetic Ising model on an N×N square lattice.

**Hamiltonian:**

$$H = -J \sum_{\langle i,j\rangle} s_i s_j \qquad s_i \in \{-1,+1\},\quad J>0$$

**Metropolis-Hastings algorithm:** For each randomly selected spin, compute
$\Delta E = 2J\,s_i \sum_{\text{nn}} s_j$.  Accept flip if $\Delta E \le 0$ or
with probability $e^{-\Delta E / k_B T}$.

**Observables per site (averaged over MC sweeps after thermalisation):**

| Symbol | Expression |
|---|---|
| $\langle E \rangle$ | mean energy |
| $\langle\|M\|\rangle$ | order parameter (magnetisation) |
| $C_v$ | $\mathrm{Var}(E)\,/\,(T^2 N^2)$ — specific heat |
| $\chi$ | $\mathrm{Var}(\|M\|)\,/\,(T\,N^2)$ — susceptibility |

**Onsager's exact critical temperature** (J = k_B = 1):

$$T_c = \frac{2J}{k_B \ln(1+\sqrt{2})} \approx 2.2692$$

Both $C_v$ and $\chi$ diverge (for infinite system) at $T_c$.

---

### 電磁気学 / Electromagnetics

#### EM Cavity & Waveguide (`electromagnetics/cavity_waveguide/`)

Analytical solutions for PEC (Perfect Electric Conductor) cavities and waveguides.

**Supported structures:**

| Structure | Modes | Key formula |
|---|---|---|
| Rectangular cavity (a×b×d) | TE_mnp, TM_mnp | f = (c/2)√((m/a)²+(n/b)²+(p/d)²) |
| Cylindrical cavity (R, L) | TM_mnp (χ_mn/R), TE_mnp (χ'_mn/R) | f = (c/2π)√((χ/R)²+(pπ/L)²) |
| Spherical cavity (R) | TM_ln, TE_ln | f = c·χ_ln/(2πR), zeros via j_l(kR)=0 |
| Rectangular waveguide (a×b) | TE_mn, TM_mn | f_c = c·k_c/(2π), k_c=π√((m/a)²+(n/b)²) |
| Circular waveguide (R) | TE_mn (χ'_mn/R), TM_mn (χ_mn/R) | f_c = c·χ/2πR |

**Dispersion relation:**

$$\omega^2 = (\beta c)^2 + (k_c c)^2 \qquad \beta = \sqrt{({\omega}/{c})^2 - k_c^2}$$

Propagating: $f > f_c$ (β real); evanescent: $f < f_c$ (β imaginary).

**Time dependence in cavities (standing wave):**

$$\mathbf{E}(\mathbf{r},t) = \mathbf{E}_0(\mathbf{r})\cos(\omega t), \qquad \mathbf{H}(\mathbf{r},t) = \mathbf{H}_0(\mathbf{r})\sin(\omega t)$$

E and H are 90° out of phase in time — energy oscillates between electric and magnetic fields.

**Visualisations produced by `cavity_demo.py` and `waveguide_demo.py`:**

| Figure | Description |
|---|---|
| `cavity_freq_chart.png` | Resonant frequency bar chart for all 3 cavity types |
| `cavity_rect_te101.png` | TE₁₀₁ mode — 4 phase panels showing E↔H energy exchange |
| `cavity_rect_atlas.png` | 6 mode field patterns (xy cross-section heatmap + streamplot) |
| `cavity_cyl_tm010.png`  | Cylindrical TM₀₁₀ — ρ-z cross-section + 4 phase evolution panels |
| `cavity_sph_tm11.png`   | Spherical TM₁₁ — r-θ cross-section in Cartesian projection |
| `waveguide_dispersion.png` | ω vs β dispersion curves for 4 modes of each waveguide type |
| `waveguide_rect_modes.png` | Rectangular waveguide field patterns: TE₁₀, TE₂₀, TE₁₁, TM₁₁, … |
| `waveguide_rect_prop.png`  | TE₁₀ field evolution along z (4 cross-sections) |
| `waveguide_circ_modes.png` | Circular waveguide TE₁₁, TM₀₁, TE₂₁, TM₁₁ field patterns |

---

## Language summary

| Language | Mechanics / Statistics | EM Cavity & Waveguide | Output |
|---|---|---|---|
| **Python** | adaptive RK45 / NumPy MC | scipy Bessel, analytical | matplotlib PNG |
| **C++** | fixed-step RK4 / Metropolis | Bessel table, analytical | CSV + CTest |
| **Rust** | fixed-step RK4 / Metropolis | series Bessel, analytical | CSV + unit tests |
| **Julia** | fixed-step RK4 / Metropolis | SpecialFunctions.jl | CSV |
| **TypeScript** | fixed-step RK4 / Metropolis | series Bessel, analytical | live browser canvas |

---

## Interactive Browser Demos

No installation required — open the pre-built apps directly:

```bash
# Double pendulum
open classical_mechanics/double_pendulum/js/dist/index.html

# 2D Ising Model
open statistical_physics/ising_model_2d/js/dist/index.html

# EM Cavity & Waveguide Visualizer
open electromagnetics/cavity_waveguide/js/dist/index.html
```

**Double pendulum panels:** Pendulum · Phase portrait · Trajectory · Energy vs time

**Ising model controls:** Temperature slider · Low T / Critical / High T presets ·
Play/Pause/Reset · Sweeps-per-frame speed

**EM Cavity & Waveguide Visualizer:**
- Switch between Rectangular Cavity, Rectangular Waveguide, Circular Waveguide
- Select mode type (TE/TM) and indices (m, n, p) via sliders
- Presets: TE₁₀₁ cavity · TM₁₁₀ cavity · TE₁₀ waveguide · TE₁₁ circular
- Live E-field heatmap + arrow vectors, real-time phase animation (ωt)
- Dispersion chart (ω vs β) with operating frequency marker

---

## Setup & Usage

### Python (shared package)

```bash
pip install -e ".[dev]"                 # installs mathphys from src/

# Double pendulum
python classical_mechanics/double_pendulum/python/examples/double_pendulum_demo.py
python -m pytest classical_mechanics/double_pendulum/python/tests/

# 2D Ising Model
python statistical_physics/ising_model_2d/python/examples/ising_demo.py
python -m pytest statistical_physics/ising_model_2d/python/tests/

# All tests at once
python -m pytest
```

---

### C++ — Double Pendulum

```bash
cd classical_mechanics/double_pendulum/cpp
mkdir -p build && cd build
cmake ..
cmake --build .
ctest --output-on-failure     # run tests
./dp_sim_exec                 # generate 8 CSV scenarios
```

### C++ — 2D Ising Model

```bash
cd statistical_physics/ising_model_2d/cpp
mkdir -p build && cd build
cmake ..
cmake --build .
ctest --output-on-failure     # run tests
./ising_sim                   # temperature sweep → ising2d_sweep.csv
```

---

### Rust — Double Pendulum

```bash
cd classical_mechanics/double_pendulum/rust
cargo test
cargo run --bin dp_sim --release
```

### Rust — 2D Ising Model

```bash
cd statistical_physics/ising_model_2d/rust
cargo test
cargo run --bin ising_sim --release
```

---

### Julia — Double Pendulum

```bash
julia --project=classical_mechanics/double_pendulum/julia \
      classical_mechanics/double_pendulum/julia/tests/runtests.jl

julia --project=classical_mechanics/double_pendulum/julia \
      classical_mechanics/double_pendulum/julia/examples/dp_sim.jl
```

### Julia — 2D Ising Model

```bash
julia --project=statistical_physics/ising_model_2d/julia \
      statistical_physics/ising_model_2d/julia/tests/runtests.jl

julia --project=statistical_physics/ising_model_2d/julia \
      statistical_physics/ising_model_2d/julia/examples/ising_sim.jl
```

---

### Python — EM Cavity & Waveguide

```bash
# Run visualisations (generates PNG files)
python electromagnetics/cavity_waveguide/python/examples/cavity_demo.py
python electromagnetics/cavity_waveguide/python/examples/waveguide_demo.py

# Tests
python -m pytest electromagnetics/cavity_waveguide/python/tests/
```

---

### C++ — EM Cavity & Waveguide

```bash
cd electromagnetics/cavity_waveguide/cpp
mkdir -p build && cd build
cmake ..
cmake --build .
ctest --output-on-failure     # 28 tests
./em_sim                      # print resonant frequencies + β tables
```

---

### Rust — EM Cavity & Waveguide

```bash
cd electromagnetics/cavity_waveguide/rust
cargo test                    # 29 tests
cargo run --bin em_sim        # print resonant frequencies + dispersion
```

---

### Julia — EM Cavity & Waveguide

```bash
julia --project=electromagnetics/cavity_waveguide/julia \
      electromagnetics/cavity_waveguide/julia/tests/runtests.jl

julia --project=electromagnetics/cavity_waveguide/julia \
      electromagnetics/cavity_waveguide/julia/examples/em_sim.jl
```

---

### JavaScript / TypeScript

```bash
# Double pendulum dev server
cd classical_mechanics/double_pendulum/js
npm install && npm run dev          # → http://localhost:5173

# Ising model dev server
cd statistical_physics/ising_model_2d/js
npm install && npm run dev          # → http://localhost:5173

# EM Cavity & Waveguide visualizer dev server
cd electromagnetics/cavity_waveguide/js
npm install && npm run dev          # → http://localhost:5173
```

---

## Tests

| Language | Double Pendulum | Ising Model | EM Cavity & Waveguide |
|---|---|---|---|
| Python | `pytest classical_mechanics/.../tests/` | `pytest statistical_physics/.../tests/` | `pytest electromagnetics/.../tests/` |
| C++ | `ctest` in `cpp/build/` | `ctest` in `cpp/build/` | `ctest` in `cpp/build/` (28 tests) |
| Rust | `cargo test` | `cargo test` | `cargo test` (29 tests) |
| Julia | `julia ... runtests.jl` | `julia ... runtests.jl` | `julia ... runtests.jl` |

---

## Energy / accuracy metrics

**Double pendulum:** A correctly implemented RK4 integrator with dt = 1 ms keeps
relative energy drift $|\Delta E / E_0| < 10^{-6}$ for the 30-second chaotic scenario.
The Python adaptive solver maintains drift $< 10^{-8}$.

**Ising model:** Energy conservation is exact in the MC framework (Metropolis
satisfies detailed balance). Finite-size effects shift the apparent $T_c$ from the
Onsager value; the peak in $C_v$ sharpens as $N \to \infty$.
