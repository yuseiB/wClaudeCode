# MathPhys — Project Roadmap

> A multi-language mathematical physics simulation toolkit.
> Same model, five languages: Python · C++ · Rust · Julia · TypeScript

---

## ✅ Completed Topics

### Classical Mechanics

- [x] **Double Pendulum** (`classical_mechanics/double_pendulum/`)
  - Exact nonlinear Lagrangian dynamics (no small-angle approximation)
  - Python (adaptive RK45), C++ / Rust / Julia / TypeScript (RK4)
  - Interactive browser demo with phase portrait and energy tracking

### Statistical Physics

- [x] **2D Ising Model** (`statistical_physics/ising_model_2d/`)
  - Metropolis-Hastings Monte Carlo on an N×N square lattice
  - Observables: energy, magnetisation, specific heat (Cv), susceptibility (χ)
  - Interactive browser demo with temperature slider

### Electromagnetics

- [x] **EM Cavity & Waveguide** (`electromagnetics/cavity_waveguide/`)
  - Analytical modes: rectangular cavity, cylindrical cavity, spherical cavity
  - Rectangular and circular waveguide dispersion
  - Interactive browser visualiser with live E-field animation

### Accelerator Physics

- [x] **Single-Particle Dynamics** (`accelerator_physics/single_particle/`)
  - FODO lattice Courant–Snyder (Twiss) map
  - Storage-ring single-particle tracking (chromaticity, thin sextupole, aperture loss)
  - FastAPI backend + TypeScript browser client

---

## 🔜 Planned Topics

### Accelerator Physics — Collective Effects

- [ ] **Storage-Ring Monte Carlo** (multi-particle beam sampling)
  - Start from `accelerator_physics/single_particle/` Python package
  - Add `src/mathphys/storage_ring.py` (multi-particle RingParams + tracking)
  - Output: emittance evolution, beam loss map CSV
  - Branch: `feature/accelerator-storage-ring-mc`

- [ ] **Collective Instabilities**
  - Rigid-bunch transverse instability (coherent wake kick model)
  - Sliced head-tail instability (chromaticity + momentum spread)
  - Add `src/mathphys/collective.py`
  - Branch: `feature/accelerator-collective-instability`

### Signal Processing

- [ ] **Time-Frequency Analysis** (`signal_processing/time_frequency_analysis/`)
  - STFT / spectrogram, Wigner-Ville distribution, Cohen class
  - Instantaneous frequency, ambiguity function
  - Jupyter notebooks (bilingual EN/JA) + Python examples
  - Branch: `feature/signal-processing-time-frequency`

### Quantum Mechanics (future)

- [ ] **Harmonic Oscillator** — analytic eigenstates + split-operator time evolution
- [ ] **Hydrogen Atom** — spherical harmonics + radial Schrödinger equation

### Fluid Dynamics (future)

- [ ] **Lorenz Attractor** — chaotic ODE with strange attractor, Lyapunov exponent

---

## Notes

- Each topic gets its own `feature/<topic>` branch and PR.
- Multi-language implementations are added in the same PR as the Python one.
- Do not mix unrelated topics in a single branch.
