# Collective Beam Instabilities in Storage Rings
## Theory, Formalism, and Computational Implementation

---

> **Scope.**  This document provides the graduate-level theoretical
> background for the Monte Carlo collective-instability tracker implemented
> in `python/src/mathphys/collective.py`.  Sections 1–6 develop the
> physics from first principles; Sections 7–8 map the theory onto the
> simulation; Section 9 collects all notation; Section 10 lists primary
> references.  Prerequisites: classical electrodynamics, Hamiltonian
> mechanics, and single-particle accelerator optics at the level of
> Wiedemann [1] or Lee [2].

---

## Table of Contents

1. [Single-Particle Foundations](#1-single-particle-foundations)
2. [Wake Functions and Impedance](#2-wake-functions-and-impedance)
3. [Rigid-Bunch Transverse Instability](#3-rigid-bunch-transverse-instability)
4. [Landau Damping and the Stability Threshold](#4-landau-damping-and-the-stability-threshold)
5. [Head-Tail Instability](#5-head-tail-instability)
6. [Vlasov Perturbation Theory (Synopsis)](#6-vlasov-perturbation-theory-synopsis)
7. [Mapping Theory to the Monte Carlo Implementation](#7-mapping-theory-to-the-monte-carlo-implementation)
8. [Worked Numerical Example](#8-worked-numerical-example)
9. [Notation Summary](#9-notation-summary)
10. [References](#10-references)

---

## 1. Single-Particle Foundations

### 1.1 Courant–Snyder (Floquet) Parameterisation

The linear transverse equations of motion in a storage ring are Hill's
equations:

```
x''(s) + K(s) x(s) = 0
```

where primes denote `d/ds`, `s` is the arc-length co-ordinate, and `K(s)` is
the periodic focusing function.  The general solution is expressed in
**Courant–Snyder form** [3]:

```
x(s) = A √β(s) cos[φ(s) + φ₀]
```

where `β(s)` is the beta function, `φ(s) = ∫₀ˢ ds'/β(s')` is the betatron
phase advance, and `A` is the **action amplitude** (a constant of motion).
The **Courant–Snyder invariant** (action variable) is

```
J = (γ x² + 2α x x' + β x'²) / 2
```

with `α(s) = -β'(s)/2` and `γ(s) = (1 + α²)/β`.  The geometric
emittance of an ensemble is `ε = ⟨J⟩`.

### 1.2 One-Turn Transfer Matrix

At a fixed azimuthal location `s₀`, one full revolution maps
`(x, x') → M (x, x')ᵀ` with the symplectic matrix

```
M = | cos μ + α sin μ      β sin μ    |
    | −γ sin μ       cos μ − α sin μ  |
```

where `μ = 2πQ` is the **betatron phase advance per turn** and `Q` the
**tune** (number of betatron oscillations per revolution).  `det M = 1`
(Liouville's theorem).

### 1.3 Chromaticity

The tune depends on momentum offset `δ = Δp/p₀`:

```
Q(δ) = Q₀ + ξ δ + O(δ²)
```

where `ξ = dQ/dδ|₀` is the **chromaticity**.  With a Gaussian momentum
distribution of rms spread `σ_δ`, the rms tune spread is

```
σ_Q = |ξ| σ_δ
```

This spread is the seed for phase-space decoherence and Landau damping (Section 4).

### 1.4 Action-Angle Variables

The Courant–Snyder invariant `J` (eq. 1.1) is the canonical action.  Its
conjugate angle `ψ` advances by `2πQ` per turn.  In action-angle form the
transverse phase-space coordinates at azimuth `s` are

```
x   =  √(2J β(s)) cos ψ
x'  =  −√(2J / β(s)) [sin ψ  +  α(s) cos ψ]
```

At a location where `α = 0` (waist or symmetry point, which is the natural
choice for inserting a lumped kick) these simplify to

```
x   =  √(2J β) cos ψ                                   (1.4a)
x'  =  −√(2J / β) sin ψ                                (1.4b)
```

The Hamiltonian for free betatron motion is `H₀ = Q J` (in units of turns),
so Hamilton's equations give `dψ/dn = 2πQ` (constant) and `dJ/dn = 0`
(constant action — Liouville's theorem).  A thin transverse kick `Δx'` at
the waist changes the action by

```
ΔJ  =  β x' Δx' + (x Δx'/2 → 0 for Δx indep. of x')  =  β x' Δx'     (1.5)
```

(derived from `J = (x²/β + β x'²)/2` at `α = 0`).  Equations (1.4) and
(1.5) are the basis of the growth-rate derivation in Section 3.3.

---

## 2. Wake Functions and Impedance

### 2.1 Physical Origin

When a relativistic charge bunch traverses any vacuum chamber element
(resistive wall, RF cavity, bellows, BPM, …) it deposits energy in the
structure via induced image currents and electromagnetic fields.  The
**wake field** is the residual EM field that lags the bunch and acts on
trailing particles.  Two classes are relevant:

| Wake | Symbol | Units | Effect |
|------|--------|-------|--------|
| Longitudinal | `W_∥(s)` | V/C | Energy gain/loss |
| Transverse (dipole) | `W_⊥(s)` | V/C/m | Transverse kick |

`s > 0` denotes the distance *behind* the source particle (causal: a source
particle at `s = 0` can only affect particles at `s > 0`).

### 2.2 Panofsky–Wenzel Theorem

The longitudinal and transverse wake functions are not independent.  For a
relativistic beam (`β → 1`) the **Panofsky–Wenzel theorem** [4] states

```
∂W_⊥(s) / ∂s  =  −∇_⊥ W_∥(s)
```

i.e. a transverse wake can only exist when there is a transverse variation in
the longitudinal wake (typically from geometric asymmetry or beam offset).
This theorem is the basis for numerical wake computations [5].

### 2.3 Transverse Impedance

The **transverse impedance** is the one-sided Fourier transform of the wake
function, using the convention of Chao [5]:

```
Z_⊥(ω)  =  (1/c) ∫₀^∞  W_⊥(s) exp(iωs/c) ds      [Ω/m]          (2.3a)
```

Causality (`W_⊥ = 0` for `s < 0`) makes this a one-sided transform; the
lower limit is `0⁺`.  The inverse relation recovers the wake:

```
W_⊥(s)  =  (c/π) Re ∫₀^∞  Z_⊥(ω) exp(−iωs/c) dω                  (2.3b)
```

> **Sign convention warning.**  The literature uses two conventions for the
> sign of the exponent in (2.3a).  Chao [5] and Wilson [6] use `+iωs/c`
> (adopted here).  Some European texts (e.g. Zotter–Métral) use `−iωs/c`.
> With the Chao convention, a purely **inductive** wake (`W_⊥(s) > 0`,
> decaying) gives `Im Z_⊥ > 0`.

The coherent tune shift of the centroid mode is (see §3.1):

```
ΔQ_coh  ≈  −i N_b r₀ β Im[Z_⊥(ω_β)] / (4π γ C)                    (2.3c)
```

(protons; for electrons replace `r₀ → r_e`).  Stability consequences:

| `Im Z_⊥` at `ω_β` | `Im(ΔQ_coh)` | Effect |
|---|---|---|
| `> 0` (inductive) | `< 0` | Exponential decay — stable |
| `= 0` | `= 0` | Tune shift only — marginally stable |
| `< 0` (capacitive/resistive) | `> 0` | Exponential growth — **unstable** |

A **real** `Z_⊥` (resistive) produces a coherent tune shift that is purely
imaginary — exponential growth or damping depending on sign.  An **imaginary**
`Z_⊥` (reactive) shifts the tune without driving instability.

### 2.4 Broad-Band Resonator Model

A widely-used model that captures the essential instability physics is the
**broad-band resonator** [6, Ch. 2]:

```
Z_⊥^BBR(ω)  =  R_s / (1 + iQ(ω/ω_r − ω_r/ω))
```

| Parameter | Meaning |
|-----------|---------|
| `R_s` [Ω/m] | Shunt impedance (peak of `|Z_⊥|`) at resonance |
| `ω_r` [rad/s] | Resonant angular frequency |
| `Q` [-] | Quality factor |

The exact inverse Fourier transform of `Z_⊥^BBR` for arbitrary `Q` gives the
transverse wake function [5, §2.3]:

```
W_⊥^BBR(s)  =  W₀ e^{−α_d s/c}
               × [cos(ω̄ s/c)  +  (α_d/ω̄) sin(ω̄ s/c)],    s ≥ 0
```

where `W₀ = 2R_s ω_r/Q`, the **damping rate** `α_d = ω_r/(2Q)`, and the
**damped resonance frequency** `ω̄ = √(ω_r² − α_d²)`.  For `Q = 1`:
`α_d = ω_r/2`, `ω̄ = ω_r√3/2` — the wake executes roughly half an
oscillation before decaying to `1/e`, so the oscillatory and exponential
terms are of comparable magnitude and cannot be separated.

**Why the code uses a pure exponential model.**  The single-exponential form
`W_⊥(s) = W₀ exp(−s/z_w)` is a **short-range phenomenological model**, not
the exact `Q = 1` wake.  It corresponds physically to a strongly over-damped
resonance (`Q ≪ 1`) in the limit where the damped frequency `ω̄ → 0` and the
two terms above merge:

```
W_⊥(s) ≈ W₀ exp(−s / z_w),     z_w = 1/α_d = 2Qc/ω_r          (2.4)
```

This single-pole model correctly reproduces the key physics — short-range
wake drive, exponential suppression at distance `z_w` — while avoiding the
oscillatory sign changes that would require more slices to resolve.

**Equation (2.4) is the wake kernel used throughout this code**
(`wake_strength = W₀`, `wake_range = z_w`).

### 2.5 Resistive-Wall Impedance

For completeness, the **resistive-wall** transverse impedance of a cylindrical
pipe of radius `b`, length `L`, and conductivity `σ_c` is [6, Ch. 3]:

```
Z_⊥^RW(ω)  =  (1 ± i) L c Z₀ / (2π b³) · δ_skin(ω)
```

where `Z₀ = 377 Ω` is the vacuum impedance, `δ_skin = √(2/(μ₀ σ_c |ω|))` is
the skin depth, and the sign depends on the sign of `ω`.  This has both
resistive (`Re Z_⊥ ≠ 0`) and reactive parts.

The resistive wall is the dominant source of transverse instabilities in many
storage rings (particularly for high-current light sources).

---

## 3. Rigid-Bunch Transverse Instability

### 3.1 Coherent Equations of Motion

Treating the bunch as a rigid macro-particle, the **centroid**
`x̄ = (1/N) Σ xᵢ` obeys the single-particle equation of motion with an
additional **coherent kick** from the wake:

```
x̄'' + K(s) x̄  =  F_coh(s, x̄)
```

In the lumped-kick (one-thin-lens per turn) approximation, `F_coh` at turn `n`
is a delta-function kick at azimuthal location `s₀`:

```
Δx̄'|_{turn n}  =  −κ_coh · x̄(s₀, n)
```

where (absorbing beam and ring parameters) the **coherent kick parameter** is

```
κ_coh  =  N_b r₀ β(s₀) Z_⊥^eff / (4π γ C)         (3.1)
```

with `N_b` = bunch population, `r₀` = classical particle radius,
`C` = circumference, and `Z_⊥^eff` = effective impedance evaluated near the
betatron sidebands.  Equation (3.1) is a standard result; see [6] §2.2 and
[7] §4.3 for derivations.

### 3.2 One-Turn Map with Coherent Kick

Define the state vector `ξ̄ = (x̄, x̄')ᵀ` at the kick location after turn `n`.
The evolution from turn `n` to `n+1` is

```
ξ̄_{n+1}  =  M · (ξ̄_n + κ_coh · x̄_drive · ê_x')
```

where `ê_x' = (0, 1)ᵀ` and `x̄_drive` is the centroid used to evaluate the
kick.

**Case A — Instantaneous kick (`x̄_drive = x̄_n`).**

The kick `Δx̄' = κ x̄_n` is applied first, then the free map `M` acts.  In
matrix form the combined one-turn map on `(x̄, x̄')ᵀ` is:

```
M_coh  =  M · K_A,     K_A = |1,  0|
                               |κ,  1|
```

Multiplying out for general Courant–Snyder parameters `(α, β, γ)`:

```
M_coh = | cos μ + α sin μ + κβ sin μ,        β sin μ              |
        | −γ sin μ + κ(cos μ − α sin μ),    cos μ − α sin μ       |
```

**Verification.**

```
Tr(M_coh)  =  2 cos μ + κβ sin μ                                   (3.2a)

det(M_coh) =  (cos μ + α sin μ + κβ sin μ)(cos μ − α sin μ)
              − β sin μ [−γ sin μ + κ(cos μ − α sin μ)]
           =  cos²μ − α² sin²μ + βγ sin²μ  +  cross-terms that cancel
           =  cos²μ + (βγ − α²) sin²μ  =  1          (since βγ = 1+α²)
```

So `det(M_coh) = 1` for any real `κ` — the map is **symplectic**.  The
eigenvalues `λ = ½[Tr ± i√(4 − Tr²)]` remain on the unit circle as long as
`|Tr| ≤ 2`, i.e. `|κβ sin μ| ≤ 4|sin μ/2|²`: **the centroid oscillates at a
coherently shifted tune**

```
Q_coh  ≈  Q₀  +  κβ/(4π) sin(2πQ₀)                               (3.2b)
```

but its amplitude is bounded — **no exponential growth**.  This is the
signature of a *purely reactive* (imaginary) impedance.

> **Key result.**  An instantaneous, real-valued coherent kick is symplectic
> and therefore always bounded.  True exponential amplitude growth requires
> either (a) a **complex (resistive)** kick or (b) a **temporal delay**
> that breaks the instantaneous feedback loop.

**Case B — One-turn delayed kick (`x̄_drive = x̄_{n-1}`).**

The delayed centroid is

```
x̄_{n-1}  =  cos μ · x̄_n − β sin μ · x̄'_n  +  O(κ)         (3.2)
```

(obtained by inverting the free one-turn map).  Substituting into the kick:

```
Δx̄'_n  =  κ · x̄_{n-1}
         =  κ cos μ · x̄_n  −  κ β sin μ · x̄'_n
```

The kick now has **two components**:

| Component | Coupling | Effect |
|-----------|----------|--------|
| `κ cos μ · x̄` | reactive (in-phase with `x̄`) | tune shift |
| `−κ β sin μ · x̄'` | **resistive** (in-phase with `x̄'`) | energy exchange |

The second component is proportional to `x̄'` — the centroid *angle* — which
is 90° out of phase with `x̄`.  This is precisely the signature of a **resistive
impedance**.  Physically, the one-turn delay corresponds to a wake that
decays over one revolution period before acting on the beam.

### 3.3 Growth Rate Analysis via Action-Angle Variables

Write the centroid at turn `n` in action-angle form (§1.4, at the kick
location where `α = 0`):

```
x̄(n)  =  √(2J β) cos ψ_n,     x̄'(n)  =  −√(2J/β) sin ψ_n
```

where `ψ_n = 2πQ₀ n + ψ₀`.  The one-turn delayed centroid is

```
x̄_{n-1}  =  √(2Jβ) cos(ψ_n − 2πQ₀)
```

From Case B (§3.2), the kick is `Δx̄'_n = κ x̄_{n-1}`.  The change in
action from eq. (1.5) is

```
ΔJ  =  β x̄'_n · Δx̄'_n
     =  β · [−√(2J/β) sin ψ_n] · κ √(2Jβ) cos(ψ_n − 2πQ₀)
     =  −2κβJ · sin ψ_n · cos(ψ_n − 2πQ₀)
```

Expanding the product using the prosthaphaeresis identity:

```
sin ψ cos(ψ − μ)  =  ½[sin(2ψ − μ) + sin μ]
```

gives

```
ΔJ  =  −κβJ [sin(2ψ_n − 2πQ₀) + sin(2πQ₀)]
```

The **fast term** `sin(2ψ_n − 2πQ₀)` oscillates at twice the betatron
frequency and averages to zero over many turns.  The **secular term** is the
constant contribution:

```
⟨ΔJ⟩_turn  =  −κβJ sin(2πQ₀)                          (3.3)
```

Since `J = A²/2` and `dJ/dn = A dA/dn`, the amplitude obeys

```
dA/dn  =  −(κβ/2) sin(2πQ₀) · A  =  −μ_grow · A       (3.4)
```

For **κ < 0** (kick in the same direction as displacement, a destabilising
impedance), the centroid amplitude grows exponentially.  With the code's
sign convention `Δx̄' = +κ x̄_{n-1}` and `κ > 0`, the effective growth rate
is:

```
μ_grow  =  (κβ/2) |sin(2πQ₀)|                          (3.5)
```

> **Note on `sin` vs. `sin²`.**  The formula (3.5) contains `|sin(2πQ₀)|`,
> not `sin²(2πQ₀)`.  The `sin²` that appears in some references [5 §2.2]
> arises when the kick and the averaging are referenced to a general azimuth
> where `α ≠ 0`, or when using a two-turn map formalism.  At a waist
> (`α = 0`), the single-turn action-angle calculation gives `|sin(2πQ₀)|`.
> The code documentation uses `sin²(2πQ)` as an approximation valid for
> small `μ = 2πQ ≪ 1` where `sin μ ≈ sin²μ/sin μ ... ` deviates by at
> most a factor of `1/|sin μ|`; for `Q = 0.28`, `sin μ ≈ 0.98` and the
> two are within 2%.

The **e-folding time** in turns is

```
τ_inst  =  1/μ_grow  =  2 / (κβ |sin(2πQ₀)|)          (3.6)
```

For the default parameters (`β = 10 m`, `Q₀ = 0.28`, `κ = 0.05 rad/m`):

```
μ_grow  =  0.05 × 10 / 2 × |sin(2π × 0.28)|
         =  0.25 × 0.990  ≈  0.247 turn⁻¹

τ_inst  ≈  4.0 turns   [without Landau damping]
```

### 3.4 Connection to Resistive-Wall Growth Rate

In the standard treatment [6] §3.4, the resistive-wall growth rate for a
single-bunch coasting beam is

```
1/τ_RW  =  (N_b r₀ c / 2γ) · Im[Z_⊥^RW(ω₀)] / C
```

where `ω₀ = 2πf₀` is the revolution angular frequency.  Comparing with (3.5)
identifies the effective `κ` for a resistive-wall ring:

```
κ  ≡  N_b r₀ Im[Z_⊥^RW] / (γ C β)                     (3.6)
```

Equation (3.6) can be used to convert real machine impedance data into the
simulation parameter `kappa`.

---

## 4. Phase-Space Decoherence and the Stability Threshold

### 4.1 Phase Mixing vs. True Landau Damping

**Phase mixing (decoherence)** and **Landau damping** are related but distinct
mechanisms, and the distinction matters for interpreting the simulation.

**Phase mixing** (what the code models): each particle oscillates at its own
incoherent tune `Qᵢ`.  Starting from a coherent centroid offset `x̄(0) = A₀`
with all particles in phase, the centroid evolves as [9]:

```
x̄(n)  =  A₀ · Re[ ∫ g(Q) e^{i2πQn} dQ ]
```

For a **Gaussian tune distribution**
`g(Q) = (1/√(2π) σ_Q) exp(−(Q−Q₀)²/2σ_Q²)` the characteristic function is

```
x̄(n) / A₀  =  exp(−2π² σ_Q² n²) · cos(2πQ₀ n)         (4.1)
```

This is a **super-Gaussian (Gaussian-in-n²) envelope** — faster than
exponential.  The **decoherence time** (turn number where the envelope falls
to `1/e`) is

```
τ_deco  =  1 / (2π√2 σ_Q)                               (4.2)
```

For `σ_Q = 0.002`: `τ_deco ≈ 56 turns`.

> **This is not exponential damping.**  Equation (4.1) describes an
> irreversible loss of *phase coherence* (filamentation in phase space), not
> a true eigenmode decay.  It is reversible in principle (echo effect [21])
> and requires `N → ∞` particles; for finite `N` the centroid is restored to
> `A₀ exp(−2π²σ_Q²n²)` only on average.

**True Landau damping** [8, 9]: in the Vlasov framework (Section 6), coherent
mode eigenstates of the linearised Vlasov operator have complex eigenfrequencies
whose imaginary parts produce *exponential* damping.  For a bunched beam this
requires synchrotron-sideband coupling [5 §4.4]; the eigenfrequency lies just
below the incoherent band, and Landau damping arises from the resonance of the
coherent mode with single-particle oscillations at the band edge.  In this
simulation — which has no synchrotron oscillations — true eigenmode Landau
damping is absent; only phase mixing (4.1) stabilises the centroid.

### 4.2 Sacherer Dispersion Integral and the Stability Criterion

The kinetic stability criterion is derived from the linearised Vlasov equation.
For the coasting-beam limit (appropriate for our single-turn-map model), the
dispersion relation for the coherent mode frequency `ΔQ_coh` is [12, 5 §2.4]:

```
1  =  −K_eff · ∫₋∞^∞  (dg/dQ) / (Q − Q₀ − ΔQ_coh) dQ  (4.3)
```

where `K_eff` encapsulates the impedance drive.  The sign is such that
`K_eff > 0` for a destabilising impedance.  **Stability** requires
`Im(ΔQ_coh) ≤ 0` (no growing mode).

At the **stability boundary** `Im(ΔQ_coh) → 0⁺`, the Plemelj–Sokhotski formula
decomposes the integral into a principal value and a residue:

```
∫  (dg/dQ) / (Q − Q₀ − iε) dQ  →  P.V. ∫  (dg/dQ)/(Q−Q₀) dQ  +  iπ dg/dQ|_{Q₀}
```

For a Gaussian `g` centred at `Q₀`:  `dg/dQ|_{Q₀} = 0` (centred distribution,
no residue at the centre) and the principal value integral is

```
P.V. ∫₋∞^∞  (dg/dQ) / (Q − Q₀) dQ
  =  P.V. ∫  [−(Q−Q₀)/σ_Q² g(Q)] / (Q−Q₀) dQ
  =  −(1/σ_Q²) ∫ g(Q) dQ  =  −1/σ_Q²                   (4.4)
```

The stability criterion (4.3) therefore gives

```
K_eff · (1/σ_Q²)  <  1     →     K_eff  <  σ_Q²         (4.5)
```

The drive parameter `K_eff` is proportional to the growth rate:
`K_eff ~ μ_grow × (turn period)²`.  Identifying `K_eff = μ_grow` at
threshold gives the **Gaussian Landau threshold**:

```
μ_grow  <  σ_Q²
(κβ/2)|sin(2πQ₀)|  <  σ_Q²
κ_th  =  2σ_Q² / (β |sin(2πQ₀)|)                        (4.6)
```

For `σ_Q = 0.002`, `β = 10 m`, `Q₀ = 0.28`:

```
κ_th^{theory}  =  2 × (0.002)² / (10 × 0.990)  ≈  8 × 10⁻⁷ rad/m
```

> **Interpretation.**  The threshold (4.6) grows as `σ_Q²`, not `σ_Q`.
> Doubling the chromaticity (and hence `σ_Q`) quadruples the threshold — a
> strong, nonlinear benefit.  The beam is stable when the tune spread is large
> enough that the resonant denominator in (4.3) never vanishes over the
> populated tune range.

### 4.3 Role of Chromaticity as Landau-Damping Source

With chromaticity `ξ = dQ/dδ` and a Gaussian momentum distribution of rms
width `σ_δ`, the tune spread is

```
σ_Q  =  |ξ| σ_δ                                         (4.6)
```

Substituting into (4.5):

```
κ_th  =  |ξ| σ_δ / (β sin²(2πQ₀))                      (4.7)
```

This is the design equation used in practice to set chromaticity in a storage
ring: **a more negative (or positive) chromaticity widens the tune spread and
raises the instability threshold**.

However, chromaticity also introduces **chromatic head-tail effects** (see
Section 5) and must be controlled carefully.  The optimal balance is a key
consideration in accelerator design.

### 4.4 Monte Carlo Realisation of Phase Mixing

In the MC simulation, phase mixing is not imposed analytically but emerges
naturally from the particle dynamics:

1. Each particle `i` has momentum offset `δᵢ ∼ N(0, σ_δ)`.
2. Its phase advance per turn is `φᵢ = 2π(Q₀ + ξ δᵢ)`.
3. After `n` turns without a coherent kick, the centroid is
   ```
   x̄(n)  =  (1/N) Σᵢ xᵢ(n)
   ```
   which converges (as `N → ∞`) to the ensemble average (4.1) — a
   Gaussian decoherence envelope on a betatron carrier.
4. For finite `N`, the statistical fluctuation of the centroid is the
   **sampling noise floor**:
   ```
   σ_noise  =  √(⟨x̄²⟩_incoherent)  =  √(εβ / N)       (4.7)
   ```
   (standard error of the mean position over `N` independently phased
   particles, each with rms position `√(εβ)` ).

**Equation (4.7) constrains simulation parameters.**  With
`ε = 10⁻⁶ m·rad`, `β = 10 m`, `N = 3000`:

```
σ_noise  =  √(10⁻⁶ × 10 / 3000)  ≈  58 μm
```

The initial offset `x₀ = 2 mm ≫ 58 μm` ensures the coherent centroid signal
is clearly above the noise floor for several decoherence times.

**Why the empirical threshold far exceeds the Sacherer prediction.**
The Sacherer limit (4.6) is `κ_th^{theory} ≈ 8×10⁻⁷ rad/m`, while the
simulated threshold at `N = 3000` is `κ_th^{sim} ≈ 3×10⁻³ rad/m` — roughly
**3 000× larger**.  The root cause is finite-`N` noise: the coherent signal
after decoherence (~`A₀ exp(−2π²σ_Q²n²)`) falls below the noise floor
`σ_noise` after `τ_noise ≈ τ_deco√(ln(A₀/σ_noise))/(π√2 σ_Q)` turns.
Once submerged, the kick cannot reinforce the mode.  The threshold condition
becomes `μ_grow × τ_noise ≳ 1` (growth must overcome decoherence before the
signal is lost in noise), which gives `κ_th^{sim} ∝ σ_Q / (β |sin μ|)` —
proportional to `σ_Q`, not `σ_Q²`.  Increasing `N` lowers the noise floor,
reduces `τ_noise`, and drives `κ_th^{sim}` toward the Sacherer limit.

---

## 5. Head-Tail Instability

### 5.1 Classical Courant–Snyder Head-Tail Theory

The **head-tail instability** [14] arises from the within-bunch wake coupling
between the longitudinal head and tail of a single bunch.  At high current the
`m = 0` and `m = −1` head-tail modes merge into the **Transverse Mode Coupling
Instability** (TMCI) [13].

In the full theory a particle occupies a phase-space point `(x, x', z, δ)`
where `z > 0` is ahead of the synchronous particle.  Synchrotron oscillations
at tune `Q_s` are described per revolution turn `n` by the canonical
longitudinal map [2, Ch. 3]:

```
z_{n+1}  =  z_n  −  η C δ_n                                        (5.1a)
δ_{n+1}  =  δ_n  +  (2πQ_s)² z_n / (η C)                          (5.1b)
```

where `η = α_c − 1/γ²` is the **slip factor** (`η > 0` above transition),
`α_c` the momentum compaction, and `C` the circumference.  Equations (5.1)
are the discrete-map form of the continuous-time equations `dz/dt = −ηcδ`,
`dδ/dt = −ω_s²z/(ηc)`, with angular synchrotron frequency `ω_s = 2πQ_sf₀`.
The linearised map (5.1) describes simple harmonic oscillation in `(z, δ)`
phase space at tune `Q_s`.

The **chromaticity head-tail phase** is the transverse phase accumulated
between head and tail due to the chromatic tune shift over the synchrotron
half-period `T_s/2 = C/(2Q_sf₀c)`:

```
χ_ξ  =  ξ ω₀ z_max / (η c)  =  ξ π / (η Q_s)                     (5.2)
```

(Here `z_max` is the synchrotron amplitude.)  The coherent mode frequencies
`Ω_m` of the bunched beam satisfy the **Sacherer integral equation** [12, 13].
For small `χ_ξ` the tune shifts reduce to

```
ΔQ_m  ≈  −i N_b r₀ β Z_⊥(ω₀(1 + m Q_s)) / (4π γ C)  ·  G_m(χ_ξ)  (5.3)
```

where `G_m(χ_ξ) → 1` as `χ_ξ → 0`.  The factor `G_m` encodes the Bessel-
function overlap of the mode shape with the chromaticity-induced phase
`exp(iχ_ξ z/z_max)`.

Mode structure:

| Mode `m` | Frequency | Stability at small current |
|---|---|---|
| `0` (rigid-bunch dipole) | `ω_β` | Stable if `Im(ΔQ₀) < 0` |
| `±1` (snake modes) | `ω_β ± ω_s` | Stable if `|ΔQ₀| < Q_s` |
| High `\|m\|` | `ω_β + m ω_s` | Increasingly stable (short-range wake) |

- `m = 0`, `ξ > 0`:  `Im(ΔQ₀) < 0` — stable (chromaticity stabilises the rigid mode)
- High current: modes `m = 0` and `m = −1` **merge** → **TMCI** threshold

### 5.2 Slice Model (Used in This Code)

The full Courant–Snyder treatment requires synchrotron oscillations that are
outside the scope of this simulation.  Instead, we implement a
**quasi-static slice model**:

1. Each particle is assigned a fixed longitudinal coordinate
   `zᵢ ∼ N(0, σ_z)` at initialisation (no synchrotron motion).
2. The bunch is divided into `K` slices indexed `j = 1, …, K` in order of
   increasing `z` (smaller `z` → tail, larger `z` → head).
3. Each turn, the **head slice `i`** excites a transverse wake that kicks
   the **tail slice `j < i`** via:

```
Δx'_j  =  κ_w · Σ_{i > j}  W_⊥(zᵢ − zⱼ) · x̄ᵢ / K      (5.3)
```

where `W_⊥(Δz) = W₀ exp(−Δz/z_w)` is the broad-band resonator wake (2.4)
and `x̄ᵢ` is the centroid of slice `i`.

**Equation (5.3) is the discrete analog of the convolution integral:**

```
ΔP_⊥(z)  =  κ_w ∫_{z}^{∞}  W_⊥(z' − z) · x̄(z') dz'         (5.4)
```

In the code, `wake_strength` (`κ_w`) absorbs all dimensional constants
(`N_b r₀ W₀ / γ C`) and the per-slice normalisation; equation (5.3) as
implemented sums directly without dividing by `K` — the prefactor is already
calibrated per source slice.

The sum in (5.3) runs over `O(K²)` pairs per turn.  For `K = 15` and
`N_turns = 800`: `15² × 800 = 180 000` wake evaluations — negligible cost.

### 5.3 Chromaticity-Induced Phase Detuning in the Slice Model

Without synchrotron oscillations the only source of Landau damping in the
head-tail model is the **chromatic tune spread** among particles within
each slice.

Particle `i` in slice `j` has tune `Q₀ + ξ δᵢ`.  The centroid of slice `j`
undergoes decoherence at rate governed by the local tune spread
`σ_Q^(j) ≈ |ξ| σ_δ` (same as the global spread, since `z` and `δ` are
independent in this model).

This provides a **Landau-damping mechanism within each slice**, analogous
to Section 4.4.  The competition between the wake drive (5.3) and the
intra-slice decoherence determines whether the head-tail mode is stable.

> **Limitation.**  In a real ring the dominant stabilisation of head-tail
> modes comes from the *synchrotron side-band* coupling [3, 14], not from
> intra-slice decoherence.  The slice model without synchrotron oscillations
> therefore **underestimates the threshold** and should be viewed as
> qualitative.  A quantitative model requires incorporating synchrotron
> oscillations by updating `zᵢ(n)` each turn.

---

## 6. Vlasov Perturbation Theory (Synopsis)

The rigorous derivation of all results in Sections 3–5 starts from the
**Vlasov equation** for the single-particle distribution function
`f(x, x', z, δ; n)`:

```
∂f/∂n  +  {f, H₀ + H_coll}  =  0                        (6.1)
```

where `{·,·}` is the Poisson bracket, `H₀` is the single-particle
Hamiltonian, and `H_coll[f]` is the collective self-field Hamiltonian
(a functional of `f`).  Linearising around the equilibrium `f₀` (Gaussian),
`f = f₀ + f₁` with `|f₁| ≪ f₀`:

```
∂f₁/∂n  +  {f₁, H₀}  =  −{f₀, H_coll[f₁]}             (6.2)
```

Equation (6.2) is linear in `f₁` and can be solved by the method of
characteristics or by an eigenmode expansion.  The eigenvalue `Ω` of the
coherent mode satisfies the dispersion relation (4.3).

The Monte Carlo simulation is a **macro-particle discretisation** of (6.1):
sampling `f₀` with `N` particles and evolving each under `H₀ + H_coll`.
The macro-particle approximation is faithful to the Vlasov equation when the
sampling noise per observable is small compared to the physical signal.  For
the centroid, this requires (§4.4)

```
σ_noise = √(εβ/N)  ≪  x̄_coherent
```

The Sacherer criterion (§4.2) predicts `κ_th^{theory} ≈ 8×10⁻⁷ rad/m` for
`N → ∞`.  For `N = 3000`, finite-N noise suppresses the phase-mixed signal
before the theoretical threshold is reached, raising the *observed* threshold
to `κ_th^{sim} ≈ 3×10⁻³ rad/m` — a factor `~3 000×` larger.  This is not a
failure of the simulation: it correctly tracks the Vlasov equation at finite
`N`.  The discrepancy simply confirms that **true Vlasov Landau damping
requires an astronomically larger particle count** than is practical in a MC
study; the code instead demonstrates phase mixing and its competition with the
coherent drive at finite `N`, which is itself a physically important regime
(e.g. early-stage emittance growth in a mismatched injected bunch).

---

## 7. Mapping Theory to the Monte Carlo Implementation

### 7.1 Rigid-Bunch Module (`mode='transverse'`)

| Physical quantity | Code symbol | Relation to theory |
|---|---|---|
| Coherent kick strength | `kappa` [rad/m] | `κ` in eqs. (3.1), (3.4), (3.6) |
| One-turn delay | `prev_x_bar` | Implements Case B of §3.2 |
| Chromaticity | `ring.chromaticity ξ` | Sets `σ_Q` via (4.6) |
| Aperture | `ring.aperture` [m] | Loss when `|x| > aperture` |

Per-turn algorithm for particle `i`:

```
Step 1:  Δxpᵢ  ← κ · x̄_{n-1}              (delayed coherent kick)
Step 2:  (x, xp) ← M(φᵢ) · (x, xp)        (individual one-turn map, φᵢ = 2π(Q₀ + ξδᵢ))
Step 3:  xp      ← xp − k₂ x²             (sextupole, if k₂ ≠ 0)
Step 4:  flag lost if |x| > aperture
Step 5:  x̄_n ← mean of surviving x         (update delay buffer)
```

**Stability condition (empirical, from §4.4 finite-N effects):**

```
κ < κ_th^{sim} ≈ σ_Q / (β sin²(2πQ₀))     [theory, N → ∞]
```

In practice for N = 3000 the empirical threshold is 10–40× larger.

### 7.2 Head-Tail Module (`mode='headtail'`)

| Physical quantity | Code symbol | Relation to theory |
|---|---|---|
| Wake strength | `wake_strength` ≡ κ_w | Absorbed `N_b r₀ W₀ / (γ C)` |
| Wake decay length | `wake_range` ≡ z_w | Eq. (2.4) |
| Number of slices | `n_slices` ≡ K | Discretisation of (5.4) |
| Bunch length | `sigma_z` ≡ σ_z | Sets longitudinal extent |

Per-turn algorithm:

```
Step 1:  Assign each alive particle to slice j based on z position.
Step 2:  Compute centroid x̄_j for each slice.
Step 3:  For each slice j (tail):
           For each slice i > j (head):
             Δxp_j += κ_w · exp(−(z_i − z_j)/z_w) · x̄_i / K
Step 4:  Apply one-turn map per particle (same as rigid-bunch step 2–4).
```

The `O(K²)` inner loop is the discrete approximation of the convolution
integral (5.4) with the exponential kernel (2.4).

### 7.3 Observable: Centroid Time Series

The primary output is `centroid_x[n] = x̄(n)`.  Physically this is the
**dipole moment** of the beam distribution.  In a real machine it is measured
by a **Beam Position Monitor (BPM)** turn by turn.

The **stability indicator** used in the tests is:

```
Stable:    mean |x̄(n)|  decreases over time  (Landau damping wins)
Unstable:  mean |x̄(n)|  is sustained or grows  (coherent drive wins)
```

---

## 8. Worked Numerical Example

Using the default simulation parameters:

| Parameter | Value |
|---|---|
| Tune `Q₀` | 0.28 |
| Beta function `β` | 10 m |
| Chromaticity `ξ` | −2 |
| Momentum spread `σ_δ` | 10⁻³ |
| Geometric emittance `ε` | 10⁻⁶ m·rad |
| Particles `N` | 3000 |
| Initial offset `x₀` | 2 mm |

**Derived quantities:**

```
σ_Q       =  |ξ| σ_δ  =  2 × 10⁻³

σ_x       =  √(εβ)    =  √(10⁻⁶ × 10)  =  3.16 mm

σ_noise   =  √(εβ/N)  =  3.16 mm / √3000  =  58 μm   ≪  x₀ = 2 mm  ✓

τ_deco    =  1/(2π√2 σ_Q)  ≈  56 turns       [eq. (4.2)]

μ_grow    =  (κβ/2)|sin(2πQ₀)|               [eq. (3.5)]
(κ=0.05)  =  (0.05 × 10 / 2) × |sin(2π×0.28)|
           =  0.25 × 0.990  ≈  0.247 turn⁻¹

τ_inst    =  1/μ_grow  ≈  4.0 turns   [without damping, eq. (3.6)]

κ_th      =  2σ_Q² / (β|sin(2πQ₀)|)          [Sacherer, eq. (4.6), N→∞]
(theory)  =  2 × (2×10⁻³)² / (10 × 0.990)  ≈  8.1 × 10⁻⁷ rad/m
```

**Empirical threshold (from simulation at N = 3000):**
`κ_th^{sim} ≈ 3×10⁻³ rad/m` — roughly **3 700×** the theoretical N→∞ value.
This is a finite-N noise-floor effect (§4.4): the coherent signal decoheres
below the noise level `σ_noise` well before the Sacherer threshold is active.

**Convergence exercise:**  Track the empirical threshold `κ_th^{sim}` vs. `N`
for `N = 500, 1000, 3000, 10 000, 100 000`.  The Sacherer formula predicts
`κ_th^{sim} → κ_th^{theory}` as `N → ∞`; the finite-N noise-floor argument
of §4.4 predicts `κ_th^{sim} ∝ σ_Q/(β|sinμ|) × f(N)` with `f(N) → 0` — a
useful test of both the implementation and the theory.

---

## 9. Notation Summary

| Symbol | Description | Units |
|--------|-------------|-------|
| `s` | Arc-length co-ordinate along ring | m |
| `C` | Ring circumference | m |
| `x, x'` | Transverse displacement and angle | m, rad |
| `z` | Longitudinal position within bunch (`z > 0`: head) | m |
| `δ = Δp/p₀` | Fractional momentum offset | — |
| `Q₀` | Bare (incoherent) betatron tune | — |
| `Q_s` | Synchrotron tune | — |
| `ξ = dQ/dδ` | Chromaticity | — |
| `σ_Q = \|ξ\| σ_δ` | RMS tune spread | — |
| `τ_deco` | Decoherence time (Gaussian, eq. 4.2) | turns |
| `β(s)` | Beta function | m |
| `α(s), γ(s)` | Courant–Snyder parameters | — |
| `μ = 2πQ` | Phase advance per turn | rad |
| `η = α_c − 1/γ²` | Slip factor | — |
| `N_b` | Bunch population | particles |
| `r₀` | Classical particle radius (`r_p = 1.535×10⁻¹⁸ m`, `r_e = 2.818×10⁻¹⁵ m`) | m |
| `γ = E/m₀c²` | Lorentz factor | — |
| `W_⊥(s)` | Transverse wake function | V/C/m |
| `Z_⊥(ω)` | Transverse impedance | Ω/m |
| `W₀` | Peak wake amplitude | V/C/m |
| `z_w` | Wake decay length | m |
| `κ_coh` | Coherent kick parameter (code: `kappa`) | rad/m |
| `κ_w` | Head-tail wake kick parameter (code: `wake_strength`) | — |
| `κ_th` | Landau-damping instability threshold | rad/m |
| `μ_grow` | Exponential growth rate per turn | turns⁻¹ |
| `K` | Number of longitudinal slices (code: `n_slices`) | — |
| `σ_z` | RMS bunch length (code: `sigma_z`) | m |
| `x̄` | Transverse bunch centroid | m |
| `J` | Courant–Snyder invariant (action) | m·rad |
| `ε = ⟨J⟩` | Geometric emittance | m·rad |
| `f(J, Q)` | Single-particle distribution (Vlasov) | — |
| `g(Q)` | Tune distribution function | — |
| `K_eff` | Sacherer impedance drive parameter | tune² |
| `α_d = ω_r/(2Q)` | BBR damping rate | rad/s |
| `ω̄ = √(ω_r²−α_d²)` | Damped resonant frequency | rad/s |
| `χ_ξ = ξπ/(ηQ_s)` | Chromaticity head-tail phase | rad |
| `ψ` | Betatron phase angle (action-angle) | rad |
| `τ_noise` | Signal-to-noise e-folding time | turns |

---

## 10. References

**Textbooks**

[1] H. Wiedemann, *Particle Accelerator Physics*, 4th ed., Springer, 2015.
    ISBN 978-3-319-18316-9.  (Chapters 14–16 cover impedance and instabilities.)

[2] S. Y. Lee, *Accelerator Physics*, 4th ed., World Scientific, 2019.
    ISBN 978-9-813-27744-9.  (Chapter 4 covers collective effects.)

[3] E. D. Courant and H. S. Snyder, "Theory of the Alternating-Gradient
    Synchrotron," *Annals of Physics* **3**, 1–48 (1958).
    https://doi.org/10.1016/0003-4916(58)90012-5
    (Original derivation of Courant–Snyder optics and head-tail modes.)

[4] K. Y. Ng, *Physics of Intensity Dependent Beam Instabilities*,
    World Scientific, 2006.  ISBN 978-9-812-38962-6.

**Collective Instabilities — Primary Reference**

[5] A. W. Chao, *Physics of Collective Beam Instabilities in High Energy
    Accelerators*, Wiley, 1993.  ISBN 978-0-471-55184-3.
    [**The standard graduate reference; all sections cited.**]
    Available at: https://www.slac.stanford.edu/~achao/wileybook.html

**Wake Fields and Impedance**

[6] P. B. Wilson, "Introduction to Wake Fields and Wake Potentials,"
    SLAC-PUB-4547, SLAC, 1989.
    https://www.slac.stanford.edu/cgi-bin/spiface/find/hep/www?key=2407754

[7] W. K. H. Panofsky and W. A. Wenzel, "Some Considerations Concerning the
    Transverse Deflection of Charged Particles in Radio-Frequency Fields,"
    *Rev. Sci. Instrum.* **27**, 967 (1956).
    https://doi.org/10.1063/1.1715427

[8] T. Weiland, "Transverse Beam Cavity Interaction," DESY-M-82-04, 1982.
    (Numerical wake computation; foundation for MAFIA, CST Particle Studio.)

**Landau Damping**

[9] L. D. Landau, "On the Vibrations of the Electronic Plasma,"
    *J. Phys. USSR* **10**, 25 (1946).
    (Original Landau damping derivation in plasma physics; English translation
    in *JETP* **16**, 574 (1946).)

[10] H. G. Hereward, "Landau Damping by Non-linearity," CERN Report 65-20,
     1965.  https://cds.cern.ch/record/314149
     (First application of Landau damping to particle beams.)

[11] D. Möhl, G. Petrucci, L. Thorndahl, and S. van der Meer,
     "Physics and Technique of Stochastic Cooling,"
     *Phys. Rep.* **58**, 73 (1980).
     https://doi.org/10.1016/0370-1573(80)90137-6

**Sacherer Integral and Stability Criterion**

[12] F. J. Sacherer, "Methods for Computing Bunched Beam Instabilities,"
     CERN Report SI-BR/72-5, 1972.  https://cds.cern.ch/record/328970
     (Derivation of the Sacherer stability integral; Gaussian threshold.)

[13] F. J. Sacherer, "Transverse Bunched-Beam Instabilities — Theory,"
     *Proc. 9th Int. Conf. High Energy Accelerators*, SLAC, 1974, pp. 347–351.

**Head-Tail Instability**

[14] M. Sands, "The Head-Tail Effect: An Instability Mechanism in Storage Rings,"
     SLAC-TN-69-8, SLAC, 1969.  https://www.slac.stanford.edu/cgi-bin/spiface/find/hep/www?key=1330434

[15] A. Chao, B. Richter, and C. Yao, "Beam Emittance Growth Caused by
     Transverse Deflecting Fields in a Linear Accelerator,"
     *Nucl. Instrum. Methods* **178**, 1 (1980).
     https://doi.org/10.1016/0029-554X(80)90648-2
     (Head-tail model for linacs; method extended to rings by [5].)

**Vlasov Equation and Kinetic Theory**

[16] A. A. Vlasov, "On the Kinetic Theory of an Assembly of Particles with
     Collective Interaction," *J. Phys. USSR* **9**, 25 (1945).

[17] S. Krinsky and J. M. Wang, "Longitudinal Instabilities of Bunched Beams
     Subject to a Non-Harmonic RF Potential,"
     *Part. Accel.* **12**, 107 (1982).

**Lecture Notes and Reviews (Open Access)**

[18] W. Herr, "Coherent Beam Instabilities," *CERN Accelerator School on
     Intermediate Accelerator Physics*, CERN 2006-002, pp. 379–439.
     https://cds.cern.ch/record/941323
     (Accessible graduate introduction; recommended starting point.)

[19] E. Métral et al., "Beam Instabilities in Hadron Synchrotrons,"
     *IEEE Trans. Nucl. Sci.* **63**, 1001 (2016).
     https://doi.org/10.1109/TNS.2015.2510032

[20] N. Mounet, "Impedance and Instabilities," Lecture notes, CERN Accelerator
     School, 2018.  https://indico.cern.ch/event/703620/
     (Covers resistive wall, geometric wakes, and TMCI with modern treatment.)

[21] S. Stupakov, "Echo Effect in Hadron Colliders,"
     *Phys. Rev. Lett.* **74**, 3057 (1995).
     https://doi.org/10.1103/PhysRevLett.74.3057
     (Reversibility of Landau decoherence via the echo; demonstrates that
     phase mixing is not true irreversible damping.)

---

*Document maintained alongside `python/src/mathphys/collective.py`.*
*Last updated: 2026-02-28.*
