# Weighted Caputo Fractional-Order Total Variation for Cauchy Noise Removal

[![MATLAB](https://img.shields.io/badge/MATLAB-R2021b%2B-blue.svg)](https://www.mathworks.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-The%20Visual%20Computer%202026-green.svg)](https://doi.org/10.1007/s00371-026-04524-9)

Official MATLAB implementation of the **Weighted Fractional-Order Total Variation (FOTV)** model with Caputo directional derivatives and a robust Cauchy data-fidelity term, solved via the Split Bregman (SB) framework with automated Bayesian hyperparameter optimization.

> **A New Weighted Caputo Fractional-Order Total Variation for Cauchy Noise Removal with Bayesian Optimization**  
> Hamdi Braiek  
> *The Visual Computer*, 2026  
> DOI: [10.1007/s00371-026-04524-9](https://doi.org/10.1007/s00371-026-04524-9)

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Mathematical Model](#mathematical-model)
- [Algorithm](#algorithm)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Options](#options)
- [Bayesian Optimization](#bayesian-optimization)
- [Benchmark Results](#benchmark-results)
- [Citation](#citation)
- [License](#license)

---

## Overview

Image denoising under **Cauchy noise** is a challenging problem because the Cauchy distribution has heavy tails, undefined variance, and cannot be handled reliably by classical Gaussian-based methods. This repository proposes a novel variational framework that addresses this challenge through three integrated mechanisms:

1. **Weighted Caputo fractional-order total variation (FOTVᵅw)** — a spatially adaptive regularizer formulated in the Caputo sense (fractional order $1 < \alpha \leq 2$), generalizing both classical TV and unweighted FOTV.
2. **Nonconvex Cauchy data-fidelity term** — derived from the Cauchy log-likelihood, robust to impulsive outliers.
3. **Bayesian Optimization (BO)** — automated, data-driven hyperparameter selection based on PSNR, eliminating the need for manual tuning.

The resulting nonsmooth, nonconvex energy is minimized efficiently using a **Split Bregman (SB)** algorithm coupled with an Iteratively Reweighted Least Squares (IRLS) half-quadratic reformulation of the Cauchy fidelity, and a **matrix-free Preconditioned Conjugate Gradient (PCG)** for the linear subproblems.

---

## Key Contributions

- **(i)** A new variational formulation integrating a robust Cauchy data-fidelity term with a spatially adaptive weighted FOTV regularizer in the Caputo sense.
- **(ii)** Theoretical proof of the existence of minimizers in the weighted fractional bounded-variation space **$BV^\alpha_w(\Omega)$**.
- **(iii)** An efficient Split Bregman algorithm combined with a half-quadratic reformulation (IRLS) to handle the nonconvex and nonsmooth structure.
- **(iv)** A Bayesian optimization framework for automatic, reproducible hyperparameter selection.
- **(v)** Comprehensive experiments on synthetic, natural, and medical images (grayscale and color), demonstrating superior performance over classical TV, penalized TV, TGV, GBLR, and standard FOTV.

---

## Mathematical Model

### Variational Energy

For an observed noisy image `$f = u + \eta$`, with `$\eta \~ Cauchy(0, \gamma)$`, the model minimizes:

$$\min_u  \text{FOTV}^\alpha_w(u)  +  \lambda \int_\Omega \log(1 + (f(x) - u(x))^2 / \gamma^2) dx$$


where:

- **$\text{FOTV}^\alpha_w(u)$** is the weighted fractional-order total variation of order $\alpha \in (1, 2)$
- **$\lambda > 0$** is the regularization parameter
- **$\gamma > 0$** is the Cauchy scale parameter

### Caputo Fractional Gradient

The Caputo directional derivatives in 2D are:

```
ᶜDₓᵅ u(x,y) = 1/Γ(2−α) ∫ₐˣ (x−t)^(1−α) ∂²ₜu(t,y) dt
ᶜDᵧᵅ u(x,y) = 1/Γ(2−α) ∫ₐʸ (y−s)^(1−α) ∂²ₛu(x,s) ds
```

and the Caputo fractional gradient is: `∇^(α,C) u = [ᶜDₓᵅ u, ᶜDᵧᵅ u]ᵀ`.

### Discrete Caputo Approximation

Backward Caputo quadrature via second differences with weights:

```
aₘ^(α) = (m+1)^(1−α) − m^(1−α),  m = 0, 1, ..., R−1
```

Symmetric boundary padding is used for well-posedness and artifact-free denoising.

### Adaptive Spatial Weight

```
w(x) = exp(−|∇(Gσ * u)(x)|)
```

This exponential edge map assigns **low weights near edges** (preserving sharp transitions) and **high weights in smooth regions** (encouraging stronger smoothing).

### IRLS Half-Quadratic Reformulation

The Cauchy fidelity is linearized iteratively using:

```
ω(x) = 2 / (γ² + (f(x) − u(x))²)
```

transforming the nonconvex fidelity into a sequence of tractable weighted least-squares problems.

---

## Algorithm

```
Algorithm: Weighted Caputo FOTV – Split Bregman + IRLS

Input:  noisy image F, weights W, parameters λ, β, α, γ
Output: denoised image U

Initialize: u ← F,  d ← Du,  b ← 0

for k = 1, 2, ..., maxOuter do
    for inner = 1, ..., maxInner do   % IRLS
        Compute IRLS weights: ω ← 2 / (γ² + (F − u)²)
        Solve linear system via PCG:
            (ω·I + β·DᵀD + μ·I) u = ω·F + β·Dᵀ(d − b)
    end
    d-update (isotropic shrinkage):
        d ← shrink(Du + b, λ·w/β)
    b-update (Bregman variable):
        b ← b + Du − d
    Check relative change: stop if ‖uᵏ − uᵏ⁻¹‖ / ‖uᵏ⁻¹‖ < tolOuter
end
```

---

## Repository Structure

```
weightedfotv/
│
├── caputo_fotv_cauchy_sb.m      # Main solver: SB + IRLS + Caputo derivative + PCG
├── applyD.m                     # Discrete Caputo fractional gradient (forward operator D)
├── applyDt.m                    # Adjoint operator Dᵀ (consistent with applyD)
├── matvec_W_beta_DtD.m          # Matrix-free matvec: (ωI + βDᵀD + μI)x for PCG
├── compute_energy.m             # Evaluates full energy functional (convergence monitoring)
├── gamma_func.m                 # Gamma function helper
└── README.md
```

### File Descriptions

| File | Description |
|------|-------------|
| `caputo_fotv_cauchy_sb.m` | Main entry point. Implements the complete Split Bregman iteration with IRLS for the Cauchy fidelity and matrix-free PCG for the u-subproblem. |
| `applyD.m` | Computes the discrete Caputo fractional gradient `[Dₓu, Dᵧu]` via weighted backward sums of second differences using the Caputo quadrature weights `aₘ^(α)`. |
| `applyDt.m` | Implements the adjoint operator `Dᵀ` consistent with the discretization in `applyD`. Needed for forming the linear system in the u-update step. |
| `matvec_W_beta_DtD.m` | Matrix-free representation of `(ωI + βDᵀD + μI)x` used as the system operator inside MATLAB's `pcg`. Avoids explicit matrix formation. |
| `compute_energy.m` | Evaluates the complete variational energy at each outer iteration for convergence monitoring and diagnostics. |
| `gamma_func.m` | Utility helper for the Gamma function used in Caputo weight computation. |

---

## Requirements

- **MATLAB R2021b** or later
- **Image Processing Toolbox** (for `padarray`, used for symmetric boundary padding)
- No other toolboxes required

---

## Usage

### Basic Call

```matlab
% Load and normalize a grayscale image
u_true = im2double(imread('cameraman.tif'));

% Add Cauchy noise
gamma_noise = 0.05;
noise = gamma_noise * tan(pi * (rand(size(u_true)) - 0.5));
f = u_true + noise;

% Run the solver with default options
u = caputo_fotv_cauchy_sb(f, struct());
```

### With Custom Options

```matlab
opts.alpha    = 1.5;    % fractional order in (1, 2)
opts.lambda   = 0.08;   % regularization weight
opts.beta     = 5.0;    % Split Bregman penalty
opts.gamma    = 0.05;   % Cauchy scale parameter (match noise level)
opts.maxOuter = 100;    % maximum SB outer iterations
opts.maxInner = 1;      % IRLS steps per outer iteration
opts.pcgTol   = 1e-4;   % PCG convergence tolerance
opts.pcgMaxIt = 200;    % PCG maximum iterations
opts.tolOuter = 1e-4;   % outer stopping criterion
opts.verbose  = true;   % print iteration info

u = caputo_fotv_cauchy_sb(f, opts);
```

### With a Custom Spatial Weight Map

```matlab
% Compute adaptive weight from the noisy image
f_smooth = imgaussfilt(f, 1.0);
grad_mag = sqrt(imgradientxy(f_smooth, 'intermediate').^2);
w = exp(-grad_mag);

opts.w = w;
u = caputo_fotv_cauchy_sb(f, opts);
```

### Evaluating Quality Metrics

```matlab
psnr_val = psnr(u, u_true);
ssim_val = ssim(u, u_true);
snr_val  = 20 * log10(norm(u_true(:)) / norm((u(:) - u_true(:))));
fprintf('PSNR: %.2f dB | SSIM: %.4f | SNR: %.2f dB\n', psnr_val, ssim_val, snr_val);
```

---

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `alpha` | `1.5` | Fractional order, must satisfy 1 < α < 2 |
| `lambda` | `0.1` | Regularization weight λ |
| `beta` | `5.0` | Split Bregman penalty parameter β |
| `gamma` | `10.0` | Cauchy scale parameter γ (set to match the noise level) |
| `maxOuter` | `100` | Maximum number of outer SB iterations |
| `maxInner` | `1` | IRLS inner iterations per outer loop |
| `pcgTol` | `1e-4` | PCG solver tolerance |
| `pcgMaxIt` | `200` | PCG maximum iterations |
| `R` | `max(N,M)` | Caputo truncation radius (full image size by default) |
| `w` | `ones(N,M)` | Spatial weight map (default: uniform flat weighting) |
| `mu` | `0` | Optional diagonal stabilizer for the linear system |
| `tolOuter` | `1e-4` | Outer loop relative-change stopping threshold |
| `verbose` | `true` | Print iteration info (relative change, energy) |

---

## Bayesian Optimization

The paper incorporates a Bayesian Optimization (BO) framework to automatically tune the key hyperparameters `Θ = {α, λ, β}`. The objective minimized by BO is:

```
J(Θ) = −PSNR(U_Θ, U_true)
```

where `U_Θ` is the restored image produced by the solver with parameters `Θ`, and `U_true` is the reference image. BO maintains a Gaussian Process (GP) surrogate of the objective and uses an Expected Improvement (EI) acquisition function to guide the search efficiently.

**Suggested BO search ranges** (used in the paper):

| Parameter | Range |
|-----------|-------|
| `alpha` | [1.1, 1.9] |
| `lambda` | [0.01, 1.0] |
| `beta` | [1.0, 20.0] |
| `gamma` | fixed (set to noise scale) |

The solver was run for **$T_{max} = 20$ BO iterations** in the experiments. This is sufficient for convergence in practice.

---

## Benchmark Results

The model was evaluated against the following baselines on 11 standard grayscale test images and color images, under multiple Cauchy noise levels (\gamma \in \{0.01, 0.05, 0.1\}$):

| Model | Description |
|-------|-------------|
| **TV** | Classical Rudin–Osher–Fatemi total variation |
| **pen-TV** | Penalized total variation |
| **TGV** | Total Generalized Variation |
| **GBLR** | Graph-based low-rank regularization |
| **FOTV** | Unweighted fractional-order total variation |
| **Weighted FOTV** | **Proposed model** |

The proposed weighted FOTV consistently achieves the **highest PSNR and SSIM** across all tested images and noise levels. Key observations:

- Classical TV suppresses outliers but introduces piecewise-constant (staircasing) artifacts.
- FOTV improves texture retention but lacks spatial adaptivity.
- The proposed **weighted FOTV** combines fractional smoothing with adaptive weighting, producing both quantitatively superior and visually superior restorations.
- Results are confirmed by PSNR, SSIM, SNR, and relative error metrics.

Experiments include grayscale (Clock, Cameraman, Barbara, Lena, MRI, etc.) and color (Splash, Bird, etc.) images. All results are fully reproducible using the code in this repository.

---

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{braiek2026newweighted,
  author  = {Hamdi Braiek},
  title   = {A New Weighted {C}aputo Fractional-Order Total Variation for {C}auchy Noise Removal with {B}ayesian Optimization},
  journal = {The Visual Computer},
  volume = {42},
  number={9},
  pages={364},
  year={2026},
  doi = {10.1007/s00371-026-04524-9},
  publisher={Springer}
}
```

---

## Author

**Hamdi Braiek**  
ESPRIT School of Engineering, Ariana, Tunisia  
Laboratory for Mathematical and Numerical Modeling in Engineering Science (LAMSIN),  
National Engineering School of Tunis, Tunisia  
📧 hamdi.houichet@gmail.com  
🔗 [ORCID: 0000-0002-3301-1385](https://orcid.org/0000-0002-3301-1385)

---

## License

This project is released for academic use. If you use this code, please cite the associated paper above.
