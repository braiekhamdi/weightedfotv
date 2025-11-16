
# 🧮 Caputo-FOTV–Cauchy Denoising (Split Bregman Implementation)

This repository contains a MATLAB implementation of the **Weighted Fractional-Order Total Variation (FOTV)** model with **Caputo directional derivatives** and a **Cauchy data-fidelity term**, optimized using the **Split Bregman (SB)** framework.
The code implements the model proposed in our article:

> **A New Weighted Caputo Fractional-Order Total Variation for Cauchy Noise Removal with Bayesian Optimization**


---

## 📌 Overview

This solver numerically minimizes the variational model

$$
\min_{u};
  \int_{\Omega} \big(-u \mathrm{div}^{\alpha} v(x)\big) dx + 
\lambda \int*{\Omega} \log!\left(1+\frac{(f(x)-u(x))^2}{\gamma^2}\right)dx ,
$$

where:

* (f) is the observed image corrupted by **impulsive Cauchy noise**,
* $\nabla^{\alpha,C}$ is the **Caputo fractional gradient** of order $1 < \alpha < 2$,
* $w(x)$ is a **spatially adaptive weighting map**,
* $\lambda > 0$ and $\gamma>0$ are regularization and fidelity parameters.

The fractional gradient is discretized using **backward Caputo quadrature**:

[
{}^{C}!D_x^{\alpha}u(i,j) =
\frac{1}{\Gamma(2-\alpha)}
\sum_{m=0}^{i-1}
\big[(m+1)^{1-\alpha} - m^{1-\alpha}\big];
\Delta_x^2 u(i-m,j),
]

and similarly for the y-direction.

The resulting minimization is solved with:

* **Split Bregman iteration** (outer loop),
* **Iteratively Reweighted Least Squares (IRLS)** for the Cauchy fidelity,
* **Matrix-free Preconditioned Conjugate Gradient (PCG)** for the linear subproblem,
* **Isotropic shrinkage** for the d-update.

This framework is efficient, robust to heavy-tailed noise, and well suited for fractional-order regularization.

---

## 📂 File Description

```
caputo_fotv_cauchy_sb.m     Main solver implementing Split Bregman + IRLS + Caputo derivative
applyD.m                    Computes the discrete Caputo fractional gradient via weighted backward sums of second differences. 
applyDt.m                   Implements the **adjoint operator** (D^T) consistent with the discretization of `applyD`
matvec_W_beta_DtD.m         Matrix-free representation of $((W + \beta D^T D + \mu I)x)$ sed inside MATLAB’s PCG.
compute_energy.m            Evaluates the full energy functional for monitoring convergence.
```

---

## ▶️ Usage

```matlab
u = caputo_fotv_cauchy_sb(f, opts);
```

### **Inputs**

* `f` – noisy grayscale image (double precision)
* `opts` – options struct (all optional)

### **Outputs**

* `u` – restored image

---

## ⚙️ Options (with defaults)

```matlab
opts.alpha            % fractional order (1 < alpha < 2)
opts.lambda           % regularization weight
opts.beta             % Split Bregman penalty
opts.gamma            % Cauchy scale parameter
opts.maxOuter         % outer SB iterations
opts.maxInner         % IRLS inner iterations per outer
opts.pcgTol           % PCG tolerance
opts.pcgMaxIt         % PCG max iterations
opts.R                % Caputo truncation radius
opts.verbose          % print iterations
opts.tolOuter         % outer stopping threshold
```

 

---

## 🧠 Mathematical Features

### ✔ Caputo fractional derivative (1 < α < 2)

Implemented via:

* backward fractional convolution,
* exact Caputo weights:
  $$
  a_m = (m+1)^{1-\alpha} - m^{1-\alpha},
  $$
* symmetric boundary padding for well-posedness.

### ✔ Robust Cauchy data fidelity

IRLS step uses weights:

$$
\omega(x)=\frac{2}{\gamma^2 + (f(x)-u(x))^2}.
$$

This handles **impulsive outliers** effectively.

### ✔ Split Bregman Minimization

Two blocks:

* **u-update** via matrix-free PCG,
* **d-update** via isotropic shrinkage,
* **b-update** enforcing consistency.

### ✔ Adaptive weighted FOTV

The algorithm accepts a dynamic weight map ($w(x)$), allowing:

* texture preservation,
* edge-aware smoothing,
* structural adaptivity.

 

---

## 🖥 MATLAB Requirements

* MATLAB R2021b or later
* Image Processing Toolbox (for `padarray`)
* No other toolboxes required

---

 

## 📜 Citation

If you use this solver in academic work, please cite our corresponding article:

```
@article{Braiek2025CaputoFOTV,
  author  = {Hamdi Braiek},
  title   = {A New Weighted Caputo Fractional-Order Total Variation for Cauchy Noise Removal with Bayesian Optimization},
  year    = {2025}
}
```



