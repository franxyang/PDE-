```markdown
# agent.md — PDE- JAX Baseline Builder
reference: https://arxiv.org/pdf/2408.12404 (this is similar project but using PyTorch, We use Jax!)
my teammate's work last week: https://github.com/markm39/pde-constrained-opt-nn/blob/main/1d_heat_force_vector.ipynb

## Context
We need a no-NN, reproducible JAX baseline for PDE-constrained optimization on the 1D heat equation. It must run **on CPU** reliably (avoid Apple Metal crashes) and be **autodiff/JIT friendly** (no tracer-boolean ifs). Output must include plots and a concise README for presentation.

## Tech Stack
- Python 3.10–3.12
- JAX (CPU), jaxlib (CPU), numpy, matplotlib
- (Optional) jaxopt for L-BFGS

## Deliverables
- `heat_inverse.py` — single entry script; runs Exp-A/B/C and saves six PNGs.
- `requirements.txt`, `README.md`, `.gitignore`
- (Optional) modularization to `solver.py` / `losses.py` / `experiments.py` is allowed, but `heat_inverse.py` must work standalone.

## Directory
```

PDE-/
heat_inverse.ipynb
requirements.txt
README.md
.gitignore

````

## System Constraints & Must-haves
- Force CPU backend to avoid Metal issues:
  ```python
  import os
  os.environ.setdefault("JAX_PLATFORMS", "cpu")  # put before importing jax
````

* Absolutely **no Python `if` on traced loop indices** inside JIT/scan; use `lax.fori_loop` / `lax.scan` and branchless formulas (or `jnp.where` when needed).
* Tri-diagonal Thomas solver must be scan/fori_loop-friendly and safe for `n==1`.

## Tasks

1. Implement grid/assembly for FD Laplacian on interior nodes (Dirichlet).
2. Implement tridiagonal `solve_tridiag(main, off, rhs)` with:

   * forward elimination using `lax.scan`
   * backward substitution using `lax.fori_loop`
   * no tracer-based conditionals
3. Implement `make_stepper(Nx, dx, dt, kappa)` returning `@jit`-compiled `step(u, s)`.
4. Implement `rollout(u0, s, step, Nt)` using `lax.scan`.
5. Implement losses:

   * `loss_finaltime`, `loss_sparsepoints`, `loss_multitime`
   * JIT their grads: `jit(grad(loss_*))`
6. Implement experiments A/B/C with default hyperparameters:

   * A: full-field final-time; lr=0.5, iters=200, λ=1e-2
   * B: sparse sensors (M≈12); lr=0.3, iters=300, λ=5e-2
   * C: multi-time (3 time indices); lr=0.4, iters=200, λ=1e-2
   * Record wall-time, print relative error `||s_est - s_true|| / ||s_true||`
7. Plot and save:

   * loss vs iteration (semilogy)
   * source `s_true` vs `s_est` (mark sensors for B)
8. Write `README.md` covering:

   * PDE + discretization
   * how to run, default params
   * description of A/B/C
   * generated files
9. Provide `requirements.txt` and `.gitignore`.

## Non-goals

* No neural networks; no PINNs.
* No GPU acceleration; no jax-metal requirement.
* No external PDE/FEM libraries.

## Quality Bar / Acceptance

* `python heat_inverse.py` runs end-to-end on CPU, creates 6 PNGs.
* No JAX tracer boolean errors; no Metal crash.
* Relative error for Exp-A typically < 0.15 with default noise (tunable via λ).
* Code is clear, with short docstrings and inline comments.

## Troubleshooting Playbook

* **TracerBoolConversionError**: remove Python `if` on traced vars; use branchless update or `jnp.where`.
* **Metal crash**: ensure `JAX_PLATFORMS=cpu` is set before importing jax, or uninstall `jax-metal`.
* **Performance**: keep `@jit` on `step` and loss/grad; avoid Python loops in time—use `lax.scan`.

## Example Commit Message

```
JAX no-NN baseline: 1D heat source inversion (Exp A/B/C) + plots
- CPU-only via JAX_PLATFORMS to avoid Metal crash
- Tri-diagonal Thomas solver with scan/fori_loop (no tracer if)
- Loss/grad JIT; three experiments implemented
- README + requirements + generated figures
```

```
