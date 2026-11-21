# EM vs. Variational Inference for Bayesian Hierarchical Models

A simulation study comparing EM (MAP estimation) and mean-field variational inference
on two Gaussian hierarchical models.

## The models

Both models have known variances `σ², τ², τ₀²` and a known hyperprior mean `μ₀`:

```
μ       ~ N(μ₀, τ₀²)
θ_k | μ ~ N(μ, τ²)          k = 1..K   (group means)
x_i     ~ N(θ_{z_i}, σ²)               (observations)
```

* **Model 1 (known groups):** every label `z_i` is observed.
* **Model 2 (unknown groups):** the labels are missing, so the data is a Gaussian mixture
  with a hierarchical prior on the component means and mixture weights `π`.

## The two methods

**EM (MAP estimate)** - [`src/emvi/em.py`](src/emvi/em.py)

* *Model 1:* nothing is missing, so EM just maximises the log posterior. Setting its
  gradient to zero gives two closed-form updates (with `n` observations per group),
  alternated until convergence:

  ```
  θ_k = (n τ² x̄_k + σ² μ) / (n τ² + σ²)
  μ   = (τ₀² Σ_k θ_k + τ² μ₀) / (K τ₀² + τ²)
  ```

  The objective is a concave quadratic, so this always reaches the unique maximum.
* *Model 2:* the labels are the missing data.
  * **E-step:** responsibilities `γ_ik ∝ π_k N(x_i | θ_k, σ²)`.
  * **M-step:** with `N_k = Σ_i γ_ik` and `S_k = Σ_i γ_ik x_i`, set `π_k = N_k / N`.
    Then solve the same two equations as in model 1, with `n x̄_k` replaced by `S_k`
    and `n` by `N_k`. Writing `θ_k = a_k + b_k μ`, this linear system has a
    closed-form solution, so no numerical optimiser is needed.

**VI (mean-field ADVI via PyMC)** - [`src/emvi/vi.py`](src/emvi/vi.py)

* *Model 1:* fit the model above directly.
* *Model 2:* ADVI cannot handle discrete latent variables. So the labels are
  marginalised out, the weights get a flat `Dirichlet(1)` prior, and the likelihood is
  `Σ_k π_k N(x | θ_k, σ²)`.
* Estimates are posterior means of the fitted approximation. Because the posterior of
  model 1 is Gaussian, VI and EM should give (almost) the same answer there.

## Simulation study

Defined in [`configs/simulation.json`](configs/simulation.json). It has four scenarios:
sample size, group sizes, number of groups, and hyperprior misspecification. Each one is
replicated `num_sims` times. Each replicate has its own seed, so EM and VI see
identical data. For model 2, estimated and true components are matched by sorting their
means.

```bash
uv sync
uv run pytest                                          # quick correctness checks
uv run pre-commit install                              # ruff, mypy and basic file checks on commit
uv run scripts/run_simulations.py --method em          # ~1 min
uv run scripts/run_simulations.py --method vi          # hours; add --quick for a smoke test
uv run scripts/make_figures.py                         # -> figures/
```

Results go to `results/` and figures to `figures/`. Neither is committed.

## Note for macOS

If PyTensor fails with `tapi error: malformed file` (Command Line Tools linker vs. a
newer SDK), run with `PYTENSOR_FLAGS="cxx="` to skip C compilation (slower), or update
the Command Line Tools.
