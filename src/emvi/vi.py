"""Mean-field variational inference (ADVI, via PyMC) for both models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymc as pm
from pymc.variational.callbacks import CheckParametersConvergence

from .em import Priors


@dataclass
class VIResult:
    theta: np.ndarray  # posterior mean of theta
    mu: float  # posterior mean of mu
    weights: np.ndarray | None  # posterior mean of mixture weights (model 2)
    elbo: np.ndarray  # ELBO per iteration
    iterations: int  # iterations actually run (may stop early on convergence)
    approx: object = None  # the fitted pymc approximation


def build_known_groups_model(X: np.ndarray, priors: Priors) -> pm.Model:
    X = np.asarray(X, dtype=float)
    K = X.shape[1]
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=priors.mu_naught, sigma=np.sqrt(priors.tau_naught2))
        theta = pm.Normal("theta", mu=mu, sigma=np.sqrt(priors.tau2), shape=K)
        pm.Normal("obs", mu=theta, sigma=np.sqrt(priors.sigma2), observed=X)
    return model


def build_mixture_model(x: np.ndarray, priors: Priors, num_groups: int) -> pm.Model:
    """Model 2 with the discrete labels marginalised out (ADVI needs a
    continuous latent space) and a flat Dirichlet prior on the weights."""
    x = np.asarray(x, dtype=float).ravel()
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=priors.mu_naught, sigma=np.sqrt(priors.tau_naught2))
        theta = pm.Normal("theta", mu=mu, sigma=np.sqrt(priors.tau2), shape=num_groups)
        weights = pm.Dirichlet("weights", a=np.ones(num_groups))
        pm.Mixture(
            "obs",
            w=weights,
            comp_dists=pm.Normal.dist(mu=theta, sigma=np.sqrt(priors.sigma2), shape=num_groups),
            observed=x,
        )
    return model


def fit_advi(
    model: pm.Model,
    max_iter: int = 15_000,
    tol: float = 1e-7,
    n_samples: int = 2_000,
    seed: int | None = None,
    progressbar: bool = False,
) -> VIResult:
    """Fit mean-field ADVI and summarise the approximate posterior by its mean."""
    with model:
        approx = pm.fit(
            n=max_iter,
            method="advi",
            random_seed=seed,
            progressbar=progressbar,
            callbacks=[CheckParametersConvergence(diff="absolute", tolerance=tol)],
        )
        idata = approx.sample(n_samples, random_seed=seed)

    post = idata.posterior
    has_weights = "weights" in post
    return VIResult(
        theta=post["theta"].mean(dim=("chain", "draw")).values,
        mu=float(post["mu"].mean()),
        weights=post["weights"].mean(dim=("chain", "draw")).values if has_weights else None,
        # PyMC records the loss, i.e. the *negative* ELBO.
        elbo=-np.asarray(approx.hist),
        iterations=len(approx.hist),
        approx=approx,
    )
