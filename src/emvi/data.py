"""Synthetic data for the two hierarchical models studied in the project.

Model 1 (known groups)::

    mu        ~ N(mu_0, tau_0^2)
    theta_k   ~ N(mu, tau^2),        k = 1..K
    x_{ik}    ~ N(theta_k, sigma^2), i = 1..n

Model 2 (unknown groups): the same hierarchy for ``mu`` and ``theta``, but
the group label of every observation is discarded, so the data is a pooled,
shuffled 1-D sample from a Gaussian mixture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class KnownGroupsSample:
    X: np.ndarray  # (n_per_group, K)
    theta: np.ndarray  # (K,)
    mu: float


@dataclass
class MixtureSample:
    X: np.ndarray  # (N,)
    theta: np.ndarray  # (K,)
    mu: float
    weights: np.ndarray  # (K,) empirical group proportions


def _draw_mu(rng: np.random.Generator, mu: float | None, mu_0: float, tau_0_2: float) -> float:
    """Return ``mu`` if fixed, otherwise draw it from its hyperprior."""
    if mu is not None:
        return float(mu)
    return float(rng.normal(mu_0, np.sqrt(tau_0_2)))


def sample_known_groups(
    rng: np.random.Generator,
    num_groups: int,
    num_obs_per_group: int,
    sigma2: float = 1.0,
    tau2: float = 1.0,
    tau_naught2: float = 1.0,
    mu_naught: float = 0.0,
    mu: float | None = None,
) -> KnownGroupsSample:
    """Draw a data set from model 1.

    If ``mu`` is given it is used as the true population mean; otherwise it is
    drawn from ``N(mu_naught, tau_naught2)``.
    """
    mu = _draw_mu(rng, mu, mu_naught, tau_naught2)
    theta = rng.normal(mu, np.sqrt(tau2), num_groups)
    X = rng.normal(theta, np.sqrt(sigma2), (num_obs_per_group, num_groups))
    return KnownGroupsSample(X=X, theta=theta, mu=mu)


def sample_mixture(
    rng: np.random.Generator,
    num_groups: int,
    num_obs_per_group: int | Sequence[int],
    sigma2: float = 1.0,
    tau2: float = 1.0,
    tau_naught2: float = 1.0,
    mu_naught: float = 0.0,
    mu: float | None = None,
) -> MixtureSample:
    """Draw a data set from model 2 (group labels are discarded)."""
    if np.isscalar(num_obs_per_group):
        counts = np.full(num_groups, int(num_obs_per_group))
    else:
        counts = np.asarray(num_obs_per_group, dtype=int)
    if counts.shape != (num_groups,):
        raise ValueError(f"num_obs_per_group must have length {num_groups}, got {counts.shape}")
    if counts.sum() == 0:
        raise ValueError("At least one observation is required.")

    mu = _draw_mu(rng, mu, mu_naught, tau_naught2)
    theta = rng.normal(mu, np.sqrt(tau2), num_groups)
    X = np.concatenate([rng.normal(t, np.sqrt(sigma2), c) for t, c in zip(theta, counts, strict=False)])
    rng.shuffle(X)
    return MixtureSample(X=X, theta=theta, mu=mu, weights=counts / counts.sum())
