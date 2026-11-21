"""EM-type MAP estimators for the two hierarchical models.

Both estimators assume the variances ``sigma2``, ``tau2`` and ``tau_naught2``
and the hyperprior mean ``mu_naught`` are known, and return the maximum a
posteriori (MAP) estimate of the group means ``theta`` and population mean
``mu`` (plus mixture weights for model 2).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

LOG_2PI = np.log(2 * np.pi)


def _normal_logpdf(x, mean, var):
    return -0.5 * (LOG_2PI + np.log(var) + (x - mean) ** 2 / var)


@dataclass
class Priors:
    """Known variances and hyperprior mean shared by both models."""

    sigma2: float = 1.0
    tau2: float = 1.0
    tau_naught2: float = 1.0
    mu_naught: float = 0.0

    @classmethod
    def from_config(cls, variances: dict, hyperpriors: dict) -> "Priors":
        return cls(
            sigma2=float(variances["sigma2"]),
            tau2=float(variances["tau2"]),
            tau_naught2=float(variances["tau_naught2"]),
            mu_naught=float(hyperpriors["mu_naught"]),
        )


@dataclass
class EMResult:
    theta: np.ndarray
    mu: float
    weights: np.ndarray | None
    log_posterior: list[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False


class EMKnownGroups:
    """MAP estimation for model 1 (group membership observed).

    With known group labels there is no missing data, so EM reduces to
    maximising the log posterior. We do this by coordinate ascent, alternating
    the two closed-form conditional maximisers

        theta_k = (n tau^2 xbar_k + sigma^2 mu) / (n tau^2 + sigma^2)
        mu      = (tau_0^2 sum_k theta_k + tau^2 mu_0) / (K tau_0^2 + tau^2)

    The objective is a concave quadratic, so the iteration converges to the
    unique global maximum from any starting point.
    """

    def __init__(self, priors: Priors, max_iter: int = 1000, tol: float = 1e-8):
        self.priors = priors
        self.max_iter = max_iter
        self.tol = tol

    def log_posterior(self, X: np.ndarray, theta: np.ndarray, mu: float) -> float:
        p = self.priors
        return float(
            _normal_logpdf(X, theta, p.sigma2).sum()
            + _normal_logpdf(theta, mu, p.tau2).sum()
            + _normal_logpdf(mu, p.mu_naught, p.tau_naught2)
        )

    def fit(self, X: np.ndarray, rng: np.random.Generator | None = None) -> EMResult:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        rng = rng if rng is not None else np.random.default_rng()
        p = self.priors
        n, K = X.shape
        xbar = X.mean(axis=0)

        mu = float(rng.normal(p.mu_naught, np.sqrt(p.tau_naught2)))
        theta = rng.normal(mu, np.sqrt(p.tau2), K)
        history = [self.log_posterior(X, theta, mu)]
        converged = False

        for it in range(1, self.max_iter + 1):
            theta = (n * p.tau2 * xbar + p.sigma2 * mu) / (n * p.tau2 + p.sigma2)
            mu = float((p.tau_naught2 * theta.sum() + p.tau2 * p.mu_naught) / (K * p.tau_naught2 + p.tau2))
            history.append(self.log_posterior(X, theta, mu))
            if abs(history[-1] - history[-2]) < self.tol:
                converged = True
                break

        if not converged:
            warnings.warn(f"EMKnownGroups did not converge in {self.max_iter} iterations.", stacklevel=2)
        return EMResult(theta=theta, mu=mu, weights=None, log_posterior=history, iterations=it, converged=converged)


class EMMixture:
    """MAP-EM for model 2 (group membership unobserved).

    Group labels ``z_i`` are treated as missing data, which turns the problem
    into a Gaussian mixture with a hierarchical prior on the component means.

    E-step: responsibilities ``gamma_ik ∝ pi_k N(x_i | theta_k, sigma^2)``
    (computed in log space).

    M-step: with ``N_k = sum_i gamma_ik`` and ``S_k = sum_i gamma_ik x_i``,
    the Q-function is maximised exactly by ``pi_k = N_k / N`` and the joint
    solution of the linear system

        theta_k = (tau^2 S_k + sigma^2 mu) / (tau^2 N_k + sigma^2)
        mu      = (tau_0^2 sum_k theta_k + tau^2 mu_0) / (K tau_0^2 + tau^2)

    obtained by substituting ``theta_k = a_k + b_k mu`` into the second line.
    """

    def __init__(
        self,
        priors: Priors,
        num_groups: int,
        max_iter: int = 1000,
        tol: float = 1e-6,
        n_init: int = 1,
    ):
        self.priors = priors
        self.num_groups = num_groups
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init

    # --- pieces -----------------------------------------------------------
    def _log_joint(self, x, theta, weights):
        """(N, K) array of log pi_k + log N(x_i | theta_k, sigma^2)."""
        with np.errstate(divide="ignore"):
            log_w = np.log(weights)
        return log_w[None, :] + _normal_logpdf(x[:, None], theta[None, :], self.priors.sigma2)

    def log_posterior(self, x, theta, mu, weights) -> float:
        p = self.priors
        return float(
            logsumexp(self._log_joint(x, theta, weights), axis=1).sum()
            + _normal_logpdf(theta, mu, p.tau2).sum()
            + _normal_logpdf(mu, p.mu_naught, p.tau_naught2)
        )

    def _e_step(self, x, theta, weights):
        log_r = self._log_joint(x, theta, weights)
        return np.exp(log_r - logsumexp(log_r, axis=1, keepdims=True))

    def _m_step(self, x, gamma):
        p = self.priors
        N_k = gamma.sum(axis=0)
        S_k = gamma.T @ x
        denom = p.tau2 * N_k + p.sigma2
        a = p.tau2 * S_k / denom
        b = p.sigma2 / denom
        mu = (p.tau_naught2 * a.sum() + p.tau2 * p.mu_naught) / (
            self.num_groups * p.tau_naught2 + p.tau2 - p.tau_naught2 * b.sum()
        )
        theta = a + b * mu
        weights = N_k / N_k.sum()
        return theta, float(mu), weights

    def _fit_once(self, x, rng) -> EMResult:
        K = self.num_groups
        theta = rng.choice(x, size=K, replace=len(x) < K).astype(float)
        mu = float(x.mean())
        weights = np.full(K, 1.0 / K)
        history = [self.log_posterior(x, theta, mu, weights)]
        converged = False

        for it in range(1, self.max_iter + 1):
            gamma = self._e_step(x, theta, weights)
            theta, mu, weights = self._m_step(x, gamma)
            history.append(self.log_posterior(x, theta, mu, weights))
            if abs(history[-1] - history[-2]) < self.tol:
                converged = True
                break
        return EMResult(theta=theta, mu=mu, weights=weights, log_posterior=history, iterations=it, converged=converged)

    def fit(self, x: np.ndarray, rng: np.random.Generator | None = None) -> EMResult:
        """Run EM ``n_init`` times from random starts and keep the best run."""
        x = np.asarray(x, dtype=float).ravel()
        rng = rng if rng is not None else np.random.default_rng()
        runs = [self._fit_once(x, rng) for _ in range(self.n_init)]
        best = max(runs, key=lambda r: r.log_posterior[-1])
        if not best.converged:
            warnings.warn(f"EMMixture did not converge in {self.max_iter} iterations.", stacklevel=2)
        return best
