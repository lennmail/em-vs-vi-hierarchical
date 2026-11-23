"""Simulation study: run EM or VI over every scenario in the config file.

Every replicate gets its own seed derived from (scenario, setting, replicate),
so EM and VI see exactly the same data sets even when run separately.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .data import sample_known_groups, sample_mixture
from .em import EMKnownGroups, EMMixture, Priors

SCENARIO_IDS = {"sample_size": 0, "group_sizes": 1, "number_of_groups": 2, "hyperprior": 3}
MODEL_IDS = {"known": 0, "mixture": 1}


def resolve_setting(scenario: dict, override: dict) -> dict:
    s = json.loads(json.dumps(scenario["defaults"]))  # deep copy
    s.update(override)
    return s


def _rngs(seed, scenario, model, setting_idx, rep):
    key = (SCENARIO_IDS[scenario], MODEL_IDS[model], setting_idx, rep)
    data_ss, fit_ss = np.random.SeedSequence(seed, spawn_key=key).spawn(2)
    return np.random.default_rng(data_ss), np.random.default_rng(fit_ss)


def _fit(
    method: str, model: str, data: np.ndarray, priors: Priors, K: int, cfg: dict, rng: np.random.Generator
) -> tuple[np.ndarray, float, np.ndarray | None, int, list[float]]:
    """Fit one data set; return (theta, mu, weights, iterations, trace)."""
    if method == "em":
        em = cfg["em"]
        if model == "known":
            r = EMKnownGroups(priors, em["max_iter"], em["tol"]).fit(data, rng)
        else:
            r = EMMixture(priors, K, **em).fit(data, rng)
        return np.asarray(r.theta), r.mu, r.weights, r.iterations, r.log_posterior

    from .vi import build_known_groups_model, build_mixture_model, fit_advi

    # Start q where EM starts, so both methods get the same initial values: for the mixture
    # the first draw from ``rng`` picks the same random data points as EMMixture.
    vi = cfg["vi"]
    if model == "known":
        pm_model, max_iter = build_known_groups_model(data, priors), vi["max_iter_known"]
        start = {"theta": data.mean(axis=0), "mu": float(data.mean())}
    else:
        pm_model, max_iter = build_mixture_model(data, priors, K), vi["max_iter_mixture"]
        start = {"theta": rng.choice(data, size=K, replace=len(data) < K), "mu": float(data.mean())}
    v = fit_advi(
        pm_model,
        max_iter=max_iter,
        learning_rate=vi["learning_rate"],
        start=start,
        n_samples=vi["n_samples"],
        seed=int(rng.integers(2**31)),
    )
    trace = [float(e) for e in v.elbo[:: cfg["trace_thin"]]]
    return np.asarray(v.theta), v.mu, v.weights, v.iterations, trace


def run_setting(method: str, scenario: str, model: str, setting_idx: int, setting: dict, cfg: dict) -> dict:
    priors = Priors.from_config(setting["variances"], setting["hyperpriors"])
    K = setting["num_groups"]
    gen_kwargs = dict(
        num_groups=K,
        num_obs_per_group=setting["num_obs_per_group"],
        mu=setting["true_mu"],
        **setting["variances"],
        **setting["hyperpriors"],
    )
    out: dict[str, list] = {
        k: []
        for k in [
            "true_theta",
            "est_theta",
            "true_mu",
            "est_mu",
            "theta_mse",
            "mu_mse",
            "true_weights",
            "est_weights",
            "weights_mse",
            "iterations",
            "seconds",
            "trace",
        ]
    }

    for rep in range(cfg["num_sims"]):
        data_rng, fit_rng = _rngs(cfg["seed"], scenario, model, setting_idx, rep)
        if model == "known":
            known = sample_known_groups(data_rng, **gen_kwargs)
            data, theta_true, mu_true = known.X, known.theta, known.mu
        else:
            mix = sample_mixture(data_rng, **gen_kwargs)
            data, theta_true, mu_true = mix.X, mix.theta, mix.mu

        start = time.perf_counter()
        theta_est, mu_est, w_est, iterations, trace = _fit(method, model, data, priors, K, cfg, fit_rng)
        seconds = time.perf_counter() - start

        if model == "mixture":
            # Components are only identified up to permutation: match them by
            # sorting true and estimated means separately.
            assert w_est is not None
            i, j = np.argsort(theta_true), np.argsort(theta_est)
            theta_true, theta_est, w_true, w_est = theta_true[i], theta_est[j], mix.weights[i], w_est[j]
            out["true_weights"].append(w_true.tolist())
            out["est_weights"].append(w_est.tolist())
            out["weights_mse"].append(float(np.mean((w_true - w_est) ** 2)))

        out["true_theta"].append(theta_true.tolist())
        out["est_theta"].append(theta_est.tolist())
        out["true_mu"].append(mu_true)
        out["est_mu"].append(float(mu_est))
        out["theta_mse"].append(float(np.mean((theta_true - theta_est) ** 2)))
        out["mu_mse"].append(float((mu_true - mu_est) ** 2))
        out["iterations"].append(int(iterations))
        out["seconds"].append(seconds)
        out["trace"].append([float(v) for v in trace])
    return out


def run(method: str, cfg: dict, out_dir: Path, scenarios=None, workers: int | None = None) -> None:
    out_dir = Path(out_dir) / method
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, scenario in cfg["scenarios"].items():
        if scenarios and name not in scenarios:
            continue
        for model in scenario["models"]:
            print(f"[{method}] {name} / {model} ...", flush=True)
            labels = list(scenario["settings"])
            jobs = [
                (method, name, model, i, resolve_setting(scenario, scenario["settings"][lab]), cfg)
                for i, lab in enumerate(labels)
            ]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(run_setting, *zip(*jobs, strict=False)))
            path = out_dir / f"{name}_{model}.json"
            path.write_text(json.dumps(dict(zip(labels, results, strict=False))))
            print(f"    -> {path}")
