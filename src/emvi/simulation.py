"""Simulation study: run EM or VI over every scenario in the config file.

Every replicate gets its own seed derived from (scenario, setting, replicate),
so EM and VI see exactly the same data sets even when run separately.
"""

from __future__ import annotations

import json
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


def _match_labels(theta_true, theta_est, w_true, w_est):
    """Mixture components are only identified up to permutation: match them
    by sorting true and estimated means separately."""
    i, j = np.argsort(theta_true), np.argsort(theta_est)
    return theta_true[i], theta_est[j], w_true[i], (w_est[j] if w_est is not None else None)


def run_setting(method, scenario, model, setting_idx, setting, cfg) -> dict:
    priors = Priors.from_config(setting["variances"], setting["hyperpriors"])
    K = setting["num_groups"]
    gen_kwargs = dict(
        num_groups=K,
        num_obs_per_group=setting["num_obs_per_group"],
        mu=setting["true_mu"],
        **setting["variances"],
        **setting["hyperpriors"],
    )
    thin = cfg["trace_thin"]
    out = {k: [] for k in ["true_theta", "est_theta", "true_mu", "est_mu", "theta_mse", "mu_mse",
                           "true_weights", "est_weights", "weights_mse", "iterations", "trace"]}

    for rep in range(cfg["num_sims"]):
        data_rng, fit_rng = _rngs(cfg["seed"], scenario, model, setting_idx, rep)
        if model == "known":
            sample = sample_known_groups(data_rng, **gen_kwargs)
            data, w_true = sample.X, None
        else:
            sample = sample_mixture(data_rng, **gen_kwargs)
            data, w_true = sample.X, sample.weights

        if method == "em":
            est = (EMKnownGroups(priors, cfg["em"]["max_iter"], cfg["em"]["tol"]) if model == "known"
                   else EMMixture(priors, K, **cfg["em"])).fit(data, fit_rng)
            trace = est.log_posterior
        else:
            from .vi import build_known_groups_model, build_mixture_model, fit_advi

            vi = cfg["vi"]
            if model == "known":
                pm_model, max_iter = build_known_groups_model(data, priors), vi["max_iter_known"]
            else:
                pm_model, max_iter = build_mixture_model(data, priors, K), vi["max_iter_mixture"]
            est = fit_advi(pm_model, max_iter=max_iter, tol=vi["tol"], n_samples=vi["n_samples"],
                           seed=int(fit_rng.integers(2**31)))
            trace = est.elbo[::thin]

        theta_true, theta_est, w_est = sample.theta, np.asarray(est.theta), est.weights
        if model == "mixture":
            theta_true, theta_est, w_true, w_est = _match_labels(theta_true, theta_est, w_true, w_est)

        out["true_theta"].append(theta_true.tolist())
        out["est_theta"].append(theta_est.tolist())
        out["true_mu"].append(sample.mu)
        out["est_mu"].append(float(est.mu))
        out["theta_mse"].append(float(np.mean((theta_true - theta_est) ** 2)))
        out["mu_mse"].append(float((sample.mu - est.mu) ** 2))
        if model == "mixture":
            out["true_weights"].append(w_true.tolist())
            out["est_weights"].append(w_est.tolist())
            out["weights_mse"].append(float(np.mean((w_true - w_est) ** 2)))
        out["iterations"].append(int(est.iterations))
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
            jobs = [(method, name, model, i, resolve_setting(scenario, scenario["settings"][lab]), cfg)
                    for i, lab in enumerate(labels)]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(run_setting, *zip(*jobs)))
            path = out_dir / f"{name}_{model}.json"
            path.write_text(json.dumps(dict(zip(labels, results))))
            print(f"    -> {path}")
