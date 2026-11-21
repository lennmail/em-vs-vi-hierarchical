"""Turn results/{em,vi}/*.json into figures/ (MSE boxplots + convergence curves)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
CFG = json.loads((ROOT / "configs" / "simulation.json").read_text())
PARAM_LABELS = {"theta_mse": r"$\theta$", "mu_mse": r"$\mu$"}


def load(method, name):
    path = RESULTS / method / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def mse_frame(res):
    rows = [
        {"setting": setting, "Parameter": label, "MSE": v}
        for setting, r in res.items()
        for key, label in PARAM_LABELS.items()
        for v in r[key]
    ]
    return pd.DataFrame(rows)


def mse_boxplots(name, xlabel):
    em, vi = load("em", name), load("vi", name)
    panels = [(t, r) for t, r in [("EM", em), ("VI", vi)] if r is not None]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), sharey=True, squeeze=False)
    for ax, (title, res) in zip(axes[0], panels, strict=False):
        sns.boxplot(data=mse_frame(res), x="setting", y="MSE", hue="Parameter", ax=ax)
        ax.set(title=title, xlabel=xlabel, yscale="log")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"mse_{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def padded_mean(traces):
    """Mean of traces with different lengths (runs may stop early)."""
    L = max(len(t) for t in traces)
    arr = np.full((len(traces), L), np.nan)
    for i, t in enumerate(traces):
        arr[i, : len(t)] = t
    return np.nanmean(arr, axis=0)


def convergence(name):
    for method, ylabel, step in [("em", "Log posterior", 1), ("vi", "ELBO", CFG["trace_thin"])]:
        res = load(method, name)
        if res is None:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for setting, r in res.items():
            m = padded_mean(r["trace"])
            ax.plot(np.arange(len(m)) * step, m, label=setting)
        ax.set(title=f"{method.upper()}: {name}", xlabel="Iteration", ylabel=f"Mean {ylabel}")
        ax.set_yscale("symlog")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(FIGURES / f"convergence_{method}_{name}.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    for scenario_name, scenario in CFG["scenarios"].items():
        for model in scenario["models"]:
            name = f"{scenario_name}_{model}"
            mse_boxplots(name, scenario["xlabel"])
            convergence(name)
    print(f"Figures written to {FIGURES}")
