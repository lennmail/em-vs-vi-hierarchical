"""Turn results/{em,vi}/*.json into figures/.

For every scenario and model:
  errors_<scenario>_<model>.png       squared error of each parameter, EM vs VI on the same data
  convergence_<scenario>_<model>.png  EM: distance to the final log posterior; VI: ELBO per observation
Overall:
  agreement.png                       EM estimate vs VI estimate of every theta_k and mu
  runtime.png                         wall-clock seconds per fit
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from emvi.simulation import resolve_setting

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
CFG = json.loads((ROOT / "configs" / "simulation.json").read_text())

METHODS = {"em": "EM", "vi": "VI"}
COLORS = dict(zip(METHODS.values(), sns.color_palette("colorblind", 2), strict=True))
MODEL_TITLES = {"known": "groups known", "mixture": "groups unknown (mixture)"}
PARAMS = {
    "theta_mse": r"$\theta$ (group means)",
    "mu_mse": r"$\mu$ (population mean)",
    "weights_mse": r"$\pi$ (mixture weights)",
}
DPI = 150

sns.set_theme(style="whitegrid", font_scale=0.95)


def load(method: str, name: str) -> dict | None:
    path = RESULTS / method / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def total_obs(scenario: dict, setting: str) -> int:
    s = resolve_setting(scenario, scenario["settings"][setting])
    n = s["num_obs_per_group"]
    return int(np.sum(n)) if isinstance(n, list) else int(n) * s["num_groups"]


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def errors(scenario_name: str, scenario: dict, model: str) -> None:
    name = f"{scenario_name}_{model}"
    rows = []
    for key, label in METHODS.items():
        res = load(key, name)
        if res is None:
            continue
        for setting, r in res.items():
            for param in PARAMS:
                for v in r[param]:
                    rows.append({"setting": setting, "Method": label, "param": param, "error": v})
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Squared errors of exactly 0 cannot be shown on a log axis. This only happens for the
    # mixture weights with a single group, where pi = 1 is known in advance.
    df = df[df["error"] > 0]
    params = [p for p in PARAMS if (df["param"] == p).any()]
    order = list(scenario["settings"])

    fig, axes = plt.subplots(1, len(params), figsize=(5.2 * len(params), 4.2), squeeze=False)
    for ax, param in zip(axes[0], params, strict=True):
        sns.boxplot(
            data=df[df["param"] == param],
            x="setting",
            y="error",
            hue="Method",
            order=order,
            hue_order=list(METHODS.values()),
            palette=COLORS,
            log_scale=True,
            fliersize=2,
            ax=ax,
        )
        ax.set(title=PARAMS[param], xlabel=scenario["xlabel"], ylabel="Squared error (log scale)")
        if ax is not axes[0][0]:
            ax.get_legend().remove()
    fig.suptitle(f"{scenario['description']}  [{MODEL_TITLES[model]}]", fontsize=11)
    fig.tight_layout()
    save(fig, f"errors_{name}")


def padded(traces: list[list[float]]) -> np.ndarray:
    """Stack traces of different lengths into an array padded with NaN."""
    arr = np.full((len(traces), max(len(t) for t in traces)), np.nan)
    for i, t in enumerate(traces):
        arr[i, : len(t)] = t
    return arr


def convergence(scenario_name: str, scenario: dict, model: str) -> None:
    name = f"{scenario_name}_{model}"
    em, vi = load("em", name), load("vi", name)
    if em is None and vi is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    palette = sns.color_palette("viridis", len(scenario["settings"]))

    if em is not None:
        # Runs stop at different iterations, so averaging traces would mix fast and slow runs.
        # Instead show how many iterations each run needed, and how many hit the cap.
        ax = axes[0]
        max_iter = CFG["em"]["max_iter"]
        df = pd.DataFrame([{"setting": s, "iterations": it} for s, r in em.items() for it in r["iterations"]])
        order = list(scenario["settings"])
        sns.stripplot(
            data=df,
            x="setting",
            y="iterations",
            order=order,
            hue="setting",
            hue_order=order,
            palette=palette,
            size=3,
            alpha=0.6,
            jitter=0.25,
            legend=False,
            ax=ax,
        )
        ax.axhline(max_iter, color="black", ls="--", lw=1)
        ax.set(
            yscale="log",
            ylim=(1, max_iter * 3),
            title=f"EM: iterations until |Δ log posterior| < {CFG['em']['tol']:g}",
            xlabel=scenario["xlabel"],
            ylabel="Iterations (log scale)",
        )
        for i, s in enumerate(order):
            capped = np.mean(np.asarray(em[s]["iterations"]) >= max_iter)
            ax.text(i, max_iter * 1.6, f"{capped:.0%} hit cap", ha="center", fontsize=8)

    if vi is not None:
        ax = axes[1]
        step = CFG["trace_thin"]
        # ADVI's ELBO is a noisy Monte Carlo estimate: take the median over replicates, then a
        # centred moving average (a trailing one would lag behind the steep initial rise).
        window = max(1, 250 // step)
        bottoms = []
        for color, (setting, r) in zip(palette, vi.items(), strict=True):
            elbo = padded(r["trace"]) / total_obs(scenario, setting)
            med = pd.Series(np.nanmedian(elbo, axis=0))
            med = med.rolling(window, center=True, min_periods=1).mean()
            ax.plot(np.arange(len(med)) * step, med, color=color, label=setting)
            bottoms.append(np.nanpercentile(med, 2))
        # The first few iterations start far below and would squash the y-axis.
        lo, hi = min(bottoms), max(line.get_ydata().max() for line in ax.get_lines())
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set(
            title="VI: ELBO per observation",
            xlabel="Iteration",
            ylabel=f"ELBO / N (median, smoothed over {window * step} it.)",
        )
        ax.legend(title=scenario["xlabel"], fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.suptitle(f"Convergence: {scenario['description']}  [{MODEL_TITLES[model]}]", fontsize=11)
    fig.tight_layout()
    save(fig, f"convergence_{name}")


def agreement() -> None:
    """EM and VI estimates for the same data sets, pooled over all scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, model in zip(axes, ["known", "mixture"], strict=True):
        em_vals, vi_vals = [], []
        for scenario_name, scenario in CFG["scenarios"].items():
            if model not in scenario["models"]:
                continue
            em, vi = load("em", f"{scenario_name}_{model}"), load("vi", f"{scenario_name}_{model}")
            if em is None or vi is None:
                continue
            for setting in scenario["settings"]:
                for a, b in zip(em[setting]["est_theta"], vi[setting]["est_theta"], strict=True):
                    em_vals += a
                    vi_vals += b
        if not em_vals:
            continue
        x, y = np.asarray(em_vals), np.asarray(vi_vals)
        lim = np.percentile(np.r_[x, y], [0.5, 99.5])
        ax.scatter(x, y, s=4, alpha=0.3, color=COLORS["EM"], edgecolor="none")
        ax.plot(lim, lim, color="black", lw=1, ls="--", label="EM = VI")
        ax.set(
            xlim=lim,
            ylim=lim,
            aspect="equal",
            xlabel=r"EM estimate of $\theta_k$",
            ylabel=r"VI estimate of $\theta_k$",
            title=f"{MODEL_TITLES[model]}\nmedian |EM − VI| = {np.median(np.abs(x - y)):.3g}",
        )
        ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "agreement")


def runtime() -> None:
    rows = []
    for scenario_name, scenario in CFG["scenarios"].items():
        for model in scenario["models"]:
            for key, label in METHODS.items():
                res = load(key, f"{scenario_name}_{model}")
                if res is None:
                    continue
                for r in res.values():
                    for s in r.get("seconds", []):
                        rows.append({"Method": label, "model": MODEL_TITLES[model], "seconds": s})
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=pd.DataFrame(rows),
        x="model",
        y="seconds",
        hue="Method",
        hue_order=list(METHODS.values()),
        palette=COLORS,
        log_scale=True,
        fliersize=2,
        ax=ax,
    )
    ax.set(title="Wall-clock time per fit (all scenarios)", xlabel="", ylabel="Seconds (log scale)")
    fig.tight_layout()
    save(fig, "runtime")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    for scenario_name, scenario in CFG["scenarios"].items():
        for model in scenario["models"]:
            errors(scenario_name, scenario, model)
            convergence(scenario_name, scenario, model)
    agreement()
    runtime()
    print(f"Figures written to {FIGURES}")
