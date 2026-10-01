"""Overlay plots for the A2C report (one figure per question)."""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = "logs"
OUT = "report/figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 0.9})


def load(name, seed):
    rows = [json.loads(l) for l in open(f"{LOG}/{name}__s{seed}.jsonl")]
    train = [r for r in rows if "losses/policy_loss" in r]
    ev = [r for r in rows if "eval/mean_return" in r]
    return train, (ev[-1] if ev else None)


def series(train, key):
    x = np.array([r["_step"] for r in train if r.get(key) is not None])
    y = np.array([r[key] for r in train if r.get(key) is not None], dtype=float)
    return x, y


def smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")


COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
STYLES = {1: "-", 2: "--"}


def rolling_std(y, k=20):
    return np.array([np.std(y[max(0, i - k):i + 1]) for i in range(len(y))])


def fig(question, runs, keys, titles, smooth_k=None, ylims=None, logy=None, symlog=None):
    n = len(keys)
    f, axes = plt.subplots(1, n, figsize=(7.2, 2.05))
    for ai, (ax, key, title) in enumerate(zip(axes, keys, titles)):
        for ci, (label, name) in enumerate(runs):
            for seed in (1, 2):
                if not os.path.exists(f"{LOG}/{name}__s{seed}.jsonl"):
                    continue
                train, _ = load(name, seed)
                if key.endswith(":rstd"):
                    x, y = series(train, key[:-5])
                    y = rolling_std(y)
                else:
                    x, y = series(train, key)
                k = (smooth_k or {}).get(key, 1)
                ax.plot(x / 1e3, smooth(y, k), STYLES[seed], color=COLORS[ci],
                        alpha=0.9, label=f"{label} s{seed}")
        ax.set_title(title)
        ax.set_xlabel("passos (mil)")
        if ylims and key in ylims:
            ax.set_ylim(*ylims[key])
        if logy and key in logy:
            ax.set_yscale("log")
        if symlog and key in symlog:
            ax.set_yscale("symlog", linthresh=symlog[key])
    h, l = axes[0].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=min(len(l), 4), frameon=False)
    nrows = (len(l) + 3) // 4
    f.tight_layout(pad=0.4, rect=(0, 0.07 * nrows, 1, 1))
    f.savefig(f"{OUT}/{question}.png", dpi=220)
    plt.close(f)


fig("q1", [("num_envs=8 (base)", "baseline"), ("num_envs=1", "Q1_envs1"),
           ("num_envs=64", "Q1_envs64"), ("num_envs=1, n=40", "Q1_envs1_steps40")],
    ["charts/episodic_return_mean_last100", "losses/policy_loss", "losses/policy_loss:rstd", "charts/SPS"],
    ["episodic_return_mean_last100", "policy_loss (symlog)", "policy_loss: desvio móvel (20 logs)", "SPS"],
    symlog={"losses/policy_loss": 1}, logy={"losses/policy_loss:rstd"})

fig("q2", [("num_steps=5 (base)", "baseline"), ("num_steps=1", "Q2_steps1"),
           ("num_steps=32", "Q2_steps32"), ("num_steps=128", "Q2_steps128")],
    ["charts/episodic_return_mean_last100", "losses/value_loss", "losses/explained_variance"],
    ["episodic_return_mean_last100", "value_loss (log, média móvel 15)", "explained_variance (média móvel 15)"],
    smooth_k={"losses/value_loss": 15, "losses/explained_variance": 15},
    ylims={"losses/explained_variance": (-1, 1.02)}, logy={"losses/value_loss"})

fig("q3", [("ent_coef=0.01 (base)", "baseline"), ("ent_coef=0", "Q3_ent0"),
           ("ent_coef=0.1", "Q3_ent0.1")],
    ["charts/episodic_return_mean_last100", "losses/entropy"],
    ["episodic_return_mean_last100", "entropy"])

fig("q4", [("com baseline (base)", "baseline"), ("sem baseline", "Q4_nobaseline")],
    ["charts/episodic_return_mean_last100", "charts/advantage_std",
     "charts/advantage_mean", "losses/entropy"],
    ["episodic_return_mean_last100", "advantage_std (log, m.m. 15)", "advantage_mean (symlog, m.m. 15)", "entropy"],
    smooth_k={"charts/advantage_std": 15, "charts/advantage_mean": 15},
    logy={"charts/advantage_std"}, symlog={"charts/advantage_mean": 1})

# Summary table
rows = []
for f_ in sorted(glob.glob(f"{LOG}/*__s*.jsonl")):
    base = os.path.basename(f_)[:-6]
    name, seed = base.rsplit("__s", 1)
    train, ev = load(name, int(seed))
    def last(key, n=20):
        _, y = series(train, key)
        y = y[~np.isnan(y)]
        return float(np.mean(y[-n:])) if len(y) else float("nan")
    def mean(key):
        _, y = series(train, key)
        y = y[~np.isnan(y)]
        return float(np.mean(y)) if len(y) else float("nan")
    x, r = series(train, "charts/episodic_return_mean_last100")
    first450 = next((int(xx) for xx, rr in zip(x, r) if rr >= 450), None)
    rows.append({
        "run": name, "seed": int(seed),
        "ret_final": last("charts/episodic_return_mean_last100", 1),
        "ret_max": float(np.max(r)) if len(r) else float("nan"),
        "step_450": first450,
        "eval": ev["eval/mean_return"] if ev else None,
        "sps": last("charts/SPS", 1),
        "pl_std": float(np.std(series(train, "losses/policy_loss")[1])),
        "ev_mean": mean("losses/explained_variance"),
        "vl_mean": mean("losses/value_loss"),
        "adv_std_mean": mean("charts/advantage_std"),
        "adv_mean_mean": mean("charts/advantage_mean"),
        "ent_final": last("losses/entropy"),
        "ent_100k": float(np.interp(1e5, *series(train, "losses/entropy"))),
    })
json.dump(rows, open("report/summary.json", "w"), indent=1)
for r in rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
