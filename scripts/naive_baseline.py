"""
Naive threshold baseline: label every pixel with diff >= tau as "wear".

Runs the same two synthetic experiments as run_synthetic_study and compares
precision / recall / F1 against the NFA results already saved in outputs/synthetic/.

Usage:
    python scripts/naive_baseline.py
    python scripts/naive_baseline.py --config configs/synthetic.yaml --output outputs/synthetic
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_synthetic_config
from src.experiments import _synthetic_mask, _nakagami_like


def threshold_metrics(diff_map: np.ndarray, mask: np.ndarray, tau: float) -> dict:
    pred = diff_map >= tau
    tp = float((pred & mask).sum())
    fp = float((pred & ~mask).sum())
    fn = float((~pred & mask).sum())
    precision = tp / max(tp + fp, 1.0)
    recall    = tp / max(tp + fn, 1.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def run(config_path: Path, output_root: Path) -> None:
    cfg = load_synthetic_config(config_path)
    sim = cfg["simulation"]
    tau = cfg["detector"]["tau"]

    rng  = np.random.default_rng(0)
    mask = _synthetic_mask(sim["image_height"], sim["image_width"])
    reps = sim["repetitions"]

    omega_values = sim["experiment_one_background_omega_values"]
    shift_values = sim["experiment_two_foreground_shifts"]
    total = (len(omega_values) + len(shift_values)) * reps

    rows_one, rows_two = [], []

    with tqdm(total=total, desc="Naive baseline", unit="run") as pbar:
        for omega in omega_values:
            for rep in range(reps):
                pbar.set_postfix({"exp": "1/2", "omega": omega})
                bg = _nakagami_like(rng, sim["background_shape_mu"], omega, mask.shape)
                fg = _nakagami_like(rng, sim["foreground_shape_mu"], sim["foreground_omega"], mask.shape) + sim["foreground_base_shift"]
                diff = bg.copy(); diff[mask] = fg[mask]
                rows_one.append({"omega": omega, "repetition": rep, **threshold_metrics(diff, mask, tau)})
                pbar.update(1)

        for shift in shift_values:
            for rep in range(reps):
                pbar.set_postfix({"exp": "2/2", "shift": shift})
                bg = _nakagami_like(rng, sim["background_shape_mu"], 6.0, mask.shape)
                fg = _nakagami_like(rng, sim["foreground_shape_mu"], sim["foreground_omega"], mask.shape) + shift
                diff = bg.copy(); diff[mask] = fg[mask]
                rows_two.append({"shift": shift, "repetition": rep, **threshold_metrics(diff, mask, tau)})
                pbar.update(1)

    df1 = pd.DataFrame(rows_one)
    df2 = pd.DataFrame(rows_two)
    output_root.mkdir(parents=True, exist_ok=True)
    df1.to_csv(output_root / "baseline_experiment_one.csv", index=False)
    df2.to_csv(output_root / "baseline_experiment_two.csv", index=False)

    # ── load NFA results for comparison ──────────────────────────────────────
    nfa_one_path = output_root / "experiment_one.csv"
    nfa_two_path = output_root / "experiment_two.csv"
    has_nfa = nfa_one_path.exists() and nfa_two_path.exists()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics_to_plot = ["f1", "precision", "recall"]
    styles = {"f1": "-", "precision": "--", "recall": ":"}
    colors_nfa      = {"f1": "tab:blue",   "precision": "tab:cyan",   "recall": "tab:steelblue"}
    colors_baseline = {"f1": "tab:orange", "precision": "tab:red",    "recall": "tab:brown"}

    for ax, (df_base, x_col, xlabel, title) in zip(axes, [
        (df1, "omega", "Background spread (omega)", "Experiment 1"),
        (df2, "shift", "Foreground shift",          "Experiment 2"),
    ]):
        grouped = df_base.groupby(x_col)
        for metric in metrics_to_plot:
            mean = grouped[metric].mean()
            ax.plot(mean.index, mean.values,
                    linestyle=styles[metric], color=colors_baseline[metric],
                    label=f"Baseline {metric}")

        if has_nfa:
            nfa_df = pd.read_csv(nfa_one_path if x_col == "omega" else nfa_two_path)
            nfa_grouped = nfa_df.groupby(x_col)
            for metric in metrics_to_plot:
                mean = nfa_grouped[metric].mean()
                ax.plot(mean.index, mean.values,
                        linestyle=styles[metric], color=colors_nfa[metric],
                        label=f"NFA {metric}")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Naive threshold (tau={tau}) vs NFA single-linkage", fontsize=11)
    fig.tight_layout()
    out_fig = output_root / "baseline_vs_nfa.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved figure to {out_fig}")
    print(f"Baseline CSVs: {output_root / 'baseline_experiment_one.csv'}")

    # ── print summary table ───────────────────────────────────────────────────
    print("\n── Experiment 1 summary (mean over repetitions) ──")
    print(df1.groupby("omega")[["precision","recall","f1"]].mean().round(3).to_string())
    print("\n── Experiment 2 summary (mean over repetitions) ──")
    print(df2.groupby("shift")[["precision","recall","f1"]].mean().round(3).to_string())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    p.add_argument("--output", type=Path, default=Path("outputs/synthetic"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config, args.output)
