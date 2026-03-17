"""
HDBSCAN comparison against NFA single-linkage on the synthetic experiments.

Uses the same candidate point cloud (same tau filtering) as the NFA method.
For each image, HDBSCAN clusters the 2D spatial coordinates of candidate points.
The best cluster (highest F1 vs ground truth) is selected — same protocol as the paper.

Usage:
    python scripts/hdbscan_comparison.py
    python scripts/hdbscan_comparison.py --config configs/synthetic.yaml --output outputs/synthetic
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

try:
    from hdbscan import HDBSCAN
except ImportError:
    raise SystemExit("hdbscan not installed. Run: pip install hdbscan")

from src.config import load_synthetic_config, DetectorConfig
from src.experiments import _synthetic_mask, _nakagami_like
from src.model_numba import apply_gray_transform, make_point_cloud


def hdbscan_best_metrics(
    point_cloud,
    mask: np.ndarray,
    min_cluster_size: int = 10,
) -> dict:
    """Run HDBSCAN on 2D spatial candidate coords, return best-cluster metrics."""
    coords = point_cloud.coordinates  # (N, 2) — [row, col]
    if len(coords) < min_cluster_size:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Cluster on 2D spatial coordinates (same fairness as NFA which also uses spatial + z)
    xy = coords[:, ::-1].astype(np.float32)  # (col, row) for HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5, core_dist_n_jobs=1)
    labels = clusterer.fit_predict(xy)

    unique_labels = [l for l in np.unique(labels) if l >= 0]
    if not unique_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for label in unique_labels:
        pred_mask = np.zeros(mask.shape, dtype=bool)
        member_coords = coords[labels == label]
        pred_mask[member_coords[:, 0], member_coords[:, 1]] = True

        tp = float((pred_mask & mask).sum())
        fp = float((pred_mask & ~mask).sum())
        fn = float((~pred_mask & mask).sum())
        prec = tp / max(tp + fp, 1.0)
        rec  = tp / max(tp + fn, 1.0)
        f1   = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        if f1 > best["f1"]:
            best = {"precision": prec, "recall": rec, "f1": f1}

    return best


def run(config_path: Path, output_root: Path) -> None:
    cfg = load_synthetic_config(config_path)
    sim = cfg["simulation"]
    det = DetectorConfig(**cfg["detector"])

    rng  = np.random.default_rng(0)
    mask = _synthetic_mask(sim["image_height"], sim["image_width"])
    reps = sim["repetitions"]

    omega_values = sim["experiment_one_background_omega_values"]
    shift_values = sim["experiment_two_foreground_shifts"]
    total = (len(omega_values) + len(shift_values)) * reps

    rows_one, rows_two = [], []

    with tqdm(total=total, desc="HDBSCAN comparison", unit="run") as pbar:
        for omega in omega_values:
            for rep in range(reps):
                pbar.set_postfix({"exp": "1/2", "omega": omega})
                bg = _nakagami_like(rng, sim["background_shape_mu"], omega, mask.shape)
                fg = _nakagami_like(rng, sim["foreground_shape_mu"], sim["foreground_omega"], mask.shape) + sim["foreground_base_shift"]
                diff = bg.copy(); diff[mask] = fg[mask]
                pc = make_point_cloud(diff, tau=det.tau, transform=det.transform,
                                      z_levels=det.quantization_levels, max_points=det.max_points)
                rows_one.append({"omega": omega, "repetition": rep,
                                  **hdbscan_best_metrics(pc, mask)})
                pbar.update(1)

        for shift in shift_values:
            for rep in range(reps):
                pbar.set_postfix({"exp": "2/2", "shift": shift})
                bg = _nakagami_like(rng, sim["background_shape_mu"], 6.0, mask.shape)
                fg = _nakagami_like(rng, sim["foreground_shape_mu"], sim["foreground_omega"], mask.shape) + shift
                diff = bg.copy(); diff[mask] = fg[mask]
                pc = make_point_cloud(diff, tau=det.tau, transform=det.transform,
                                      z_levels=det.quantization_levels, max_points=det.max_points)
                rows_two.append({"shift": shift, "repetition": rep,
                                  **hdbscan_best_metrics(pc, mask)})
                pbar.update(1)

    df1 = pd.DataFrame(rows_one)
    df2 = pd.DataFrame(rows_two)
    output_root.mkdir(parents=True, exist_ok=True)
    df1.to_csv(output_root / "hdbscan_experiment_one.csv", index=False)
    df2.to_csv(output_root / "hdbscan_experiment_two.csv", index=False)

    # ── load NFA results for comparison ──────────────────────────────────────
    nfa_one_path = output_root / "experiment_one.csv"
    nfa_two_path = output_root / "experiment_two.csv"
    has_nfa = nfa_one_path.exists() and nfa_two_path.exists()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    styles   = {"f1": "-", "precision": "--", "recall": ":"}
    colors_nfa     = {"f1": "tab:blue",   "precision": "tab:cyan",  "recall": "tab:steelblue"}
    colors_hdbscan = {"f1": "tab:orange", "precision": "tab:red",   "recall": "tab:brown"}

    for ax, (df_hdb, x_col, xlabel, title, nfa_path) in zip(axes, [
        (df1, "omega", "Background spread (omega)", "Experiment 1", nfa_one_path),
        (df2, "shift", "Foreground shift",          "Experiment 2", nfa_two_path),
    ]):
        grouped = df_hdb.groupby(x_col)
        for metric in ["f1", "precision", "recall"]:
            mean = grouped[metric].mean()
            ax.plot(mean.index, mean.values,
                    linestyle=styles[metric], color=colors_hdbscan[metric],
                    label=f"HDBSCAN {metric}")

        if has_nfa and nfa_path.exists():
            nfa_df = pd.read_csv(nfa_path)
            for metric in ["f1", "precision", "recall"]:
                mean = nfa_df.groupby(x_col)[metric].mean()
                ax.plot(mean.index, mean.values,
                        linestyle=styles[metric], color=colors_nfa[metric],
                        label=f"NFA {metric}")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("HDBSCAN (best cluster) vs NFA single-linkage — same candidate set", fontsize=11)
    fig.tight_layout()
    out_fig = output_root / "hdbscan_vs_nfa.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved figure to {out_fig}")

    print("\n── HDBSCAN Experiment 1 summary ──")
    print(df1.groupby("omega")[["precision","recall","f1"]].mean().round(3).to_string())
    print("\n── HDBSCAN Experiment 2 summary ──")
    print(df2.groupby("shift")[["precision","recall","f1"]].mean().round(3).to_string())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    p.add_argument("--output", type=Path, default=Path("outputs/synthetic"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config, args.output)
