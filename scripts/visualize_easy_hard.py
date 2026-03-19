"""
Generates report-ready figures for the easy and hard synthetic cases:

  Easy : omega=2,  shift=4.0
  Hard : omega=9,  shift=1.5

Per case, saves:
  figures/report/<case>_full.png       — 5-panel figure (diff, GT, candidates, clusters, TP/FP/FN)
  figures/report/<case>_input.png      — 2-panel input view (GT mask + diff map)

Also prints a LaTeX table row with P / R / F1 for both cases.

Usage:
    python scripts/visualize_easy_hard.py
    python scripts/visualize_easy_hard.py --config configs/synthetic.yaml --seed 0
"""

import argparse
from pathlib import Path
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_synthetic_config, DetectorConfig
from src.detector import AContrarioDetector
from src.experiments import _synthetic_mask, _nakagami_like


CLUSTER_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]

CASES = [
    {"name": "easy", "omega": 2.0, "shift": 4.0},
    {"name": "hard", "omega": 9.0, "shift": 1.5},
]


# ─────────────────────────────────────────────────────────────────────────────

def cluster_metrics(pred: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.logical_and(pred, mask).sum())
    fp = float(np.logical_and(pred, ~mask).sum())
    fn = float(np.logical_and(~pred, mask).sum())
    prec = tp / max(tp + fp, 1.0)
    rec  = tp / max(tp + fn, 1.0)
    f1   = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def save_input_figure(diff_map: np.ndarray, mask: np.ndarray, case: dict,
                      det_cfg: DetectorConfig, out_path: Path) -> None:
    """2-panel: ground truth mask + annotated diff map."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # GT mask
    axes[0].imshow(mask, cmap="gray", interpolation="nearest")
    axes[0].set_title("Ground truth mask", fontsize=12)
    axes[0].axis("off")

    # Diff map with tau contour
    im = axes[1].imshow(diff_map, cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=axes[1], fraction=0.046, label="ΔI")
    # Overlay tau threshold as a contour
    axes[1].contour(diff_map >= det_cfg.tau, levels=[0.5],
                    colors=["cyan"], linewidths=[0.8])
    axes[1].set_title(f"Difference map ΔI  (cyan = τ={det_cfg.tau} boundary)", fontsize=11)
    axes[1].axis("off")

    label = "Easy" if case["name"] == "easy" else "Hard"
    fig.suptitle(
        f"{label} case  |  ω={case['omega']}  shift={case['shift']}  "
        f"(background spread / foreground contrast)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  input figure  → {out_path}")


def save_full_figure(diff_map: np.ndarray, mask: np.ndarray,
                     result, case: dict, det_cfg: DetectorConfig,
                     out_path: Path) -> tuple[float, float, float]:
    """5-panel figure; returns (precision, recall, f1) of the union."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.2))

    # 1 — diff map
    im = axes[0].imshow(diff_map, cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=axes[0], fraction=0.046, label="ΔI")
    axes[0].set_title("Diff map", fontsize=11)
    axes[0].axis("off")

    # 2 — GT
    axes[1].imshow(mask, cmap="gray", interpolation="nearest")
    axes[1].set_title("Ground truth", fontsize=11)
    axes[1].axis("off")

    # 3 — candidates
    coords = result.point_cloud.coordinates
    cand_img = np.zeros(diff_map.shape, dtype=np.float32)
    if len(coords) > 0:
        cand_img[coords[:, 0], coords[:, 1]] = 1.0
    axes[2].imshow(diff_map, cmap="gray", alpha=0.4, interpolation="nearest")
    axes[2].imshow(cand_img, cmap="Reds", alpha=0.8, interpolation="nearest", vmin=0, vmax=1)
    axes[2].set_title(f"Candidates (N={len(coords)})", fontsize=11)
    axes[2].axis("off")

    # 4 — selected clusters
    axes[3].imshow(diff_map, cmap="gray", interpolation="nearest")
    legend_patches = []
    if not result.selected_clusters:
        axes[3].set_title("No clusters selected", fontsize=11)
    else:
        for i, cluster in enumerate(result.selected_clusters):
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            pred = result.cluster_mask(cluster)
            prec, rec, f1 = cluster_metrics(pred, mask)
            ys, xs = np.where(pred)
            axes[3].scatter(xs, ys, s=2, c=color, alpha=0.7, linewidths=0)
            legend_patches.append(mpatches.Patch(
                color=color,
                label=f"C{i+1}  S={cluster.significance:.1f}\nP={prec:.2f} R={rec:.2f} F1={f1:.2f}",
            ))
        axes[3].set_title("Selected clusters", fontsize=11)
        axes[3].legend(handles=legend_patches, loc="upper right", fontsize=6, framealpha=0.85)
    axes[3].axis("off")

    # 5 — TP / FP / FN overlay
    axes[4].imshow(mask, cmap="Greens", alpha=0.3, interpolation="nearest")
    prec_u = rec_u = f1_u = 0.0
    if result.selected_clusters:
        union = np.zeros(diff_map.shape, dtype=bool)
        for cluster in result.selected_clusters:
            union |= result.cluster_mask(cluster)
        tp_map = union & mask
        fp_map = union & ~mask
        fn_map = ~union & mask

        overlay = np.zeros((*diff_map.shape, 4), dtype=np.float32)
        overlay[tp_map] = [1.0, 1.0, 1.0, 0.9]   # white  — TP
        overlay[fp_map] = [1.0, 0.0, 0.0, 0.75]  # red    — FP
        overlay[fn_map] = [0.0, 0.4, 1.0, 0.55]  # blue   — FN
        axes[4].imshow(overlay, interpolation="nearest")

        prec_u, rec_u, f1_u = cluster_metrics(union, mask)
        axes[4].set_title(f"Union vs GT\nP={prec_u:.2f}  R={rec_u:.2f}  F1={f1_u:.2f}", fontsize=11)
        axes[4].legend(handles=[
            mpatches.Patch(color="white",  label="TP"),
            mpatches.Patch(color="red",    label="FP"),
            mpatches.Patch(color=(0, .4, 1), label="FN (missed)"),
        ], loc="upper right", fontsize=7, framealpha=0.85)
    else:
        axes[4].set_title("Union vs GT\n(no clusters)", fontsize=11)
    axes[4].axis("off")

    label = "Easy" if case["name"] == "easy" else "Hard"
    fig.suptitle(
        f"{label} case  |  ω={case['omega']}  shift={case['shift']}  "
        f"|  τ={det_cfg.tau}  max_points={det_cfg.max_points}  z_levels={det_cfg.quantization_levels}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  full figure   → {out_path}")

    return prec_u, rec_u, f1_u


# ─────────────────────────────────────────────────────────────────────────────

def run(config_path: Path, seed: int, out_dir: Path) -> None:
    cfg = load_synthetic_config(config_path)
    sim = cfg["simulation"]
    det_cfg = DetectorConfig(**cfg["detector"])
    mask = _synthetic_mask(sim["image_height"], sim["image_width"])

    table_rows: list[dict] = []

    for case in CASES:
        print(f"\n{'='*60}")
        print(f"  Case: {case['name'].upper()}  (ω={case['omega']}, shift={case['shift']})")
        print(f"{'='*60}")

        rng = np.random.default_rng(seed)
        background = _nakagami_like(rng, sim["background_shape_mu"], case["omega"], mask.shape)
        foreground = _nakagami_like(
            rng, sim["foreground_shape_mu"], sim["foreground_omega"], mask.shape
        ) + case["shift"]
        diff_map = background.copy()
        diff_map[mask] = foreground[mask]

        detector = AContrarioDetector(det_cfg)
        result = detector.run(diff_map)

        save_input_figure(diff_map, mask, case, det_cfg,
                          out_dir / f"{case['name']}_input.png")

        prec, rec, f1 = save_full_figure(diff_map, mask, result, case, det_cfg,
                                          out_dir / f"{case['name']}_full.png")

        table_rows.append({"case": case["name"], "omega": case["omega"],
                           "shift": case["shift"],
                           "precision": prec, "recall": rec, "f1": f1})
        print(f"  Union metrics: P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  LaTeX table (copy into report)")
    print("="*60)
    print(r"\begin{table}[h]")
    print(r"  \centering")
    print(r"  \begin{tabular}{lcccc}")
    print(r"    \hline")
    print(r"    \textbf{Case} & $\omega$ & \textbf{shift} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\")
    print(r"    \hline")
    for row in table_rows:
        label = "Easy" if row["case"] == "easy" else "Hard"
        print(f"    {label} & {row['omega']:.0f} & {row['shift']:.1f} "
              f"& {row['precision']:.3f} & {row['recall']:.3f} & {row['f1']:.3f} \\\\")
    print(r"    \hline")
    print(r"  \end{tabular}")
    print(r"  \caption{NFA single-linkage performance on easy and hard synthetic cases (seed=" + str(seed) + r").}")
    print(r"  \label{tab:easy_hard}")
    print(r"\end{table}")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    p.add_argument("--seed",   type=int,  default=0)
    p.add_argument("--output", type=Path, default=Path("figures/report"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config, args.seed, args.output)
