"""
Visualize NFA single-linkage clustering results on one synthetic image.

Produces a 5-panel figure:
  1. Diff map (raw values)
  2. Ground truth mask
  3. All candidate points entering the algorithm (after tau filtering)
  4. Selected clusters overlaid on diff map (each cluster in a different color)
  5. Precision / recall breakdown per selected cluster

Usage:
    python scripts/visualize_synthetic.py
    python scripts/visualize_synthetic.py --omega 6.0 --shift 4.0 --seed 0
    python scripts/visualize_synthetic.py --output outputs/synthetic/viz.png
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_synthetic_config, DetectorConfig
from src.detector import AContrarioDetector
from src.experiments import _synthetic_mask, _nakagami_like


CLUSTER_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    p.add_argument("--omega", type=float, default=2.0, help="Background spread (Experiment 1 scenario)")
    p.add_argument("--shift", type=float, default=4.0, help="Foreground shift (Experiment 2 scenario)")
    p.add_argument("--seed",  type=int,   default=0)
    p.add_argument("--output", type=Path, default=Path("outputs/synthetic/visualization.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_synthetic_config(args.config)
    sim = cfg["simulation"]
    detector_cfg = DetectorConfig(**cfg["detector"])

    rng = np.random.default_rng(args.seed)
    mask = _synthetic_mask(sim["image_height"], sim["image_width"])

    background = _nakagami_like(rng, sim["background_shape_mu"], args.omega, mask.shape)
    foreground = _nakagami_like(rng, sim["foreground_shape_mu"], sim["foreground_omega"], mask.shape) + args.shift
    diff_map = background.copy()
    diff_map[mask] = foreground[mask]

    detector = AContrarioDetector(detector_cfg)
    result = detector.run(diff_map)

    # ── metrics per cluster ──────────────────────────────────────────────────
    def cluster_metrics(cluster):
        pred = result.cluster_mask(cluster)
        tp = float(np.logical_and(pred, mask).sum())
        fp = float(np.logical_and(pred, ~mask).sum())
        fn = float(np.logical_and(~pred, mask).sum())
        prec = tp / max(tp + fp, 1.0)
        rec  = tp / max(tp + fn, 1.0)
        f1   = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        return prec, rec, f1

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    # Panel 1 — diff map
    im = axes[0].imshow(diff_map, cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    axes[0].set_title("Diff map")
    axes[0].axis("off")

    # Panel 2 — ground truth
    axes[1].imshow(mask, cmap="gray", interpolation="nearest")
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    # Panel 3 — candidate points (all points surviving tau)
    candidate_img = np.zeros(diff_map.shape, dtype=np.float32)
    coords = result.point_cloud.coordinates  # (N, 2) — [row, col]
    if len(coords) > 0:
        candidate_img[coords[:, 0], coords[:, 1]] = 1.0
    axes[2].imshow(diff_map, cmap="gray", alpha=0.4, interpolation="nearest")
    axes[2].imshow(candidate_img, cmap="Reds", alpha=0.8, interpolation="nearest",
                   vmin=0, vmax=1)
    axes[2].set_title(f"Candidates (N={len(coords)})")
    axes[2].axis("off")

    # Panel 4 — selected clusters on diff map
    axes[3].imshow(diff_map, cmap="gray", interpolation="nearest")
    legend_patches = []
    if not result.selected_clusters:
        axes[3].set_title("No clusters selected")
    else:
        for i, cluster in enumerate(result.selected_clusters):
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            pred = result.cluster_mask(cluster)
            prec, rec, f1 = cluster_metrics(cluster)

            # Scatter the cluster points
            ys, xs = np.where(pred)
            axes[3].scatter(xs, ys, s=2, c=color, alpha=0.7, linewidths=0)

            label = f"C{i+1} S={cluster.significance:.1f}\nP={prec:.2f} R={rec:.2f} F1={f1:.2f}"
            legend_patches.append(mpatches.Patch(color=color, label=label))

        axes[3].set_title("Selected clusters")
        axes[3].legend(handles=legend_patches, loc="upper right", fontsize=6,
                       framealpha=0.8)
    axes[3].axis("off")

    # Panel 5 — GT vs union of all selected clusters
    axes[4].imshow(mask, cmap="Greens", alpha=0.5, interpolation="nearest")
    if result.selected_clusters:
        union = np.zeros(diff_map.shape, dtype=bool)
        for cluster in result.selected_clusters:
            union |= result.cluster_mask(cluster)
        tp_map = union & mask
        fp_map = union & ~mask
        fn_map = ~union & mask

        # TP=white, FP=red, FN=blue
        overlay = np.zeros((*diff_map.shape, 4), dtype=np.float32)
        overlay[tp_map] = [1, 1, 1, 0.9]
        overlay[fp_map] = [1, 0, 0, 0.7]
        overlay[fn_map] = [0, 0.4, 1, 0.5]
        axes[4].imshow(overlay, interpolation="nearest")

        tp = float(tp_map.sum())
        fp = float(fp_map.sum())
        fn = float(fn_map.sum())
        prec = tp / max(tp + fp, 1.0)
        rec  = tp / max(tp + fn, 1.0)
        f1   = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        axes[4].set_title(f"Union vs GT\nP={prec:.2f} R={rec:.2f} F1={f1:.2f}")

        legend = [
            mpatches.Patch(color="white",  label="TP"),
            mpatches.Patch(color="red",    label="FP"),
            mpatches.Patch(color=(0,.4,1), label="FN (missed)"),
        ]
        axes[4].legend(handles=legend, loc="upper right", fontsize=7, framealpha=0.8)
    else:
        axes[4].set_title("Union vs GT\n(no clusters)")
    axes[4].axis("off")

    fig.suptitle(
        f"NFA single-linkage | tau={detector_cfg.tau} max_points={detector_cfg.max_points} "
        f"z_levels={detector_cfg.quantization_levels} | omega={args.omega} shift={args.shift}",
        fontsize=9,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
