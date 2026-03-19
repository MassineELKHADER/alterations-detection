"""
EDA visualizations for WoodSample1 real data.

Produces and saves to figures/EDA/:
  1. reference_vs_current.png   — side-by-side reference / aligned+corrected image
  2. diff_map.png               — CIEDE2000 difference map ΔI (heatmap)
  3. thresholded_candidates.png — pixels with ΔI > tau overlaid on the current image
  4. point_cloud_3d.png         — 3D scatter: tanh transform (left) vs inverse transform (right)

Usage:
    python scripts/visualize_eda.py
    python scripts/visualize_eda.py --ref data/WoodSample1/IMG/t0_1.png \
                                    --cur data/WoodSample1/IMG/t10_1.png \
                                    --tau 3.0 --output figures/EDA
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    register_to_reference,
    match_illumination,
    color_difference_map,
)
from src.model_numba import apply_gray_transform


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_rgb(path: Path) -> np.ndarray:
    img = imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def _prepare(ref_path: Path, cur_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, register and illumination-correct; return (ref, corrected_current, diff_map)."""
    ref = _load_rgb(ref_path)
    cur = _load_rgb(cur_path)
    reg = register_to_reference(ref, cur)
    corrected = match_illumination(ref, reg.aligned)
    diff = color_difference_map(ref, corrected)
    return ref, corrected, diff


# ──────────────────────────────────────────────────────────────────────────────
# individual figure functions
# ──────────────────────────────────────────────────────────────────────────────

def fig_reference_vs_current(ref: np.ndarray, cur: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(ref)
    axes[0].set_title("Reference image (t0)", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(cur)
    axes[1].set_title("Current image (aligned + illumination-corrected)", fontsize=12)
    axes[1].axis("off")
    fig.suptitle("WoodSample1 — Reference vs Current", fontsize=13)
    fig.tight_layout()
    out = out_dir / "reference_vs_current.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def fig_diff_map(diff: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(diff, cmap="hot")
    fig.colorbar(im, ax=ax, label="CIEDE2000 ΔE")
    ax.set_title("Difference map ΔI (CIEDE2000)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    out = out_dir / "diff_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def fig_thresholded_candidates(cur: np.ndarray, diff: np.ndarray, tau: float, out_dir: Path) -> None:
    mask = diff >= tau  # candidate pixels

    # overlay: dim background, highlight candidates in red
    overlay = cur.copy().astype(np.float32) / 255.0
    overlay[mask] = [1.0, 0.15, 0.15]  # bright red for candidates

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(cur)
    axes[0].set_title("Current image", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Thresholded candidates (ΔI ≥ τ={tau:.1f})  —  {mask.sum()} px  "
        f"({100 * mask.mean():.1f}% of image)",
        fontsize=10,
    )
    axes[1].axis("off")
    fig.suptitle("Candidate alteration pixels", fontsize=13)
    fig.tight_layout()
    out = out_dir / "thresholded_candidates.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def fig_point_cloud_3d(diff: np.ndarray, tau: float, out_dir: Path) -> None:
    """Side-by-side 3D scatter: tanh (left) vs inverse (right)."""
    transforms = [
        ("tanh",    "tanh transform  f(x) = 1 + tanh(τ−x)",  "tab:blue"),
        ("inverse", "inverse transform  f(x) = 1/(x−τ)",     "tab:orange"),
    ]

    fig = plt.figure(figsize=(16, 7))

    for col, (kind, subtitle, color) in enumerate(transforms):
        z_vals = apply_gray_transform(diff, tau=tau, kind=kind)
        finite = np.isfinite(z_vals)
        ys, xs = np.nonzero(finite)
        z = z_vals[finite]

        # sub-sample for readability (at most 5000 points)
        if len(z) > 5000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(z), 5000, replace=False)
            ys, xs, z = ys[idx], xs[idx], z[idx]

        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        sc = ax.scatter(xs, ys, z, c=z, cmap="plasma", s=2, alpha=0.6)
        fig.colorbar(sc, ax=ax, label="f(ΔI)", shrink=0.6, pad=0.1)

        ax.set_xlabel("x (col)", fontsize=8)
        ax.set_ylabel("y (row)", fontsize=8)
        ax.set_zlabel("f(ΔI)", fontsize=8)
        ax.set_title(subtitle, fontsize=10)
        ax.invert_yaxis()  # image convention: row 0 at top

        n_total = int(finite.sum())
        ax.text2D(
            0.02, 0.95,
            f"{n_total} candidates\n(showing {len(z)})",
            transform=ax.transAxes,
            fontsize=8,
            color="black",
        )

    fig.suptitle(
        f"3D candidate point cloud — WoodSample1  (τ={tau:.1f})\n"
        "z-axis = gray-level transform applied to ΔI",
        fontsize=12,
    )
    fig.tight_layout()
    out = out_dir / "point_cloud_3d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def run(ref_path: Path, cur_path: Path, tau: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading images …\n  ref : {ref_path}\n  cur : {cur_path}")
    ref, cur, diff = _prepare(ref_path, cur_path)
    print(f"  image shape : {ref.shape}  |  ΔI range : [{diff.min():.2f}, {diff.max():.2f}]  |  τ = {tau}")

    print("\nGenerating figures …")
    fig_reference_vs_current(ref, cur, out_dir)
    fig_diff_map(diff, out_dir)
    fig_thresholded_candidates(cur, diff, tau, out_dir)
    fig_point_cloud_3d(diff, tau, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref",    type=Path, default=Path("data/WoodSample1/IMG/t0_1.png"))
    p.add_argument("--cur",    type=Path, default=Path("data/WoodSample1/IMG/t10_1.png"))
    p.add_argument("--tau",    type=float, default=3.0)
    p.add_argument("--output", type=Path, default=Path("figures/EDA"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.ref, args.cur, args.tau, args.output)
