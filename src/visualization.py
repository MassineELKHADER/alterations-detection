from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .detector import DetectionResult
from .experiments import PreparedPair


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def displayable_rgb(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.size == 0:
        return image
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max <= 1.0 and image_min >= 0.0:
        return image
    if image_max - image_min < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return (image - image_min) / (image_max - image_min)


def binary_mask_image(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 255, 0).astype(np.uint8)


def save_cluster_mask(mask: np.ndarray, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    plt.imsave(out_path, binary_mask_image(mask), cmap="gray", vmin=0, vmax=255)


def save_cluster_gallery(result: DetectionResult, pair: PreparedPair, out_path: Path) -> None:
    ensure_dir(out_path.parent)

    num_clusters = len(result.selected_clusters)
    columns = max(2, num_clusters + 2)
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))

    axes = np.atleast_1d(axes)
    axes[0].imshow(displayable_rgb(pair.target))
    axes[0].set_title("Current image")
    axes[0].axis("off")

    if num_clusters == 0:
        axes[1].imshow(np.zeros(pair.diff_map.shape, dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("No selected cluster")
        axes[1].axis("off")
        for axis in axes[2:]:
            axis.axis("off")
    else:
        union_mask = np.zeros(pair.diff_map.shape, dtype=bool)
        for cluster in result.selected_clusters:
            union_mask |= result.cluster_mask(cluster)
        axes[1].imshow(binary_mask_image(union_mask), cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Union mask")
        axes[1].axis("off")

        for idx, cluster in enumerate(result.selected_clusters, start=2):
            mask = result.cluster_mask(cluster)
            significance = cluster.significance
            axes[idx].imshow(binary_mask_image(mask), cmap="gray", vmin=0, vmax=255)
            axes[idx].set_title(f"Cluster {idx - 1}\nscore={significance:.2f}")
            axes[idx].axis("off")

        for axis in axes[num_clusters + 2:]:
            axis.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_cluster_overlay_panel(result: DetectionResult, pair: PreparedPair, out_path: Path) -> None:
    ensure_dir(out_path.parent)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = np.atleast_1d(axes)

    axes[0].imshow(displayable_rgb(pair.reference))
    axes[0].set_title("Reference")
    axes[0].axis("off")

    axes[1].imshow(displayable_rgb(pair.target))
    axes[1].set_title("Current")
    axes[1].axis("off")

    axes[2].imshow(displayable_rgb(pair.target))
    axes[2].set_title("Clusters overlay")
    axes[2].axis("off")

    colors = [
        (31 / 255.0, 119 / 255.0, 180 / 255.0),
        (214 / 255.0, 39 / 255.0, 40 / 255.0),
        (44 / 255.0, 160 / 255.0, 44 / 255.0),
        (255 / 255.0, 127 / 255.0, 14 / 255.0),
        (148 / 255.0, 103 / 255.0, 189 / 255.0),
        (140 / 255.0, 86 / 255.0, 75 / 255.0),
    ]

    for index, cluster in enumerate(result.selected_clusters, start=1):
        color = colors[(index - 1) % len(colors)]
        mask = result.cluster_mask(cluster).astype(np.uint8)

        tinted = np.zeros((*mask.shape, 4), dtype=np.float32)
        tinted[..., :3] = color
        tinted[..., 3] = mask * 0.35
        axes[2].imshow(tinted)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour[:, 0, :]
            axes[2].plot(contour[:, 0], contour[:, 1], color=color, linewidth=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
