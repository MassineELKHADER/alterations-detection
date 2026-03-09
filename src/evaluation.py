from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from .detector import DetectionResult


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """ Loads a binary mask from the given path and resizes it to the specified shape if necessary."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def precision_recall_f1(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """ Computes the precision, recall and F1-score between the binary prediction and target masks."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = float(np.logical_and(prediction, target).sum())
    fp = float(np.logical_and(prediction, np.logical_not(target)).sum())
    fn = float(np.logical_and(np.logical_not(prediction), target).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def best_cluster_metrics(result: DetectionResult, target_mask: np.ndarray) -> dict[str, float]:
    """ Evaluates the selected clusters in the detection result against the target mask and returns the metrics of the best cluster."""
    rows = []
    for rank, cluster in enumerate(result.selected_clusters, start=1):
        metrics = precision_recall_f1(result.cluster_mask(cluster), target_mask)
        rows.append({"rank": rank, **metrics, "significance": cluster.significance})
    if not rows:
        return {"rank": -1, "precision": 0.0, "recall": 0.0, "f1": 0.0, "significance": float("-inf")}
    frame = pd.DataFrame(rows)
    best = frame.sort_values("f1", ascending=False).iloc[0]
    return {key: float(best[key]) for key in ["rank", "precision", "recall", "f1", "significance"]}
