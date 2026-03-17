from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DetectorConfig, PreprocessingConfig, SequenceConfig
from .datasets import UviflSequence, read_image
from .detector import AContrarioDetector, DetectionResult
from .evaluation import best_cluster_metrics, load_mask
from .preprocessing import (
    Roi,
    auto_roi_from_difference,
    choose_best_shot,
    color_difference_map,
    crop_roi,
    match_illumination,
    resize_max_side,
)


@dataclass(frozen=True)
class PreparedPair:
    reference: np.ndarray
    target: np.ndarray
    diff_map: np.ndarray
    roi: Roi
    chosen_shot: int


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_uvifl_pair(
    sequence_config: SequenceConfig,
    preprocessing: PreprocessingConfig,
    time: int,
) -> PreparedPair:
    """ 
    Prepares a pair of reference and target images for a given sequence and time point by performing the following steps:
        1. Loads the reference image and candidate images for the specified time point.
        2. Chooses the best candidate image based on registration quality with the reference.
        3. Optionally determines a region of interest (ROI) based on the color difference between the reference and aligned candidate image.
        4. Crops the reference and target images to the ROI and resizes them if necessary.
        5. Matches the illumination of the target image to the reference and computes the color difference map.
    
    """
    sequence = UviflSequence(sequence_config)
    reference = read_image(sequence.reference_frame().path)
    candidates = sequence.candidate_frames(time)
    candidate_images = [read_image(frame.path) for frame in candidates]
    best_index, best_registration = choose_best_shot(reference, candidate_images)
    aligned = best_registration.aligned

    if sequence_config.use_auto_roi:
        roi = auto_roi_from_difference(
            reference=reference,
            image=aligned,
            percentile=preprocessing.auto_roi_percentile,
            padding=preprocessing.auto_roi_padding,
            min_component_area=preprocessing.min_component_area,
        )
    else:
        roi = Roi(0, reference.shape[0], 0, reference.shape[1])

    ref_crop = crop_roi(reference, roi)
    tgt_crop = crop_roi(aligned, roi)
    ref_crop = resize_max_side(ref_crop, sequence_config.resize_max_side)
    tgt_crop = resize_max_side(tgt_crop, sequence_config.resize_max_side)
    tgt_crop = match_illumination(ref_crop, tgt_crop)
    diff_map = color_difference_map(ref_crop, tgt_crop)
    return PreparedPair(reference=ref_crop, target=tgt_crop, diff_map=diff_map, roi=roi, chosen_shot=candidates[best_index].shot)


def save_detection_panel(result: DetectionResult, pair: PreparedPair, out_path: Path) -> None:
    """ 
    Saves a visualization panel showing the reference image, target image and detected 
    clusters for a given detection result and prepared pair.
    The panel consists of three subplots:
        1. The reference image.
        2. The target image.
        3. The target image with the contours of the selected clusters overlaid."""
    _ensure_dir(out_path.parent)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(pair.reference)
    axes[0].set_title("Reference")
    axes[1].imshow(pair.target)
    axes[1].set_title("Current")
    axes[2].imshow(pair.target)
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
    for color, cluster in zip(colors, result.selected_clusters, strict=False):
        mask = result.cluster_mask(cluster).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour[:, 0, :]
            axes[2].plot(contour[:, 0], contour[:, 1], color=color, linewidth=2)
    axes[2].set_title("Top clusters")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_uvifl_timepoint(
    detector_config: DetectorConfig,
    preprocessing: PreprocessingConfig,
    sequence_config: SequenceConfig,
    time: int,
    output_root: Path,
) -> dict[str, float]:
    """ 
    Runs the alteration detection process for a specific time point in a given sequence and returns a summary of the results.
    """
    pair = prepare_uvifl_pair(sequence_config, preprocessing, time)
    detector = AContrarioDetector(detector_config)
    result = detector.run(pair.diff_map)
    panel_path = output_root / sequence_config.name / f"time_{time:02d}.png"
    save_detection_panel(result, pair, panel_path)

    summary = {
        "time": float(time),
        "shot": float(pair.chosen_shot),
        "roi_y0": float(pair.roi.y0),
        "roi_y1": float(pair.roi.y1),
        "roi_x0": float(pair.roi.x0),
        "roi_x1": float(pair.roi.x1),
        "num_candidates": float(len(result.candidates)),
        "num_selected": float(len(result.selected_clusters)),
    }

    annotation_path = Path("data/annotations") / sequence_config.name / f"t{time:02d}.png"
    if annotation_path.exists():
        metrics = best_cluster_metrics(result, load_mask(annotation_path, pair.diff_map.shape))
        summary.update(metrics)
    return summary


def run_uvifl_sequence(
    detector_config: DetectorConfig,
    preprocessing: PreprocessingConfig,
    sequence_config: SequenceConfig,
    times: list[int],
    output_root: Path,
) -> pd.DataFrame:
    rows = [run_uvifl_timepoint(detector_config, preprocessing, sequence_config, time, output_root) for time in times]
    table = pd.DataFrame(rows).sort_values("time")
    _ensure_dir(output_root / sequence_config.name)
    table.to_csv(output_root / sequence_config.name / "summary.csv", index=False)
    return table


def _synthetic_mask(height: int, width: int) -> np.ndarray:
    """ Generates a synthetic binary mask with two elliptical regions and a notch, simulating a realistic alteration pattern."""
    yy, xx = np.mgrid[:height, :width]
    left = ((xx - width * 0.35) / 20.0) ** 2 + ((yy - height * 0.5) / 28.0) ** 2 <= 1.0
    left &= yy > 20
    right = ((xx - width * 0.72) / 14.0) ** 2 + ((yy - height * 0.4) / 18.0) ** 2 <= 1.0
    notch = ((xx - width * 0.31) / 10.0) ** 2 + ((yy - height * 0.62) / 8.0) ** 2 <= 1.0
    return (left | right) & (~notch)


def _nakagami_like(rng: np.random.Generator, shape_mu: float, spread_omega: float, size: tuple[int, int]) -> np.ndarray:
    """ Generates a Nakagami-like random field by sampling from a gamma distribution and applying a square root transformation."""
    samples = rng.gamma(shape=shape_mu, scale=spread_omega / shape_mu, size=size)
    return np.sqrt(samples).astype(np.float32)


def _run_detector_on_diff(detector_config: DetectorConfig, diff_map: np.ndarray, target_mask: np.ndarray) -> dict[str, float]:
    """ 
    Runs the alteration detection process on the input difference map and evaluates 
    the results against the target mask, returning the best metrics achieved by any of the selected clusters.
    """
    result = AContrarioDetector(detector_config).run(diff_map)
    if not result.selected_clusters:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    all_metrics = []
    for cluster in result.selected_clusters:
        prediction = result.cluster_mask(cluster)
        tp = float(np.logical_and(prediction, target_mask).sum())
        fp = float(np.logical_and(prediction, np.logical_not(target_mask)).sum())
        fn = float(np.logical_and(np.logical_not(prediction), target_mask).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        all_metrics.append({"precision": precision, "recall": recall, "f1": f1})
    return max(all_metrics, key=lambda item: item["f1"])


def run_synthetic_study(config: dict, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Runs a synthetic study to evaluate the performance of the alteration detection method under 
    controlled conditions by generating synthetic difference maps.
    The study consists of two experiments:
        1. Varying the background omega parameter while keeping the foreground parameters fixed.
        2. Varying the foreground shift parameter while keeping the background parameters fixed.
    The results of both experiments are saved as CSV files and summary plots are generated to visualize the performance trends.
    """
    detector = DetectorConfig(**config["detector"])
    sim = config["simulation"]
    _ensure_dir(output_root)
    mask = _synthetic_mask(sim["image_height"], sim["image_width"])
    rng = np.random.default_rng(0)

    omega_values = sim["experiment_one_background_omega_values"]
    shift_values = sim["experiment_two_foreground_shifts"]
    repetitions = sim["repetitions"]
    total_runs = (len(omega_values) + len(shift_values)) * repetitions

    pbar = tqdm(total=total_runs, desc="Synthetic study", unit="run")

    rows_one = []
    for omega in omega_values:
        for repetition in range(repetitions):
            pbar.set_postfix({"exp": "1/2", "omega": omega, "rep": repetition})
            background = _nakagami_like(rng, sim["background_shape_mu"], omega, mask.shape)
            foreground = _nakagami_like(rng, sim["foreground_shape_mu"], sim["foreground_omega"], mask.shape) + sim["foreground_base_shift"]
            diff_map = background.copy()
            diff_map[mask] = foreground[mask]
            metrics = _run_detector_on_diff(detector, diff_map, mask)
            rows_one.append({"omega": omega, "repetition": repetition, **metrics})
            pbar.update(1)

    rows_two = []
    for shift in shift_values:
        for repetition in range(repetitions):
            pbar.set_postfix({"exp": "2/2", "shift": shift, "rep": repetition})
            background = _nakagami_like(rng, sim["background_shape_mu"], 6.0, mask.shape)
            foreground = _nakagami_like(rng, sim["foreground_shape_mu"], sim["foreground_omega"], mask.shape) + shift
            diff_map = background.copy()
            diff_map[mask] = foreground[mask]
            metrics = _run_detector_on_diff(detector, diff_map, mask)
            rows_two.append({"shift": shift, "repetition": repetition, **metrics})
            pbar.update(1)

    pbar.close()

    table_one = pd.DataFrame(rows_one)
    table_two = pd.DataFrame(rows_two)
    table_one.to_csv(output_root / "experiment_one.csv", index=False)
    table_two.to_csv(output_root / "experiment_two.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(table_one.groupby("omega")["f1"].mean().index, table_one.groupby("omega")["f1"].mean().values)
    axes[0].set_title("Experiment 1")
    axes[0].set_xlabel("Background omega")
    axes[0].set_ylabel("Mean F1")
    axes[1].plot(table_two.groupby("shift")["f1"].mean().index, table_two.groupby("shift")["f1"].mean().values)
    axes[1].set_title("Experiment 2")
    axes[1].set_xlabel("Foreground shift")
    axes[1].set_ylabel("Mean F1")
    fig.tight_layout()
    fig.savefig(output_root / "synthetic_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return table_one, table_two
