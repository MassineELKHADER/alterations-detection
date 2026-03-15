import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_uvifl_config
from src.detector import AContrarioDetector
from src.experiments import prepare_uvifl_pair
from src.visualization import ensure_dir, save_cluster_gallery, save_cluster_mask, save_cluster_overlay_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export predicted UVIFL cluster masks without evaluation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--time", type=int)
    parser.add_argument("--all-times", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/uvifl_masks"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_uvifl_config(args.config)
    sequence_config = config.sequences[args.sequence]

    if args.all_times:
        times = sequence_config.candidate_times
    elif args.time is not None:
        times = [args.time]
    else:
        raise SystemExit("Either --time or --all-times must be set.")

    detector = AContrarioDetector(config.detector)

    for time in times:
        pair = prepare_uvifl_pair(sequence_config, config.preprocessing, time)
        result = detector.run(pair.diff_map)

        time_dir = args.output_root / sequence_config.name / f"time_{time:02d}"
        ensure_dir(time_dir)

        save_cluster_gallery(result, pair, time_dir / "gallery.png")
        save_cluster_overlay_panel(result, pair, time_dir / "overlay.png")

        union_mask = None
        for index, cluster in enumerate(result.selected_clusters, start=1):
            mask = result.cluster_mask(cluster)
            save_cluster_mask(mask, time_dir / f"cluster_{index:02d}.png")
            union_mask = mask if union_mask is None else (union_mask | mask)

        if union_mask is not None:
            save_cluster_mask(union_mask, time_dir / "union_mask.png")

        print(
            f"time={time:02d} shot={pair.chosen_shot} "
            f"clusters={len(result.selected_clusters)} output={time_dir}"
        )


if __name__ == "__main__":
    main()
