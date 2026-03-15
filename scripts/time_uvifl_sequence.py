import argparse
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_uvifl_config
from src.detector import AContrarioDetector
from src.experiments import prepare_uvifl_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure runtime for one full UVIFL sequence.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--time", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/timings"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_uvifl_config(args.config)
    sequence_config = config.sequences[args.sequence]
    detector = AContrarioDetector(config.detector)
    times = [args.time] if args.time is not None else sequence_config.candidate_times

    args.output_root.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"Sequence: {sequence_config.name}")
    lines.append(f"Config: {args.config}")
    lines.append(f"Max points: {config.detector.max_points}")
    lines.append(f"Quantization levels: {config.detector.quantization_levels}")
    lines.append("")

    total_start = perf_counter()
    for time in times:
        start = perf_counter()
        pair = prepare_uvifl_pair(sequence_config, config.preprocessing, time)
        result = detector.run(pair.diff_map)
        elapsed = perf_counter() - start
        lines.append(
            f"time={time:02d} shot={pair.chosen_shot} "
            f"elapsed_seconds={elapsed:.4f} "
            f"selected_clusters={len(result.selected_clusters)}"
        )
    total_elapsed = perf_counter() - total_start

    lines.append("")
    lines.append(f"Total elapsed seconds: {total_elapsed:.4f}")
    lines.append(f"Average per timepoint seconds: {total_elapsed / len(times):.4f}")

    max_points = "none" if config.detector.max_points is None else str(config.detector.max_points)
    quantization = config.detector.quantization_levels
    time_suffix = f"_time_{args.time:02d}" if args.time is not None else "_all_times"
    output_path = (
        args.output_root
        / f"{sequence_config.name}{time_suffix}_maxpoints_{max_points}_qlevels_{quantization}_timing.txt"
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote timing report to {output_path}")
    print(f"Total elapsed seconds: {total_elapsed:.4f}")


if __name__ == "__main__":
    main()
