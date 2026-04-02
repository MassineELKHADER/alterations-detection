import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_uvifl_config
from src.experiments import run_uvifl_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UVIFL reproduction pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--time", type=int)
    parser.add_argument("--all-times", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/uvifl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_uvifl_config(args.config)
    sequence = config.sequences[args.sequence]
    if args.all_times:
        times = sequence.candidate_times
    elif args.time is not None:
        times = [args.time]
    else:
        raise SystemExit("Either --time or --all-times must be set.")

    table = run_uvifl_sequence(
        detector_config=config.detector,
        preprocessing=config.preprocessing,
        sequence_config=sequence,
        times=times,
        output_root=args.output_root,
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
