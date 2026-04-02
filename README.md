# Alterations detection

This project is an implementation and study of the paper *A-contrario framework for detection of alterations in varnished surfaces*. We reproduce the main pipeline, study it theoretically and experimentally, then compare different implementation choices, baselines and experimental setups.

## Contributions

- Implemented the alteration-detection pipeline described in the paper in `src/`, including preprocessing, clustering, evaluation and visualization.
- Reproduced the UVIFL sequence workflow from config files in `configs/uvifl_sequences.yaml`.
- Since the violin data does not come with ground-truth alteration masks, we generated synthetic data to evaluate the method quantitatively under controlled conditions.
- Studied the method both theoretically and experimentally through synthetic benchmarks and runtime analysis.
- Compared different setups, including the main NFA-based pipeline, a naive threshold baseline, and an HDBSCAN-based alternative.

## Project Structure

- `src/`: core pipeline.
- `scripts/`: runnable entry points.
- `configs/`: experiment settings.
- `data/`: UVIFL images and annotations.
- `outputs/`: generated figures, csv summaries, masks, and timing reports.

## Setup

Create a Python 3.10+ environment. If you use Conda:

```bash
conda create -n alterations-detection python=3.10
conda activate alterations-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

1. Run the synthetic benchmark:
   ```bash
   python scripts/run_synthetic.py --config configs/synthetic.yaml --output outputs/synthetic_fullRun
   ```
2. Run the naive threshold baseline on the same synthetic setup:
   ```bash
   python scripts/naive_baseline.py --config configs/synthetic.yaml --output outputs/synthetic_naiveBaseline
   ```
3. Run the HDBSCAN comparison:
   ```bash
   python scripts/hdbscan_comparison.py --config configs/synthetic.yaml --output outputs/synthetic_hdbscan
   ```
4. Run the real-sequence UVIFL pipeline for one sequence and all candidate times:
   ```bash
   python scripts/run_uvifl.py --config configs/uvifl_sequences.yaml --sequence WS01 --all-times --output-root outputs/uvifl
   ```
5. Export predicted masks for one time step or for all times:
   ```bash
   python scripts/export_uvifl_masks.py --config configs/uvifl_sequences.yaml --sequence WS01 --time 1 --output-root outputs/uvifl_masks
   ```
6. Measure runtime on a sequence:
   ```bash
   python scripts/time_uvifl_sequence.py --config configs/uvifl_sequences.yaml --sequence WS01 --time 1 --output-root outputs/timings
   ```

## Outputs

- Synthetic runs save `experiment_one.csv`, `experiment_two.csv`, and summary plots.
- UVIFL runs save per-time visualizations and a `summary.csv` per sequence.

## Reference

- Rezaei et al., *A-contrario framework for detection of alterations in varnished surfaces*, Journal of Visual Communication and Image Representation, 2022.
