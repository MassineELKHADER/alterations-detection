from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DetectorConfig:
    tau: float
    c_weight: float
    quantization_levels: int
    transform: str
    significance_top_k: int
    max_points: int | None = None


@dataclass(frozen=True)
class PreprocessingConfig:
    illumination_matching: str
    auto_roi_padding: int
    auto_roi_percentile: float
    min_component_area: int


@dataclass(frozen=True)
class SequenceConfig:
    name: str
    root: Path
    pattern: str
    reference_time: int
    candidate_times: list[int]
    shots: list[int]
    resize_max_side: int
    use_auto_roi: bool


@dataclass(frozen=True)
class UviflConfig:
    detector: DetectorConfig
    preprocessing: PreprocessingConfig
    sequences: dict[str, SequenceConfig]


def _load_yaml(path) -> dict[str, Any]:
    """ takes a Path instance and returns a dictionary with the contents of the yaml file """
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_uvifl_config(path) -> UviflConfig:
    """ takes a Path instance and returns a UviflConfig instance with the contents of the yaml file """
    raw = _load_yaml(path)
    detector = DetectorConfig(**raw["detector"])
    preprocessing = PreprocessingConfig(**raw["preprocessing"])
    sequences = {
        name: SequenceConfig(name=name, root=Path(values["root"]), **{k: v for k, v in values.items() if k != "root"})
        for name, values in raw["sequences"].items()
    }
    return UviflConfig(detector=detector, preprocessing=preprocessing, sequences=sequences)


def load_synthetic_config(path) -> dict[str, Any]:
    return _load_yaml(path)
