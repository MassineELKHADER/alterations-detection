from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import SequenceConfig


@dataclass(frozen=True)
class SequenceFrame:
    time: int
    shot: int
    path: Path


def read_image(path: Path) -> np.ndarray:
    """ Reads an image from the given path and converts it to RGB format."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class UviflSequence:
    """ Represents a sequence of frames for a given configuration. """
    def __init__(self, config: SequenceConfig) -> None:
        self.config = config

    def frame_path(self, time: int, shot: int) -> Path:
        """ Constructs the path for a frame given its time and shot."""
        return self.config.root / self.config.pattern.format(time=time, shot=shot)

    def reference_frame(self) -> SequenceFrame:
        """ Returns the reference frame for this sequence. """
        shot = self.config.shots[0]
        return SequenceFrame(
            time=self.config.reference_time,
            shot=shot,
            path=self.frame_path(self.config.reference_time, shot),
        )

    def candidate_frames(self, time: int) -> list[SequenceFrame]:
        """ Returns the candidate frames for the given time. """
        return [SequenceFrame(time=time, shot=shot, path=self.frame_path(time, shot)) for shot in self.config.shots]

    def all_target_times(self) -> Iterable[int]:
        """ Returns all target times for this sequence. """
        return self.config.candidate_times

