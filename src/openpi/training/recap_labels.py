"""Episode rewards and clipped return targets for RECAP."""

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np

DEFAULT_FAILURE_COST_RATIO = 0.3
VALUE_LABEL = "value_label"
REWARD_LABEL = "reward_label"
METHOD = "recap_tmax_clipped_v1"


def parse_success(value) -> bool:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        state = value.strip().lower()
        if state in {"true", "1", "success", "succeeded", "take_over"}:
            return True
        if state in {"false", "0", "fail", "failed", "failure"}:
            return False
    raise ValueError(f"Unknown episode outcome: {value!r}")


@dataclass(frozen=True)
class RecapLabelScale:
    tmax: int
    failure_cost_ratio: float = DEFAULT_FAILURE_COST_RATIO

    def __post_init__(self):
        if self.tmax <= 0:
            raise ValueError("Tmax must be a positive episode frame count")
        if not math.isfinite(self.failure_cost_ratio) or self.failure_cost_ratio < 0:
            raise ValueError("failure_cost_ratio must be finite and nonnegative")

    @classmethod
    def from_lengths(cls, lengths: Iterable[int], failure_cost_ratio: float = DEFAULT_FAILURE_COST_RATIO):
        return cls(max(lengths, default=0), failure_cost_ratio)

    def labels_at(self, length: int, success: bool, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not 0 < length <= self.tmax:
            raise ValueError(f"Expected 0 < episode length <= Tmax; got {length} and {self.tmax}")
        frames = np.asarray(frames, dtype=np.int64)
        if np.any((frames < 0) | (frames >= length)):
            raise ValueError("Frame index outside its episode")
        terminal = 0.0 if success else -self.failure_cost_ratio
        remaining = length - 1 - frames  # T denotes the last frame index.
        values = np.clip(terminal - remaining / self.tmax, -1.0, 0.0)
        rewards = np.where(remaining == 0, terminal, -1.0 / self.tmax)
        return values.astype(np.float32), rewards.astype(np.float32)

    def labels(self, length: int, success: bool) -> tuple[np.ndarray, np.ndarray]:
        return self.labels_at(length, success, np.arange(length))

    def metadata(self) -> dict:
        return dict(method=METHOD, tmax=self.tmax, failure_cost_ratio=self.failure_cost_ratio,
                    failure_cost=self.tmax * self.failure_cost_ratio, gamma=1.0)
