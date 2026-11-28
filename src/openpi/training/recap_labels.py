"""Episode rewards and return targets for RECAP."""

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np

DEFAULT_FAILURE_COST_RATIO = 0.3
VALUE_LABEL = "value_label"
REWARD_LABEL = "reward_label"
TMAX_CLIP = "tmax-clip"
PLUS_C = "plus-c"
METHODS = {TMAX_CLIP: "recap_tmax_clipped_v1", PLUS_C: "recap_global_return_min_v1"}
FULL_FRAME_REWARD_METHODS = frozenset(METHODS.values())


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
    failure_cost: float
    divisor: float
    normalization: str

    @classmethod
    def from_episodes(cls, episodes: Iterable[tuple[int, bool]], *, normalization: str = TMAX_CLIP,
                      failure_cost_ratio: float | None = None, failure_cost: float | None = None):
        episodes = list(episodes)
        if not episodes or any(length <= 0 for length, _ in episodes):
            raise ValueError("Tmax must be a positive episode frame count")
        tmax = max(length for length, _ in episodes)
        if normalization not in METHODS:
            raise ValueError(f"Unknown normalization: {normalization}")
        if failure_cost is not None and failure_cost_ratio is not None:
            raise ValueError("Specify only one of failure_cost and failure_cost_ratio")
        if failure_cost is None:
            ratio = DEFAULT_FAILURE_COST_RATIO if failure_cost_ratio is None else failure_cost_ratio
            failure_cost = tmax * ratio
        if not math.isfinite(failure_cost) or failure_cost < 0:
            raise ValueError("Failure cost must be finite and nonnegative")
        divisor = float(tmax)
        if normalization == PLUS_C:
            # The most negative observed return need not be on the longest episode.
            worst = max(length - 1 + (0 if success else failure_cost) for length, success in episodes)
            divisor = float(worst or 1)
        return cls(tmax, failure_cost, divisor, normalization)

    @classmethod
    def from_lengths(cls, lengths: Iterable[int], failure_cost_ratio: float = DEFAULT_FAILURE_COST_RATIO):
        """Fit the clipped mode, whose divisor needs only the episode lengths."""
        return cls.from_episodes(((length, True) for length in lengths), failure_cost_ratio=failure_cost_ratio)

    @property
    def formula(self) -> str:
        formula = "(-(T-t)-failure_cost*failed)/divisor"
        return f"clip({formula}, -1, 0)" if self.normalization == TMAX_CLIP else formula

    def labels_at(self, length: int, success: bool, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not 0 < length <= self.tmax:
            raise ValueError(f"Expected 0 < episode length <= Tmax; got {length} and {self.tmax}")
        frames = np.asarray(frames, dtype=np.int64)
        if np.any((frames < 0) | (frames >= length)):
            raise ValueError("Frame index outside its episode")
        terminal = 0.0 if success else -self.failure_cost
        remaining = length - 1 - frames  # T denotes the last frame index.
        values = (terminal - remaining) / self.divisor
        if self.normalization == TMAX_CLIP:
            values = np.clip(values, -1.0, 0.0)
        elif np.any(values < -1.0):
            raise ValueError("Episode falls outside the fitted return range; fit on the full input corpus")
        rewards = np.where(remaining == 0, terminal / self.divisor, -1.0 / self.divisor)
        return values.astype(np.float32), rewards.astype(np.float32)

    def labels(self, length: int, success: bool) -> tuple[np.ndarray, np.ndarray]:
        return self.labels_at(length, success, np.arange(length))

    def metadata(self) -> dict:
        return dict(method=METHODS[self.normalization], normalization=self.normalization, tmax=self.tmax,
                    divisor=self.divisor, failure_cost=self.failure_cost,
                    failure_cost_ratio=self.failure_cost / self.tmax, gamma=1.0, formula=self.formula)


def add_normalization_args(parser) -> None:
    parser.add_argument("--normalization", choices=tuple(METHODS), default=TMAX_CLIP,
                        help="tmax-clip: divide by Tmax and clip values; plus-c: divide by |minimum corpus return|")
    cost = parser.add_mutually_exclusive_group()
    cost.add_argument("--failure-cost-ratio", type=float,
                      help="Raw failure penalty as a multiple of Tmax (default: 0.3 in both modes)")
    cost.add_argument("--failure-cost", type=float,
                      help="Override the raw failure penalty in steps")
