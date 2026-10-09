"""AgileX HDF5 dataset for VLM value-function training.

Reads converted episode folders the same way manifold_demo does:
  <dataset_dir>/<date>/episode_xxx/episode.hdf5
  observation/camera/{head,left,right}  — JPEG base64 frames

Value labels:
  success            : value_label = -(T - t) / T   (start -1, end ~0)
  fail / take_over   : value_label = -1.0 for every frame
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger("openpi")

DEFAULT_PROMPT = "Sort the item into the correct bin box."
FAIL_STATUSES = frozenset({"fail", "take_over"})


def looks_like_agilex_hdf5_dir(data_dir: str | Path) -> bool:
    """True if data_dir looks like converted AgileX episodes, not LeRobot."""
    root = Path(data_dir)
    if not root.is_dir():
        return False
    if (root / "data").is_dir() and any((root / "data").rglob("*.parquet")):
        return False
    if (root / "episode.hdf5").exists():
        return True
    try:
        children = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return False
    for child in children[:32]:
        if (child / "episode.hdf5").exists():
            return True
        try:
            grand = [p for p in child.iterdir() if p.is_dir()]
        except OSError:
            continue
        if any((g / "episode.hdf5").exists() for g in grand[:16]):
            return True
    return False


def scan_episode_folders(dataset_dirs: list[Path]) -> list[Path]:
    """Collect episode folders that contain episode.hdf5 (1–3 levels)."""
    episode_folders: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            episode_folders.append(path)

    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            logger.warning("Dataset directory does not exist: %s", dataset_dir)
            continue
        if (dataset_dir / "episode.hdf5").exists():
            _add(dataset_dir)
        for subdir1 in dataset_dir.iterdir():
            if not subdir1.is_dir():
                continue
            if (subdir1 / "episode.hdf5").exists():
                _add(subdir1)
                continue
            for subdir2 in subdir1.iterdir():
                if subdir2.is_dir() and (subdir2 / "episode.hdf5").exists():
                    _add(subdir2)

    episode_folders.sort()
    return episode_folders


def read_if_success(episode_dir: Path) -> str | None:
    meta_path = episode_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f).get("if_success")
    except (OSError, json.JSONDecodeError):
        return None


def read_meta_prompt(episode_dir: Path) -> str | None:
    meta_path = episode_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            instruction = json.load(f).get("instruction")
        if isinstance(instruction, str):
            text = instruction.strip()
            if text and text.lower() not in {"default", "none", "n/a"}:
                return text
    except (OSError, json.JSONDecodeError):
        return None
    return None


def episode_length(hdf5_path: Path) -> int:
    with h5py.File(hdf5_path, "r") as f:
        for key in (
            "observation/camera/head",
            "action",
            "observations/qpos",
        ):
            if key in f:
                return int(f[key].shape[0])
    raise ValueError(f"Cannot determine episode length from {hdf5_path}")


def compute_success_value_labels(episode_length_n: int) -> np.ndarray:
    t = np.arange(episode_length_n)
    return (-(episode_length_n - t) / episode_length_n).astype(np.float32)


def compute_fail_value_labels(episode_length_n: int) -> np.ndarray:
    return np.full((episode_length_n,), -1.0, dtype=np.float32)


def decode_hdf5_camera_frame(hdf5_file: h5py.File, camera_name: str, frame_idx: int) -> np.ndarray:
    """Decode one JPEG-base64 camera frame. Same path as manifold_demo AgileXDataset."""
    camera_path = f"observation/camera/{camera_name}"
    if camera_path not in hdf5_file:
        raise ValueError(f"{camera_path} not found in HDF5")
    camera_ds = hdf5_file[camera_path]
    if frame_idx >= len(camera_ds):
        frame_idx = len(camera_ds) - 1
    base64_str = camera_ds[frame_idx]
    if isinstance(base64_str, bytes):
        base64_str = base64_str.decode("utf-8")
    jpeg_bytes = base64.b64decode(base64_str)
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode frame {frame_idx} from HDF5 camera {camera_name}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class AgilexHdf5ValueDataset(TorchDataset):
    """Frame-level value dataset over converted AgileX HDF5 episodes."""

    def __init__(
        self,
        dataset_dir: str | list[str],
        *,
        default_prompt: str | None = None,
        filter_if_success: list[str] | tuple[str, ...] | None = None,
    ):
        if isinstance(dataset_dir, str):
            self.dataset_dirs = [Path(dataset_dir)]
        else:
            self.dataset_dirs = [Path(d) for d in dataset_dir]
        self.default_prompt = default_prompt or DEFAULT_PROMPT
        self.filter_if_success = set(filter_if_success) if filter_if_success else None

        folders = scan_episode_folders(self.dataset_dirs)
        if not folders:
            raise ValueError(f"No episode.hdf5 folders found in {dataset_dir}")

        self.episodes: list[dict] = []
        skipped = 0
        for folder in folders:
            status = read_if_success(folder)
            if self.filter_if_success is not None:
                if status is None or status not in self.filter_if_success:
                    skipped += 1
                    continue
            hdf5_path = folder / "episode.hdf5"
            length = episode_length(hdf5_path)
            if length <= 0:
                skipped += 1
                continue
            prompt = read_meta_prompt(folder) or self.default_prompt
            labels = (
                compute_fail_value_labels(length)
                if status in FAIL_STATUSES
                else compute_success_value_labels(length)
            )
            self.episodes.append(
                {
                    "folder": folder,
                    "hdf5_path": hdf5_path,
                    "length": length,
                    "if_success": status,
                    "prompt": prompt,
                    "value_labels": labels,
                }
            )

        if not self.episodes:
            raise ValueError(
                f"No usable AgileX HDF5 episodes in {dataset_dir} "
                f"(filter_if_success={self.filter_if_success})"
            )

        self.episode_offsets = [0]
        for ep in self.episodes:
            self.episode_offsets.append(self.episode_offsets[-1] + ep["length"])
        self.total_frames = self.episode_offsets[-1]

        status_counts: dict[str, int] = {}
        for ep in self.episodes:
            key = ep["if_success"] or "unknown"
            status_counts[key] = status_counts.get(key, 0) + 1
        logger.info(
            "AgileX HDF5 value dataset: episodes=%s frames=%s skipped=%s status=%s",
            len(self.episodes),
            self.total_frames,
            skipped,
            status_counts,
        )

    def __len__(self) -> int:
        return self.total_frames

    def _locate(self, idx: int) -> tuple[int, int]:
        for i, offset in enumerate(self.episode_offsets[1:], 1):
            if idx < offset:
                episode_idx = i - 1
                return episode_idx, idx - self.episode_offsets[episode_idx]
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int) -> dict:
        episode_idx, frame_idx = self._locate(int(idx))
        ep = self.episodes[episode_idx]
        with h5py.File(ep["hdf5_path"], "r") as f:
            cam_high = decode_hdf5_camera_frame(f, "head", frame_idx)
            cam_left = decode_hdf5_camera_frame(f, "left", frame_idx)
            cam_right = decode_hdf5_camera_frame(f, "right", frame_idx)

        return {
            "observation.images.cam_high": cam_high,
            "observation.images.cam_left_wrist": cam_left,
            "observation.images.cam_right_wrist": cam_right,
            "prompt": ep["prompt"],
            "task": ep["prompt"],
            "value_label": np.float32(ep["value_labels"][frame_idx]),
        }
