"""Camera layout shared by value training, checkpoint metadata, and inference."""
import json
import os
from pathlib import Path

import numpy as np

CAMERA_NAMES = ("head", "left", "right")
CAMERA_DATA_KEYS = {"head": "cam_high", "left": "cam_left_wrist", "right": "cam_right_wrist"}
# Keep the historical left-wrist key so old two-view checkpoints retain their layout.
CAMERA_MODEL_KEYS = {"head": "base_0_rgb", "left": "wrist_0_rgb", "right": "right_wrist_0_rgb"}
CAMERA_CONFIG_FILE = "value_camera_config.json"


def normalize_camera_names(names) -> tuple[str, ...]:
    if isinstance(names, str):
        names = names.split()
    names = tuple(names)
    if not 1 <= len(names) <= 3 or len(set(names)) != len(names) or any(n not in CAMERA_NAMES for n in names):
        raise ValueError("camera_names must select 1-3 distinct cameras from head, left, right")
    return tuple(n for n in CAMERA_NAMES if n in names)


def image_is_present(value) -> bool:
    return value is not None and not (isinstance(value, (float, np.floating)) and np.isnan(value))


def resolve_camera_names(explicit=None, checkpoint=None, *, default=("head", "left")) -> tuple[str, ...]:
    if explicit is not None:
        return normalize_camera_names(explicit)
    if checkpoint is not None:
        path = Path(checkpoint)
        for folder in (path, path.parent):
            config = folder / CAMERA_CONFIG_FILE
            if config.is_file():
                return normalize_camera_names(json.loads(config.read_text())["camera_names"])
    return normalize_camera_names(default)


def write_camera_config(folder, names) -> None:
    names = normalize_camera_names(names)
    folder = Path(folder)
    path = folder / CAMERA_CONFIG_FILE
    if path.exists():
        if resolve_camera_names(checkpoint=folder) != names:
            raise ValueError(f"Camera layout differs in {folder}; use a new checkpoint_dir/work_dir")
        return
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"camera_names": list(names), "model_keys": [CAMERA_MODEL_KEYS[n] for n in names], "version": 1}
    temporary = folder / f".{CAMERA_CONFIG_FILE}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2)+'\n')
    temporary.replace(path)
