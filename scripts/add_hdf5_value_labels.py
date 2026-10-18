"""Write value_label into converted AgileX episode.hdf5 files.

success / take_over: value_label = -(T - t - 1) / (T - 1)
  frame 0 is -1, the last frame is 0. A one-frame episode is 0.
fail: value_label = -1 for every frame

Datasets in SUCCESS_DATASETS are treated as success even when meta.json has no if_success.
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path
import sys
import traceback

import h5py
import numpy as np

ROOT = Path("/ML-vePFS/shared/world_action_model_data/self_produced/agilex_project/sort_product")
SUCCESS_DATASETS = (
    "260728_v1",
    "260729_v1",
    "260730_v1",
    "260804_v1",
    "260805_v1",
    "260807_v1",
    "260810_v1",
)
STATUS_DATASETS = (
    "sort_product_339_904",
    "sort_product_339_905",
    "sort_product_217",
    "sort_hybrid_tag",
    "sort_hybrid_tag_v1",
    "sort_hybrid_tag_v2",
    "sort_hybrid_tag_v3",
)
FAIL_STATUS = "fail"


def iter_episode_hdf5(dataset_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not dataset_dir.is_dir():
        return files
    for child in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        direct = child / "episode.hdf5"
        if direct.is_file():
            files.append(direct)
            continue
        for episode in sorted(path for path in child.iterdir() if path.is_dir()):
            hdf5_path = episode / "episode.hdf5"
            if hdf5_path.is_file():
                files.append(hdf5_path)
    return files


def read_raw_status(hdf5_path: Path) -> str:
    with h5py.File(hdf5_path, "r") as handle:
        if "meta" not in handle:
            return "success"
        source = handle["meta"].attrs.get("camera_source_episode_dir")
    if isinstance(source, bytes):
        source = source.decode()
    if not source:
        return "success"
    meta_path = Path(str(source)) / "meta.json"
    if not meta_path.is_file():
        return "success"
    try:
        status = json.loads(meta_path.read_text(encoding="utf-8")).get("if_success")
    except (OSError, json.JSONDecodeError):
        return "success"
    if isinstance(status, str) and status.strip():
        return status.strip()
    return "success"


def value_labels(length: int, status: str) -> np.ndarray:
    if status == FAIL_STATUS:
        return np.full((length,), -1.0, dtype=np.float32)
    if length <= 1:
        return np.zeros((length,), dtype=np.float32)
    timesteps = np.arange(length)
    return (-(length - timesteps - 1) / (length - 1)).astype(np.float32)


def label_one(task: tuple[str, str]) -> dict:
    hdf5_path = Path(task[0])
    force_success = task[1] == "success"
    try:
        status = "success" if force_success else read_raw_status(hdf5_path)
        with h5py.File(hdf5_path, "a") as handle:
            if "action" in handle:
                length = int(handle["action"].shape[0])
            elif "observations/qpos" in handle:
                length = int(handle["observations/qpos"].shape[0])
            else:
                length = int(handle["observation/camera/head"].shape[0])
            labels = value_labels(length, status)
            if "value_label" in handle:
                del handle["value_label"]
            handle.create_dataset("value_label", data=labels)
            handle.attrs["value_label_if_success"] = status
            handle.attrs["value_label_formula"] = "-(T-t-1)/(T-1)"
        return {"path": str(hdf5_path), "status": status, "frames": length, "ok": True}
    except Exception as err:
        return {"path": str(hdf5_path), "ok": False, "error": f"{err}\n{traceback.format_exc()}"}


def label_dataset(name: str, force_success: bool, num_workers: int) -> None:
    files = iter_episode_hdf5(ROOT / name)
    tasks = [(str(path), "success" if force_success else "raw") for path in files]
    print(f"[START] {name} episodes={len(tasks)} force_success={force_success}", flush=True)
    counts: dict[str, int] = {}
    failed = 0
    with Pool(processes=num_workers) as pool:
        for done, result in enumerate(pool.imap_unordered(label_one, tasks, chunksize=4), start=1):
            if not result["ok"]:
                failed += 1
                print(f"[FAIL] {result['path']}: {result['error']}", flush=True)
            else:
                counts[result["status"]] = counts.get(result["status"], 0) + 1
            if done % 200 == 0 or done == len(tasks):
                print(f"[PROGRESS] {name} {done}/{len(tasks)} failed={failed} status={counts}", flush=True)
    print(f"[DONE] {name} failed={failed} status={counts}", flush=True)
    if failed:
        raise SystemExit(f"{name} had {failed} failed episodes")


def main() -> None:
    workers = 8
    requested = sys.argv[1:]
    if requested:
        known = set(SUCCESS_DATASETS) | set(STATUS_DATASETS)
        unknown = [name for name in requested if name not in known]
        if unknown:
            raise SystemExit(f"未知数据集: {unknown}")
        for name in requested:
            label_dataset(name, force_success=name in SUCCESS_DATASETS, num_workers=workers)
        return
    for name in SUCCESS_DATASETS:
        label_dataset(name, force_success=True, num_workers=workers)
    for name in STATUS_DATASETS:
        label_dataset(name, force_success=False, num_workers=workers)


if __name__ == "__main__":
    main()
