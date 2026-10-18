"""Write intervention into AgileX episode.hdf5 from takeover_clip.json.

Episodes with has_takeover: frames before start_frame are 0, start_frame and
later frames are 1. Episodes without takeover are all 0.
"""

from __future__ import annotations

import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import traceback

import h5py
import numpy as np

_TRIM_SPEC = importlib.util.spec_from_file_location(
    "convert_agilex_hdf5_instruction",
    "/ML-vePFS/infra_rd/polaris/data_pipeline/wam_pipeline/src/conversion/convert_agilex_hdf5_instruction.py",
)
_TRIM = importlib.util.module_from_spec(_TRIM_SPEC)
sys.modules[_TRIM_SPEC.name] = _TRIM
_TRIM_SPEC.loader.exec_module(_TRIM)

ROOT = Path("/ML-vePFS/shared/world_action_model_data/self_produced/agilex_project/sort_product")
DATASETS = (
    "sort_product_339_904",
    "sort_product_339_905",
    "sort_product_217",
    "sort_hybrid_tag",
)
INDEX_STATE_DATASETS = ("sort_hybrid_tag_v3",)


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


def load_clips(dataset_dir: Path) -> dict[str, dict]:
    sidecar = dataset_dir / "takeover_clip.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    clips: dict[str, dict] = {}
    for episode_name, info in data.items():
        clips[str((dataset_dir / episode_name).resolve())] = info
    return clips


def intervention_labels(length: int, info: dict | None) -> tuple[np.ndarray, int]:
    labels = np.zeros((length,), dtype=np.int64)
    if not info or not info.get("has_takeover"):
        return labels, 0
    start_frame = int(info.get("start_frame", 0))
    if start_frame < 0:
        start_frame = 0
    if start_frame < length:
        labels[start_frame:] = 1
    return labels, start_frame


def label_one(task: tuple[str, str, dict | None]) -> dict:
    hdf5_path = Path(task[0])
    try:
        info = task[2]
        with h5py.File(hdf5_path, "a") as handle:
            if "action" in handle:
                length = int(handle["action"].shape[0])
            elif "observations/qpos" in handle:
                length = int(handle["observations/qpos"].shape[0])
            else:
                length = int(handle["observation/camera/head"].shape[0])
            labels, start_frame = intervention_labels(length, info)
            if "intervention" in handle:
                del handle["intervention"]
            handle.create_dataset("intervention", data=labels)
            handle.attrs["intervention_has_takeover"] = bool(info and info.get("has_takeover"))
            handle.attrs["intervention_start_frame"] = int(start_frame)
        return {
            "path": str(hdf5_path),
            "ok": True,
            "has_takeover": bool(info and info.get("has_takeover")),
            "intervention_frames": int(labels.sum()),
            "frames": length,
            "missing_clip": info is None,
        }
    except Exception as err:
        return {"path": str(hdf5_path), "ok": False, "error": f"{err}\n{traceback.format_exc()}"}


def label_dataset(name: str, num_workers: int) -> None:
    dataset_dir = ROOT / name
    clips = load_clips(dataset_dir)
    files = iter_episode_hdf5(dataset_dir)
    tasks = []
    for path in files:
        info = clips.get(str(path.parent.resolve()))
        tasks.append((str(path), name, info))
    print(f"[START] {name} episodes={len(tasks)} clips={len(clips)}", flush=True)
    failed = 0
    takeover = 0
    missing = 0
    intervention_frames = 0
    with Pool(processes=num_workers) as pool:
        for done, result in enumerate(pool.imap_unordered(label_one, tasks, chunksize=8), start=1):
            if not result["ok"]:
                failed += 1
                print(f"[FAIL] {result['path']}: {result['error']}", flush=True)
            else:
                takeover += int(result["has_takeover"])
                missing += int(result["missing_clip"])
                intervention_frames += result["intervention_frames"]
            if done % 200 == 0 or done == len(tasks):
                print(
                    f"[PROGRESS] {name} {done}/{len(tasks)} failed={failed} "
                    f"takeover={takeover} missing_clip={missing} intervention_frames={intervention_frames}",
                    flush=True,
                )
    print(
        f"[DONE] {name} failed={failed} takeover={takeover} missing_clip={missing} "
        f"intervention_frames={intervention_frames}",
        flush=True,
    )
    if failed:
        raise SystemExit(f"{name} had {failed} failed episodes")


def raw_action(handle: h5py.File) -> np.ndarray:
    return np.concatenate(
        [
            handle["actions/qpos/left_arm"][:],
            handle["actions/gripper/left_gripper"][:],
            handle["actions/qpos/right_arm"][:],
            handle["actions/gripper/right_gripper"][:],
        ],
        axis=1,
    ).astype(np.float32)


def takeover_start_frame(handle: h5py.File) -> int | None:
    if "index_state" not in handle:
        return None
    starts = [int(key) for key in handle["index_state"].keys() if int(handle["index_state"][key][()]) == 2]
    if not starts:
        return None
    return min(starts)


def map_start_through_trim(raw_path: Path, current_length: int) -> tuple[int, bool]:
    """Map the raw takeover frame onto the already trimmed episode."""
    with h5py.File(raw_path, "r") as handle:
        action = raw_action(handle)
        start = takeover_start_frame(handle)
        fps = float(handle.attrs.get("fps", 30) or 30)
    if start is None:
        return 0, False
    segments = _TRIM.find_static_segments(
        action,
        fps=fps,
        ee_threshold=1e-4,
        gripper_threshold=1e-4,
        min_static_frames=15,
        min_static_sec=0.0,
    )
    keep = _TRIM.build_keep_mask(len(action), segments, keep_start_frames=5, keep_end_frames=0)
    if int(keep.sum()) != current_length:
        raise ValueError(f"静止帧掩码长度 {int(keep.sum())} 和当前 episode {current_length} 不一致: {raw_path}")
    kept = np.flatnonzero(keep)
    position = int(np.searchsorted(kept, start, side="left"))
    if position >= len(kept):
        return current_length, True
    return position, True


def label_index_state_one(hdf5_path: str) -> dict:
    path = Path(hdf5_path)
    try:
        with h5py.File(path, "a") as handle:
            length = int(handle["action"].shape[0]) if "action" in handle else int(handle["observations/qpos"].shape[0])
            source = handle["meta"].attrs.get("camera_source_episode_dir")
            if isinstance(source, bytes):
                source = source.decode()
            if not source:
                raise ValueError("缺少 meta.camera_source_episode_dir")
            raw_hdf5 = Path(str(source)) / "episode.hdf5"
            start_frame, has_takeover = map_start_through_trim(raw_hdf5, length)
            info = {"has_takeover": has_takeover, "start_frame": start_frame} if has_takeover else None
            labels, stored_start = intervention_labels(length, info)
            if "intervention" in handle:
                del handle["intervention"]
            handle.create_dataset("intervention", data=labels)
            handle.attrs["intervention_has_takeover"] = has_takeover
            handle.attrs["intervention_start_frame"] = int(stored_start)
            handle.attrs["intervention_raw_source"] = str(source)
        return {
            "path": str(path),
            "ok": True,
            "has_takeover": has_takeover,
            "intervention_frames": int(labels.sum()),
            "frames": length,
            "missing_clip": not has_takeover,
        }
    except Exception as err:
        return {"path": str(path), "ok": False, "error": f"{err}\n{traceback.format_exc()}"}


def label_index_state_dataset(name: str, num_workers: int) -> None:
    files = iter_episode_hdf5(ROOT / name)
    print(f"[START] {name} episodes={len(files)} source=index_state", flush=True)
    failed = 0
    takeover = 0
    missing = 0
    intervention_frames = 0
    with Pool(processes=num_workers) as pool:
        for done, result in enumerate(pool.imap_unordered(label_index_state_one, [str(path) for path in files], chunksize=4), start=1):
            if not result["ok"]:
                failed += 1
                print(f"[FAIL] {result['path']}: {result['error']}", flush=True)
            else:
                takeover += int(result["has_takeover"])
                missing += int(result["missing_clip"])
                intervention_frames += result["intervention_frames"]
            if done % 100 == 0 or done == len(files):
                print(
                    f"[PROGRESS] {name} {done}/{len(files)} failed={failed} "
                    f"takeover={takeover} missing={missing} intervention_frames={intervention_frames}",
                    flush=True,
                )
    print(
        f"[DONE] {name} failed={failed} takeover={takeover} missing={missing} "
        f"intervention_frames={intervention_frames}",
        flush=True,
    )
    if failed:
        raise SystemExit(f"{name} had {failed} failed episodes")


def main() -> None:
    requested = sys.argv[1:] or list(DATASETS)
    for name in requested:
        if name in INDEX_STATE_DATASETS:
            label_index_state_dataset(name, num_workers=8)
        elif name in DATASETS:
            label_dataset(name, num_workers=8)
        else:
            raise SystemExit(f"未知数据集: {name}")


if __name__ == "__main__":
    main()
