"""按旧版规则重写历史数据的 reward_label。

新版标签请用 add_value_labels.py 或 add_hdf5_value_labels.py 同时生成价值和奖励。

规则：
0. 若缺少 reward_label，则创建
1. 先按训练抽帧得到保留帧，T 为保留帧数
2. 除最后一帧外，reward_label = -1 / T
3. 最后一帧 value_label 为 -1 时，reward_label = -1
4. 最后一帧 value_label 为 0，或成功进度标签的 -1/T 时，reward_label = 0

HDF5 里被抽掉的帧写 NaN。LeRobot parquet 不抽帧，T 为 episode 全长。

用法:
    python scripts/rewrite_rewards.py --data_dir /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from openpi.shared import console
from openpi.shared import progress
from openpi.training.recap_labels import FULL_FRAME_REWARD_METHODS


REWARD_COLUMN = "reward_label"
VALUE_LABEL_COLUMN = "value_label"
LEGACY_VALUE_LABEL_COLUMN = "value_lable"


def _episode_row_slices(df: pd.DataFrame) -> list[slice]:
    if "episode_index" not in df.columns:
        return [slice(0, len(df))]

    episode_ids = df["episode_index"].to_numpy()
    if len(episode_ids) == 0:
        return []

    boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(df)]))
    return [slice(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _terminal_reward(last_value_label: float) -> float:
    if np.isclose(last_value_label, -1.0):
        return -1.0
    # 成功和 take_over 的进度标签在最后一帧是 -1/T，按 0 处理。
    if np.isclose(last_value_label, 0.0) or last_value_label > -0.5:
        return 0.0
    raise ValueError(
        f"episode 最后一帧的 {VALUE_LABEL_COLUMN} 必须是 -1 或 0，实际为 {last_value_label}"
    )


def _rewrite_episode_rewards(value_labels: np.ndarray) -> np.ndarray:
    episode_length = len(value_labels)
    if episode_length == 0:
        return value_labels.astype(np.float32)

    updated = np.full((episode_length,), -1.0 / float(episode_length), dtype=np.float32)
    updated[-1] = _terminal_reward(float(value_labels[-1]))
    return updated


def frame_keep_mode(episode_dir: Path) -> str:
    """904/905/217/hybrid 与 v1 的 9月21-24日每 3 帧留 2 帧。"""
    parts = episode_dir.resolve().parts
    if "sort_hybrid_tag_v1" in parts and any(
        part in {"20260921", "20260922", "20260923", "20260924"} for part in parts
    ):
        return "two_of_three"
    if any(
        part in {"sort_product_339_904", "sort_product_339_905", "sort_product_217", "sort_hybrid_tag"}
        for part in parts
    ):
        return "two_of_three"
    return "all"


def kept_frame_indices(length: int, mode: str) -> np.ndarray:
    indices = np.arange(length, dtype=np.int64)
    if mode == "two_of_three":
        return indices[indices % 3 != 2]
    if mode != "all":
        raise ValueError(f"未知抽帧模式: {mode}")
    return indices


def iter_episode_hdf5(dataset_dir: Path) -> list[Path]:
    files: list[Path] = []
    direct = dataset_dir / "episode.hdf5"
    if direct.is_file():
        return [direct]
    if not dataset_dir.is_dir():
        return files
    for child in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        episode_file = child / "episode.hdf5"
        if episode_file.is_file():
            files.append(episode_file)
            continue
        for episode in sorted(path for path in child.iterdir() if path.is_dir()):
            hdf5_path = episode / "episode.hdf5"
            if hdf5_path.is_file():
                files.append(hdf5_path)
    return files


def rewrite_rewards_in_hdf5(hdf5_path: str) -> dict:
    path = Path(hdf5_path)
    try:
        with h5py.File(path, "a") as handle:
            if handle.attrs.get("value_label_method") in FULL_FRAME_REWARD_METHODS:
                raise ValueError("Use add_hdf5_value_labels.py to regenerate paired RECAP rewards")
            if VALUE_LABEL_COLUMN not in handle:
                raise ValueError(f"缺少 {VALUE_LABEL_COLUMN}")
            value_labels = np.asarray(handle[VALUE_LABEL_COLUMN], dtype=np.float32)
            length = int(value_labels.shape[0])
            stride = frame_keep_mode(path.parent)
            kept = kept_frame_indices(length, stride)
            kept_values = value_labels[kept]
            kept_rewards = _rewrite_episode_rewards(kept_values)
            rewards = np.full((length,), np.nan, dtype=np.float32)
            rewards[kept] = kept_rewards
            if REWARD_COLUMN in handle:
                del handle[REWARD_COLUMN]
            handle.create_dataset(REWARD_COLUMN, data=rewards)
            handle.attrs["reward_label_stride"] = stride
            handle.attrs["reward_label_kept_frames"] = int(kept.size)
        return {"path": str(path), "ok": True, "stride": stride, "frames": length, "kept": int(kept.size)}
    except Exception as err:
        return {"path": str(path), "ok": False, "error": str(err)}


def _get_value_labels(df: pd.DataFrame, parquet_path: Path) -> tuple[np.ndarray, str]:
    for column_name in (VALUE_LABEL_COLUMN, LEGACY_VALUE_LABEL_COLUMN):
        if column_name in df.columns:
            return df[column_name].to_numpy(dtype=np.float32, copy=True), column_name
    raise ValueError(
        f"{parquet_path} 缺少价值标签列，尝试过: {(VALUE_LABEL_COLUMN, LEGACY_VALUE_LABEL_COLUMN)}"
    )


def rewrite_rewards_in_parquet(parquet_path: Path) -> tuple[int, int]:
    df = pd.read_parquet(parquet_path)
    value_labels, source_column = _get_value_labels(df, parquet_path)
    rewards = np.empty((len(df),), dtype=np.float32)
    episode_count = 0

    for row_slice in _episode_row_slices(df):
        rewards[row_slice] = _rewrite_episode_rewards(value_labels[row_slice])
        episode_count += 1

    if REWARD_COLUMN not in df.columns:
        print(console.warn(f"{parquet_path} 缺少 {REWARD_COLUMN} 列，已按 {source_column} 新建"))
    df[REWARD_COLUMN] = rewards
    df.to_parquet(parquet_path, index=False)
    return len(df), episode_count


def update_info_json(info_path: Path) -> None:
    with open(info_path) as f:
        info = json.load(f)

    if "features" not in info:
        info["features"] = {}

    reward_feature = info["features"].setdefault(
        REWARD_COLUMN,
        {
            "dtype": "float32",
            "shape": [1],
        },
    )
    reward_feature["description"] = (
        "Reward rewritten per episode: non-terminal frames are -1/T; "
        "last frame is set from terminal value_label (-1->-1, 0->0)."
    )

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)


def _looks_like_hdf5_dir(data_dir: Path) -> bool:
    return bool(iter_episode_hdf5(data_dir))


def rewrite_hdf5_dir(data_dir: Path, num_workers: int) -> None:
    files = iter_episode_hdf5(data_dir)
    print(console.info(f"{data_dir.name}: {len(files)} 个 episode.hdf5"), flush=True)
    failed = 0
    kept_frames = 0
    stride_counts: dict[int, int] = {}
    with Pool(processes=num_workers) as pool:
        for done, result in enumerate(pool.imap_unordered(rewrite_rewards_in_hdf5, [str(path) for path in files], chunksize=8), start=1):
            if not result["ok"]:
                failed += 1
                print(f"[FAIL] {result['path']}: {result['error']}", flush=True)
                continue
            kept_frames += result["kept"]
            stride_counts[result["stride"]] = stride_counts.get(result["stride"], 0) + 1
            if done % 400 == 0 or done == len(files):
                print(f"[PROGRESS] {data_dir.name} {done}/{len(files)} failed={failed} stride={stride_counts}", flush=True)
    print(console.ok(f"{data_dir.name} 完成 failed={failed} kept_frames={kept_frames} stride={stride_counts}"), flush=True)
    if failed:
        raise SystemExit(f"{data_dir} 有 {failed} 条失败")


def main() -> None:
    parser = argparse.ArgumentParser(description="按抽帧后的长度重写 reward_label")
    parser.add_argument("--data_dir", type=str, nargs="+", required=True, help="LeRobot 数据集或 AgileX HDF5 根目录")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    for raw_dir in args.data_dir:
        data_dir = Path(raw_dir)
        if not data_dir.exists():
            raise ValueError(f"数据目录不存在: {data_dir}")
        if (data_dir / "meta" / "recap_value_labels.json").is_file():
            raise ValueError("Use add_value_labels.py to regenerate paired RECAP rewards")
        if _looks_like_hdf5_dir(data_dir):
            rewrite_hdf5_dir(data_dir, args.num_workers)
            continue

        parquet_dir = data_dir / "data"
        if not parquet_dir.exists():
            raise ValueError(f"找不到 data 目录: {parquet_dir}")

        parquet_files = sorted(parquet_dir.rglob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"找不到 parquet 文件: {parquet_dir}")

        print(console.info(f"找到 {len(parquet_files)} 个 parquet 文件"))

        total_frames = 0
        total_episodes = 0
        pbar = tqdm(
            parquet_files,
            desc="重写 reward_label",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        progress.sync_pbar_color(pbar)
        for parquet_path in pbar:
            progress.sync_pbar_color(pbar)
            frames, episodes = rewrite_rewards_in_parquet(parquet_path)
            total_frames += frames
            total_episodes += episodes

        info_path = data_dir / "meta" / "info.json"
        if info_path.exists():
            update_info_json(info_path)

        print(console.ok(f"\n完成! 共处理 {len(parquet_files)} 个 parquet, {total_episodes} 个 episodes, {total_frames} 帧"))


if __name__ == "__main__":
    main()
