"""Label adv_ind on AgileX HDF5 using the VLM value model and pi0.5 frame stride.

Early date folders are marked intervention=1 and adv_ind=positive.
Frames at and after a takeover start are already intervention=1 and are also
written positive. Other kept frames are scored with
A_t = sum(r[t:t+N]) + V_{t+N} - V_t on the subsampled sequence.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import json
import logging
import math
import multiprocessing
import pickle
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from openpi.training.agilex_hdf5_value_dataset import (
    BASKET_COLOR,
    PI05_PROMPT,
    decode_hdf5_camera_frame,
    load_prompt_segments,
    prompt_for_frame,
    read_barcodes,
    scan_episode_folders,
)

LOG = logging.getLogger("openpi")
EARLY_DATASETS = frozenset({
    "260728_v1",
    "260729_v1",
    "260730_v1",
    "260804_v1",
    "260805_v1",
    "260807_v1",
    "260810_v1",
})
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


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


def _is_early(path: Path) -> bool:
    return any(part in EARLY_DATASETS for part in path.parts)


def _episode_length(handle: h5py.File) -> int:
    if "action" in handle:
        return int(handle["action"].shape[0])
    if "observations/qpos" in handle:
        return int(handle["observations/qpos"].shape[0])
    for camera in ("head", "left", "right"):
        key = f"observation/camera/{camera}"
        if key in handle:
            return int(handle[key].shape[0])
    raise ValueError("Cannot determine episode length: no action, state, or camera frames")


def _write_strings(handle: h5py.File, name: str, values: np.ndarray) -> None:
    if name in handle:
        del handle[name]
    handle.create_dataset(name, data=np.asarray(values, dtype=STRING_DTYPE))


def _write_positive_episode(folder: str) -> dict:
    path = Path(folder) / "episode.hdf5"
    try:
        with h5py.File(path, "a") as handle:
            length = _episode_length(handle)
            if "intervention" in handle:
                del handle["intervention"]
            handle.create_dataset("intervention", data=np.ones((length,), dtype=np.int64))
            _write_strings(handle, "adv_ind", np.array(["positive"] * length, dtype=object))
            handle.attrs["intervention_has_takeover"] = False
            handle.attrs["intervention_start_frame"] = 0
            handle.attrs["adv_ind_source"] = "early_episode_positive"
        return {"path": str(path), "ok": True, "frames": length}
    except Exception as err:
        return {"path": str(path), "ok": False, "error": str(err)}


def write_early_positive(folders: list[Path], num_workers: int) -> None:
    if not folders:
        return
    print(f"[EARLY] episodes={len(folders)}", flush=True)
    failed = 0
    frames = 0
    with multiprocessing.Pool(processes=num_workers) as pool:
        for done, result in enumerate(
            pool.imap_unordered(_write_positive_episode, [str(folder) for folder in folders], chunksize=8),
            start=1,
        ):
            if not result["ok"]:
                failed += 1
                print(f"[FAIL] {result['path']}: {result['error']}", flush=True)
            else:
                frames += result["frames"]
            if done % 200 == 0 or done == len(folders):
                print(f"[EARLY] {done}/{len(folders)} failed={failed} frames={frames}", flush=True)
    if failed:
        raise SystemExit(f"early positive labeling failed: {failed}")


def _load_episode(folder: Path, lookahead: int) -> dict | None:
    path = folder / "episode.hdf5"
    with h5py.File(path, "r") as handle:
        length = _episode_length(handle)
        if "reward_label" not in handle:
            raise ValueError(f"缺少 reward_label: {path}")
        reward = np.asarray(handle["reward_label"], dtype=np.float32)
        if "intervention" in handle:
            intervention = np.asarray(handle["intervention"], dtype=np.int64).reshape(-1)
        else:
            intervention = np.zeros((length,), dtype=np.int64)
        barcodes = read_barcodes(handle)
    if reward.shape[0] != length or intervention.shape[0] != length:
        raise ValueError(f"标签长度和 episode 不一致: {path}")
    stride = frame_keep_mode(folder)
    kept = kept_frame_indices(length, stride)
    reward_kept = reward[kept]
    if not np.isfinite(reward_kept).all():
        raise ValueError(f"抽帧后的 reward_label 含 NaN: {path}")
    intervention_kept = (intervention[kept] != 0).astype(np.int64)
    autonomous = np.flatnonzero(intervention_kept == 0)
    needed: set[int] = set(autonomous.tolist())
    for pos in autonomous.tolist():
        future = pos + lookahead
        if future < len(kept):
            needed.add(future)
    return {
        "path": str(path),
        "folder": str(folder),
        "length": length,
        "kept": kept,
        "reward": reward_kept,
        "intervention": intervention_kept,
        "needed": np.asarray(sorted(needed), dtype=np.int64),
        "barcodes": barcodes,
        "segments": load_prompt_segments(folder, length),
        "stride": stride,
    }


def build_jobs(folders: list[Path], lookahead: int) -> tuple[list[dict], np.ndarray]:
    jobs: list[dict] = []
    frames: list[tuple[int, int]] = []
    for folder in tqdm(folders, desc="准备抽帧索引"):
        job = _load_episode(folder, lookahead)
        episode_index = len(jobs)
        for kept_pos in job["needed"].tolist():
            frames.append((episode_index, int(kept_pos)))
        jobs.append(job)
    frame_array = np.asarray(frames, dtype=np.int64).reshape(-1, 2)
    return jobs, frame_array


def _readable_barcodes(barcodes: dict) -> dict[int, str]:
    readable = {}
    for index, barcode in barcodes.items():
        parts = str(barcode).split("-")
        if len(parts) >= 2 and parts[1] in BASKET_COLOR:
            readable[int(index)] = str(barcode)
    return readable


def _inference_prompt(template: str, barcodes: dict, frame_idx: int, segments) -> str:
    """Keep inference moving when a scanned barcode is not 01/02."""
    try:
        return prompt_for_frame(template, barcodes, frame_idx, segments)
    except ValueError:
        readable = _readable_barcodes(barcodes)
        if readable:
            try:
                return prompt_for_frame(template, readable, frame_idx, None)
            except ValueError:
                pass
        return template.split("\n", 1)[0].strip()


class InferFrameDataset:
    def __init__(self, jobs: list[dict], frames: np.ndarray, prompt: str, camera_names=("head", "left")):
        self.jobs = jobs
        self.frames = frames
        self.prompt = prompt
        from openpi.shared.value_cameras import normalize_camera_names
        self.camera_names = normalize_camera_names(camera_names)

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, index: int) -> dict:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices) -> list[dict]:
        grouped: dict[int, list[tuple[int, int]]] = {}
        for position, index in enumerate(indices):
            episode_index, kept_pos = self.frames[int(index)]
            grouped.setdefault(int(episode_index), []).append((position, int(kept_pos)))
        samples: list[dict | None] = [None] * len(indices)
        for episode_index, requests in grouped.items():
            job = self.jobs[episode_index]
            with h5py.File(job["path"], "r") as handle:
                for position, kept_pos in requests:
                    frame_idx = int(job["kept"][kept_pos])
                    prompt = _inference_prompt(self.prompt, job["barcodes"], frame_idx, job["segments"])
                    samples[position] = {
                        "prompt": prompt,
                        "task": prompt,
                        "value_label": np.float32(0.0),
                    }
                    from openpi.shared.value_cameras import CAMERA_DATA_KEYS
                    for camera in self.camera_names:
                        if f"observation/camera/{camera}" in handle:
                            samples[position][f"observation.images.{CAMERA_DATA_KEYS[camera]}"] = decode_hdf5_camera_frame(handle, camera, frame_idx)
        return samples


def _gpu_count() -> int:
    try:
        lines = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return 1
    return max(1, len([line for line in lines if line.strip()]))


def infer_shard(args, jobs: list[dict], frames: np.ndarray, shard: int, num_shards: int) -> None:
    import jax
    import jax.numpy as jnp
    import torch
    from openpi.models import model as _model
    from openpi.models.value_model_config import ValueModelConfig
    import openpi.training.data_loader as _data_loader
    from train_value import build_value_data_config

    import label_advantage_from_vlm as advantage

    if args.frame_start is not None or args.frame_end is not None:
        if args.frame_start is None or args.frame_end is None:
            raise ValueError("--frame_start 和 --frame_end 需要同时设置")
        start = args.frame_start
        end = args.frame_end
    else:
        start = shard * math.ceil(len(frames) / num_shards)
        end = min(len(frames), (shard + 1) * math.ceil(len(frames) / num_shards))
    shard_frames = frames[start:end]
    print(f"[SHARD {shard}] frames={len(shard_frames)} device={jax.devices()}", flush=True)
    if len(shard_frames) == 0:
        return

    checkpoint_path = advantage._resolve_checkpoint_path(Path(args.checkpoint_dir), args.checkpoint_name)
    config = ValueModelConfig(camera_names=args.camera_names)
    params = advantage._load_checkpoint_params(checkpoint_path, use_ema=args.use_ema)
    model = config.load(params, remove_extra_params=True)
    supports = jnp.linspace(-1.0, 0.0, 201, dtype=jnp.float32)
    tokenizer_path = advantage._resolve_local_gemma_tokenizer_path(args.tokenizer_path)
    advantage._validate_gemma_tokenizer(tokenizer_path)
    data_config = build_value_data_config(
        args.data_dir,
        config,
        tokenizer_path=tokenizer_path,
        default_prompt=PI05_PROMPT,
    )
    dataset = _data_loader.transform_dataset(
        InferFrameDataset(jobs, shard_frames, PI05_PROMPT, camera_names=config.camera_names),
        data_config,
        skip_norm_stats=True,
    )
    num_workers = 4 if args.num_workers is None else max(0, args.num_workers)
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    generator = torch.Generator()
    generator.manual_seed(args.seed + shard)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "multiprocessing_context": mp_context,
        "persistent_workers": num_workers > 0,
        "collate_fn": _data_loader._collate_fn,
        "worker_init_fn": _data_loader._worker_init_fn,
        "drop_last": False,
        "generator": generator,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = torch.utils.data.DataLoader(**loader_kwargs)
    values = np.memmap(Path(args.work_dir) / "values.dat", dtype=np.float32, mode="r+", shape=(len(frames),))
    rng = jax.random.key(shard)

    @jax.jit
    def infer_fn(observation: _model.Observation) -> jax.Array:
        logits = model(observation, train=False)
        probs = jax.nn.softmax(logits, axis=-1)
        return jnp.sum(probs * supports, axis=-1)

    cursor = start
    for step, batch in enumerate(loader):
        batch.pop("value", None)
        observation = _model.Observation.from_dict(batch)
        image_keys = list(observation.images.keys())
        observation = _model.preprocess_observation(rng, observation, train=False, image_keys=image_keys)
        predicted = np.asarray(jax.device_get(infer_fn(observation)), dtype=np.float32)
        values[cursor : cursor + len(predicted)] = predicted
        cursor += len(predicted)
        if step % 20 == 0:
            print(f"[SHARD {shard}] {cursor - start}/{len(shard_frames)}", flush=True)
            values.flush()
    values.flush()
    if cursor != end:
        raise RuntimeError(f"shard {shard} wrote {cursor - start} values, expected {end - start}")
    print(f"[SHARD {shard}] done", flush=True)


def _launch_shards(args, num_shards: int) -> None:
    script = Path(__file__).resolve().parent / "label_advantage_from_vlm.py"
    processes = []
    for shard in range(num_shards):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard)
        command = [
            sys.executable,
            str(script),
            "--checkpoint_dir",
            args.checkpoint_dir,
            "--work_dir",
            args.work_dir,
            "--batch_size",
            str(args.batch_size),
            "--lookahead",
            str(args.lookahead),
            "--top_percent",
            str(args.top_percent),
            "--seed",
            str(args.seed),
            "--num_shards",
            str(num_shards),
            "--infer_shard",
            str(shard),
            "--num_workers",
            str(4 if args.num_workers is None else args.num_workers),
            "--data_dir",
            *args.data_dir,
        ]
        command.extend(["--camera_names", *args.camera_names])
        if args.checkpoint_name:
            command.extend(["--checkpoint_name", args.checkpoint_name])
        if args.tokenizer_path:
            command.extend(["--tokenizer_path", args.tokenizer_path])
        if not args.use_ema:
            command.append("--no-use_ema")
        processes.append(subprocess.Popen(command, env=env))
    failed = [process.wait() for process in processes]
    if any(code != 0 for code in failed):
        raise SystemExit(f"推理进程失败: {failed}")


def _nan_spans(values: np.ndarray) -> list[tuple[int, int]]:
    nan_idx = np.flatnonzero(~np.isfinite(values))
    if len(nan_idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(nan_idx) != 1) + 1
    groups = np.split(nan_idx, cuts)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def _split_spans(spans: list[tuple[int, int]], num_shards: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for start, end in spans:
        chunk = math.ceil((end - start) / num_shards)
        for shard in range(num_shards):
            piece_start = start + shard * chunk
            piece_end = min(end, piece_start + chunk)
            if piece_start < piece_end:
                ranges.append((piece_start, piece_end))
    return ranges


def _launch_ranges(args, ranges: list[tuple[int, int]]) -> None:
    script = Path(__file__).resolve().parent / "label_advantage_from_vlm.py"
    processes = []
    for shard, (start, end) in enumerate(ranges):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard)
        command = [
            sys.executable,
            str(script),
            "--checkpoint_dir",
            args.checkpoint_dir,
            "--work_dir",
            args.work_dir,
            "--batch_size",
            str(args.batch_size),
            "--lookahead",
            str(args.lookahead),
            "--top_percent",
            str(args.top_percent),
            "--seed",
            str(args.seed),
            "--num_shards",
            str(len(ranges)),
            "--infer_shard",
            str(shard),
            "--frame_start",
            str(start),
            "--frame_end",
            str(end),
            "--num_workers",
            str(4 if args.num_workers is None else args.num_workers),
            "--data_dir",
            *args.data_dir,
        ]
        command.extend(["--camera_names", *args.camera_names])
        if args.checkpoint_name:
            command.extend(["--checkpoint_name", args.checkpoint_name])
        if args.tokenizer_path:
            command.extend(["--tokenizer_path", args.tokenizer_path])
        if not args.use_ema:
            command.append("--no-use_ema")
        print(f"[RESUME] shard={shard} frames={start}:{end} gpu={shard}", flush=True)
        processes.append(subprocess.Popen(command, env=env))
    failed = [process.wait() for process in processes]
    if any(code != 0 for code in failed):
        raise SystemExit(f"推理进程失败: {failed}")


def _write_inferred_episode(task: tuple[dict, np.ndarray, str]) -> dict:
    job, values_for_needed, source = task
    try:
        labels = np.array(["none"] * job["length"], dtype=object)
        kept_labels = np.array(["negative"] * len(job["kept"]), dtype=object)
        kept_labels[job["intervention"] == 1] = "positive"
        autonomous = np.flatnonzero(job["intervention"] == 0)
        value_map = np.full((len(job["kept"]),), np.nan, dtype=np.float32)
        value_map[job["needed"]] = values_for_needed
        advantages = []
        for pos in autonomous.tolist():
            reward_end = min(pos + job["lookahead"], len(job["kept"]))
            reward_sum = float(np.sum(job["reward"][pos:reward_end]))
            future = 0.0 if pos + job["lookahead"] >= len(job["kept"]) else float(value_map[pos + job["lookahead"]])
            advantage = reward_sum + future - float(value_map[pos])
            advantages.append(advantage)
            if advantage >= job["threshold"]:
                kept_labels[pos] = "positive"
        labels[job["kept"]] = kept_labels
        with h5py.File(job["path"], "a") as handle:
            _write_strings(handle, "adv_ind", labels)
            handle.attrs["adv_ind_source"] = source
            handle.attrs["adv_ind_stride"] = job["stride"]
            handle.attrs["adv_ind_lookahead"] = int(job["lookahead"])
            handle.attrs["adv_ind_threshold"] = float(job["threshold"])
        return {
            "ok": True,
            "positive": int(np.sum(kept_labels == "positive")),
            "negative": int(np.sum(kept_labels == "negative")),
            "autonomous": len(advantages),
        }
    except Exception as err:
        return {"ok": False, "path": job["path"], "error": str(err)}


def write_inferred(jobs: list[dict], frames: np.ndarray, values: np.ndarray, threshold: float, lookahead: int, source: str) -> None:
    by_episode: list[list[float]] = [[] for _ in jobs]
    for (episode_index, _), value in zip(frames, values, strict=True):
        by_episode[int(episode_index)].append(float(value))
    tasks = []
    for job, episode_values in zip(jobs, by_episode, strict=True):
        payload = dict(job)
        payload["lookahead"] = lookahead
        payload["threshold"] = threshold
        tasks.append((payload, np.asarray(episode_values, dtype=np.float32), source))
    positive = negative = failed = 0
    autonomous_count = 0
    with multiprocessing.Pool(processes=8) as pool:
        for done, result in enumerate(pool.imap_unordered(_write_inferred_episode, tasks, chunksize=4), start=1):
            if not result["ok"]:
                failed += 1
                print(f"[FAIL] {result['path']}: {result['error']}", flush=True)
                continue
            positive += result["positive"]
            negative += result["negative"]
            autonomous_count += result["autonomous"]
            if done % 400 == 0 or done == len(tasks):
                print(
                    f"[WRITE] {done}/{len(tasks)} failed={failed} positive={positive} negative={negative}",
                    flush=True,
                )
    print(
        f"[DONE] episodes={len(tasks)} failed={failed} kept_positive={positive} "
        f"kept_negative={negative} autonomous={autonomous_count} threshold={threshold}",
        flush=True,
    )
    if failed:
        raise SystemExit(f"adv_ind 写回失败 {failed} 条")


def run(args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    from openpi.shared.value_cameras import resolve_camera_names, write_camera_config
    import label_advantage_from_vlm as advantage
    checkpoint = advantage._resolve_checkpoint_path(Path(args.checkpoint_dir), args.checkpoint_name)
    args.camera_names = resolve_camera_names(getattr(args, "camera_names", None), checkpoint)
    work_dir = Path(args.work_dir)
    if getattr(args, "resume", False) and not (work_dir / "value_camera_config.json").exists() and args.camera_names != ("head", "left"):
        raise ValueError("Legacy inference cache uses head+left; choose a new work_dir for other cameras")
    write_camera_config(work_dir, args.camera_names)
    print(f"[CAMERAS] {args.camera_names}", flush=True)
    jobs_path = work_dir / "jobs.pkl"
    frames_path = work_dir / "frames.npy"
    if args.infer_shard is not None:
        with open(jobs_path, "rb") as handle:
            jobs = pickle.load(handle)
        frames = np.load(frames_path)
        infer_shard(args, jobs, frames, args.infer_shard, args.num_shards or 1)
        return

    values_path = work_dir / "values.dat"
    if getattr(args, "resume", False):
        if not jobs_path.exists() or not frames_path.exists() or not values_path.exists():
            raise RuntimeError(f"找不到可续跑的中间文件: {work_dir}")
        with open(jobs_path, "rb") as handle:
            jobs = pickle.load(handle)
        frames = np.load(frames_path)
        existing = np.memmap(values_path, dtype=np.float32, mode="r", shape=(len(frames),))
        spans = _nan_spans(existing)
        done = int(np.isfinite(existing).sum())
        del existing
        print(f"[RESUME] frames={len(frames)} done={done} missing_spans={spans}", flush=True)
        if spans:
            num_shards = args.num_shards or _gpu_count()
            _launch_ranges(args, _split_spans(spans, num_shards))
    else:
        roots = [Path(path) for path in args.data_dir]
        folders = scan_episode_folders(roots)
        early = [folder for folder in folders if _is_early(folder)]
        infer_folders = [folder for folder in folders if not _is_early(folder)]
        write_early_positive(early, num_workers=8)
        print(f"[INDEX] infer_episodes={len(infer_folders)}", flush=True)
        jobs, frames = build_jobs(infer_folders, args.lookahead)
        with open(jobs_path, "wb") as handle:
            pickle.dump(jobs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(frames_path, frames)
        values = np.memmap(values_path, dtype=np.float32, mode="w+", shape=(len(frames),))
        values[:] = np.nan
        values.flush()
        del values
        print(f"[INDEX] infer_frames={len(frames)}", flush=True)
        num_shards = args.num_shards or _gpu_count()
        if len(frames) == 0:
            num_shards = 0
        if num_shards > 1:
            _launch_shards(args, num_shards)
        elif num_shards == 1:
            infer_shard(args, jobs, frames, 0, 1)

    values = np.memmap(values_path, dtype=np.float32, mode="r", shape=(len(frames),))
    if len(frames) and not np.isfinite(values).all():
        raise RuntimeError(f"推理结果不完整: nan={int(np.isnan(values).sum())}")
    advantages = []
    value_maps = []
    for job in jobs:
        value_map = np.full((len(job["kept"]),), np.nan, dtype=np.float32)
        value_maps.append(value_map)
    for (episode_index, kept_pos), value in zip(frames, values, strict=True):
        value_maps[int(episode_index)][int(kept_pos)] = float(value)
    for job, value_map in zip(jobs, value_maps, strict=True):
        for pos in np.flatnonzero(job["intervention"] == 0).tolist():
            reward_end = min(pos + args.lookahead, len(job["kept"]))
            reward_sum = float(np.sum(job["reward"][pos:reward_end]))
            future = 0.0 if pos + args.lookahead >= len(job["kept"]) else float(value_map[pos + args.lookahead])
            advantages.append(reward_sum + future - float(value_map[pos]))
    if advantages:
        threshold = float(np.percentile(np.asarray(advantages, dtype=np.float32), 100.0 - args.top_percent))
    else:
        threshold = 0.0
    print(
        f"[THRESHOLD] top={args.top_percent} threshold={threshold} autonomous={len(advantages)}",
        flush=True,
    )
    checkpoint_name = args.checkpoint_name or Path(args.checkpoint_dir).name
    write_inferred(jobs, frames, np.asarray(values), threshold, args.lookahead, f"vlm_{checkpoint_name}")
