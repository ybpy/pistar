"""Use the VLM value function to label advantage indicators for LeRobot datasets.

Pipeline:
1) Classify each episode by `intervention`: all-1 episodes are demos and are skipped;
   episodes with any 0 are rollouts and are fully relabeled.
2) Run VLM value inference for rollout rows and the lookahead endpoint rows
   needed to compute their N-step advantage.
3) Convert 201-dim logits -> softmax -> expectation over supports in [-1.0, 0.0].
4) Compute N-step Advantage per rollout time step:
   A_t = sum_{k=0}^{N-1} r_{t+k} + V_{t+N} - V_t
   If t+N >= T, use V_{T-1} and sum rewards until the end.
5) Compute the percentile threshold over rollout advantages of non-intervention steps.
6) For rollout rows only:
   - if `intervention = 1`, set `adv_ind = positive`
   - if `intervention = 0`, mark the configured top percentage as `positive`, otherwise `negative`
   Existing labels on rollout rows are overwritten; demo rows are preserved.

LeRobot Dataset Format:
- Key columns: intervention, reward_label, image, wrist_image, task_index
- Intervention field: dtype=int64, shape=[1], intervention_flag
- Reward field: dtype=float32, shape=[1], reward_label

Usage:
  python scripts/label_advantage_from_vlm.py \
    --data_dir /path/to/lerobot_dataset \
    --checkpoint_dir /path/to/value_ckpt_dir
"""

from __future__ import annotations

import os

if os.getenv("OPENPI_SILENCE_TF", "1") != "0":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")
    os.environ.setdefault("ABSL_LOG_SEVERITY_THRESHOLD", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")

import argparse
import dataclasses
import io
import json
import logging
import math
import multiprocessing
import sys
from pathlib import Path
from typing import Any

from datasets.features.features import Features, _FEATURE_TYPES
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

if "List" not in _FEATURE_TYPES:
    _FEATURE_TYPES["List"] = _FEATURE_TYPES["Sequence"]

_original_from_arrow_schema = Features.from_arrow_schema


def _patched_from_arrow_schema(arrow_schema):
    if hasattr(arrow_schema, "metadata") and arrow_schema.metadata:
        metadata = dict(arrow_schema.metadata)
        if b"info" in metadata:
            try:
                info = json.loads(metadata[b"info"])

                def _replace_list(feature_dict):
                    for _, value in feature_dict.items():
                        if isinstance(value, dict):
                            if value.get("_type") == "List":
                                value["_type"] = "Sequence"
                            _replace_list(value)

                if "features" in info:
                    _replace_list(info["features"])
                metadata[b"info"] = json.dumps(info).encode()
                arrow_schema = arrow_schema.with_metadata(metadata)
            except Exception:
                pass
    return _original_from_arrow_schema(arrow_schema)


Features.from_arrow_schema = _patched_from_arrow_schema

# Add project root to Python path to handle module import issues
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from openpi.shared import console
from openpi.shared import progress
import openpi.transforms as _transforms


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

# Force single GPU usage for inference
os.environ.setdefault("JAX_PLATFORMS", "cuda")

LOG = logging.getLogger("openpi")

POS_SUFFIX = ". Advantage: positive"
NEG_SUFFIX = ". Advantage: negative"
REWARD_COLUMN_CANDIDATES = ("reward_label",)


def _resolve_checkpoint_path(checkpoint_dir: Path, checkpoint_name: str | None) -> Path:
    checkpoint_dir = checkpoint_dir.resolve()
    if checkpoint_name:
        checkpoint_path = (checkpoint_dir / checkpoint_name).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("step_"):
        return checkpoint_dir

    candidates = sorted(checkpoint_dir.glob("step_*"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in: {checkpoint_dir}")
    return candidates[-1]


def _load_checkpoint_params(checkpoint_path: Path, *, use_ema: bool) -> dict:
    """Load checkpoint params, handling device mismatch for inference."""
    import jax
    import orbax.checkpoint as ocp

    # Get available devices
    available_devices = jax.devices()
    target_device = available_devices[0]  # Use first available device for inference

    def _single_device_sharding(device: jax.Device) -> jax.sharding.Sharding:
        try:
            return jax.sharding.SingleDeviceSharding(device)
        except Exception:
            mesh = jax.sharding.Mesh([device], ("x",))
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    sharding = _single_device_sharding(target_device)

    LOG.info(f"Loading checkpoint with {len(available_devices)} device(s) available")
    LOG.info(f"Target device for inference: {target_device}")

    with ocp.PyTreeCheckpointer() as ckptr:
        # Load with explicit restore args to avoid sharding=None issues in newer JAX/Orbax.
        def _restore_with(restore_type: type[np.ndarray] | type[jax.Array]) -> dict:
            metadata = ckptr.metadata(str(checkpoint_path))
            if isinstance(metadata, dict):
                item = {k: metadata[k] for k in ("params", "ema_params") if k in metadata}
                if not item:
                    item = metadata
            else:
                item = metadata
            restore_args = jax.tree_util.tree_map(
                lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type),
                item,
            )
            return ckptr.restore(
                str(checkpoint_path),
                ocp.args.PyTreeRestore(item=item, restore_args=restore_args),
            )

        try:
            restored = _restore_with(jax.Array)
        except Exception as e:
            LOG.warning(f"Restore with jax.Array failed: {e}, falling back to numpy arrays...")
            try:
                restored = _restore_with(np.ndarray)
            except Exception as e2:
                LOG.error(f"Restore with numpy arrays also failed: {e2}")
                raise RuntimeError(
                    f"Failed to load checkpoint from {checkpoint_path}. "
                    f"Original error: {e}, Fallback error: {e2}"
                )

    if use_ema and "ema_params" in restored:
        params = restored["ema_params"]
    elif "params" not in restored:
        raise KeyError(f"Checkpoint missing 'params': {checkpoint_path}")
    else:
        params = restored["params"]

    # Final device placement check
    def _array_device(arr: jax.Array) -> jax.Device:
        dev = arr.device
        if callable(dev):
            return dev()
        return dev

    def ensure_single_device(item):
        """Ensure all arrays are on the target device."""
        if isinstance(item, jax.Array):
            if _array_device(item) != target_device:
                try:
                    return jax.device_put(jax.device_get(item), target_device)
                except Exception:
                    return item
            return item
        if isinstance(item, np.ndarray):
            return jax.device_put(item, target_device)
        return item

    params = jax.tree_util.tree_map(ensure_single_device, params)

    LOG.info("Successfully loaded checkpoint and ensured all parameters are on target device")
    return params


def _load_tasks_map(data_dir: Path) -> dict[int, str]:
    tasks_file = data_dir / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        raise ValueError(f"Task file not found: {tasks_file}")
    tasks: dict[int, str] = {}
    with open(tasks_file, "r") as f:
        for line in f:
            task_data = json.loads(line.strip())
            tasks[int(task_data["task_index"])] = task_data["task"]
    return tasks


def _resolve_local_gemma_tokenizer_path(tokenizer_path: str | None) -> str | None:
    if tokenizer_path:
        path = Path(tokenizer_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"指定的 tokenizer_path 不存在: {path}")
        return str(path)

    candidates: list[Path] = []
    for env_name in ("OPENPI_VALUE_TOKENIZER_PATH", "GEMMA_TOKENIZER_PATH"):
        raw = os.getenv(env_name)
        if raw:
            candidates.append(Path(raw).expanduser())

    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend(
        [
            repo_root / "gemma" / "tokenizer.model",
            Path("/public/home/wangsenbao_it/litianheng/checkpoint/tokenizer.model"),
            Path("/public/home/wangsenbao_it/litianheng/checkpoint/gemma-3-270m/tokenizer.model"),
            Path("/data/train_dataset/checkpoint/gemma-3-270m/tokenizer.model"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _validate_gemma_tokenizer(tokenizer_path: str | None) -> None:
    gemma_path = Path(__file__).resolve().parent.parent / "gemma"
    if str(gemma_path) not in sys.path:
        sys.path.insert(0, str(gemma_path))

    from gemma.gm.text._tokenizer import Gemma3Tokenizer

    try:
        tokenizer = Gemma3Tokenizer(path=tokenizer_path) if tokenizer_path else Gemma3Tokenizer()
        tokenizer.encode("sanity check\nValue:", add_bos=True, add_eos=False)
    except Exception as exc:
        source = tokenizer_path or "Gemma3Tokenizer 默认路径"
        raise RuntimeError(
            f"无法初始化 Gemma3 tokenizer: {source}. "
            "请显式传入 --tokenizer_path /path/to/tokenizer.model"
        ) from exc


class GemmaValueTokenizer:
    """Lazy tokenizer wrapper so DataLoader workers initialize Gemma locally."""

    def __init__(self, max_len: int = 48, tokenizer_path: str | None = None):
        self._max_len = max_len
        self._tokenizer_path = tokenizer_path
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer

        gemma_path = Path(__file__).resolve().parent.parent / "gemma"
        if str(gemma_path) not in sys.path:
            sys.path.insert(0, str(gemma_path))

        from gemma.gm.text._tokenizer import Gemma3Tokenizer

        if self._tokenizer_path:
            self._tokenizer = Gemma3Tokenizer(path=self._tokenizer_path)
        else:
            self._tokenizer = Gemma3Tokenizer()
        return self._tokenizer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        return state

    def tokenize(self, prompt: str, state: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
        del state

        tokenizer = self._get_tokenizer()
        text = f"{str(prompt).rstrip()}\nValue:"
        tokens = tokenizer.encode(text, add_bos=True, add_eos=False)
        if len(tokens) > self._max_len:
            tokens = tokens[: self._max_len]
        else:
            tokens = tokens + [0] * (self._max_len - len(tokens))

        tokens = np.asarray(tokens, dtype=np.int32)
        mask = tokens != 0
        return tokens, mask


def _normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.rstrip()
    if text.endswith(POS_SUFFIX):
        return text[: -len(POS_SUFFIX)].rstrip()
    if text.endswith(NEG_SUFFIX):
        return text[: -len(NEG_SUFFIX)].rstrip()
    return text


def _append_label(text: str, *, positive: bool) -> str:
    base = _normalize_text(text)
    suffix = POS_SUFFIX if positive else NEG_SUFFIX
    return f"{base}{suffix}"


def _to_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return value != 0
    if isinstance(value, float):
        if np.isnan(value):
            return False
        return value != 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "y", "t"):
            return True
        if lowered in ("0", "false", "no", "n", "f", ""):
            return False
        return lowered in ("human", "intervention", "expert", "teleop")
    return False


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return np.asarray(value)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_scalar_int(value: Any) -> int:
    array = _as_numpy(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar integer-like value, got shape {array.shape}")
    return int(array.reshape(()))


def _to_scalar_float(value: Any, *, name: str = "value") -> float:
    array = _as_numpy(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar float-like {name}, got shape {array.shape}")
    return float(np.asarray(array, dtype=np.float32).reshape(()))


def _to_int_array(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    return np.asarray(array, dtype=np.int64).reshape(-1)


def _column_candidates(name: str | None) -> list[str]:
    if not name:
        return []
    if "/" in name or "." in name:
        return [name]
    return [name, f"observation/{name}", f"observation.{name}"]


def _get_by_path(data: dict[str, Any], path: str) -> tuple[Any, bool]:
    if path in data:
        return data[path], True

    def _traverse(parts: list[str]) -> tuple[Any, bool]:
        cur: Any = data
        for part in parts:
            if not isinstance(cur, dict) or part not in cur:
                return None, False
            cur = cur[part]
        return cur, True

    if "/" in path:
        value, ok = _traverse(path.split("/"))
        if ok:
            return value, True
    if "." in path:
        value, ok = _traverse(path.split("."))
        if ok:
            return value, True
    return None, False


def _get_first(data: dict[str, Any], candidates: list[str], *, required: bool = False, name: str = "value") -> Any:
    for key in candidates:
        value, ok = _get_by_path(data, key)
        if ok:
            return value
    if required:
        raise KeyError(f"Missing {name}. Tried keys: {candidates}. Available top-level keys: {list(data.keys())}")
    return None


def _parse_image(image: Any) -> np.ndarray:
    if image is None:
        raise ValueError("Image is None")

    if isinstance(image, dict) and "bytes" in image:
        image = image["bytes"]
    elif isinstance(image, dict) and "path" in image and image["path"] is not None:
        image = image["path"]

    if isinstance(image, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
    if isinstance(image, str):
        return np.asarray(Image.open(image).convert("RGB"))
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class LabelAdvantageInputs(_transforms.DataTransformFn):
    base_image_col: str
    wrist_image_col: str | None
    right_wrist_image_col: str | None
    copy_wrist_to_right: bool
    instruction_col: str | None
    tasks_map: dict[int, str] | None
    camera_names: tuple[str, ...] | None = None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        prompt = self._extract_prompt(data)
        from openpi.policies.value_policy import ValueInputs
        from openpi.shared.value_cameras import CAMERA_DATA_KEYS, image_is_present
        names = self.camera_names
        if names is None:
            names = ("head", "left", "right") if self.right_wrist_image_col or self.copy_wrist_to_right else ("head", "left")
        sample = {"prompt": prompt, "value": np.float32(0)}
        raw = {}
        for name, column in (("head", self.base_image_col), ("left", self.wrist_image_col),
                             ("right", self.right_wrist_image_col)):
            if column:
                value = _get_first(data, _column_candidates(column), required=False, name=name)
                if image_is_present(value):
                    raw[name] = value
        if self.copy_wrist_to_right and "right" not in raw and "left" in raw:
            raw["right"] = raw["left"]
        for name, value in raw.items():
            sample[f"observation.images.{CAMERA_DATA_KEYS[name]}"] = value
        state = _get_first(data, ["state", "observation/state", "observation.state"], required=False, name="state")
        if state is not None:
            sample["state"] = state
        result = ValueInputs(camera_names=names)(sample)
        result.pop("value")

        for meta_key in ("episode_index", "frame_index", "index"):
            value = _get_first(data, _column_candidates(meta_key), required=False, name=meta_key)
            if value is not None:
                result[meta_key] = np.asarray(_to_scalar_int(value), dtype=np.int64)

        return result

    def _extract_prompt(self, data: dict[str, Any]) -> str:
        candidates: list[str] = []
        if self.instruction_col:
            candidates.extend(_column_candidates(self.instruction_col))
        else:
            for name in ("prompt", "instruction", "task", "language_instruction", "text"):
                candidates.extend(_column_candidates(name))

        prompt = _get_first(data, candidates, required=False, name="prompt")
        if prompt is not None:
            array = _as_numpy(prompt)
            if array.size == 1:
                return str(array.reshape(()).item())
            return str(prompt)

        if self.tasks_map is not None:
            task_index = _get_first(data, _column_candidates("task_index"), required=False, name="task_index")
            if task_index is not None:
                return self.tasks_map.get(_to_scalar_int(task_index), "unknown task")

        raise ValueError(f"Unable to resolve prompt for sample. Available keys: {list(data.keys())}")


def _select_reward_column(df: pd.DataFrame, specified: str | None) -> str:
    if specified is not None:
        if specified in df.columns:
            return specified
        raise ValueError(f"Reward column '{specified}' not found in available columns: {list(df.columns)}")

    for candidate in REWARD_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "No reward label column found. Tried columns: "
        f"{list(REWARD_COLUMN_CANDIDATES)}. Available columns: {list(df.columns)}"
    )


def _series_to_scalar_float_array(series: pd.Series, *, name: str) -> np.ndarray:
    return np.asarray([_to_scalar_float(value, name=name) for value in series.tolist()], dtype=np.float32)


def _series_to_scalar_int_array(series: pd.Series, *, name: str) -> np.ndarray:
    return np.asarray([_to_scalar_int(value) for value in series.tolist()], dtype=np.int64)


def _to_scalar_str(value: Any, *, name: str = "value") -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float) and np.isnan(value):
        return ""

    array = _as_numpy(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar string-like {name}, got shape {array.shape}")

    scalar = array.reshape(()).item()
    if scalar is None:
        return ""
    if isinstance(scalar, float) and np.isnan(scalar):
        return ""
    return str(scalar)


def _extract_human_mask(df: pd.DataFrame, human_col: str | None) -> np.ndarray:
    if not human_col or human_col not in df.columns:
        return np.zeros((len(df),), dtype=bool)

    intervention_data = df[human_col].tolist()
    if not intervention_data:
        return np.zeros((0,), dtype=bool)

    if isinstance(intervention_data[0], dict) and "bytes" not in intervention_data[0]:
        human_mask = []
        for val in intervention_data:
            if isinstance(val, dict) and "scalar" in val:
                human_mask.append(val["scalar"] == 1)
            else:
                human_mask.append(_to_bool(val))
        return np.asarray(human_mask, dtype=bool)

    return np.asarray([_to_bool(v) for v in intervention_data], dtype=bool)


def _resolve_step_indices(df: pd.DataFrame) -> np.ndarray:
    if "frame_index" in df.columns:
        return _series_to_scalar_int_array(df["frame_index"], name="frame_index")
    if "index" in df.columns:
        return _series_to_scalar_int_array(df["index"], name="index")
    raise ValueError("数据中缺少 frame_index/index，无法对齐 advantage 所需的推理结果")


def _resolve_dataset_row_indices(df: pd.DataFrame, flat_offset: int) -> np.ndarray:
    if "index" in df.columns:
        return _series_to_scalar_int_array(df["index"], name="index")
    return np.arange(flat_offset, flat_offset + len(df), dtype=np.int64)


def _update_info_json(info_path: Path, column_name: str, *, description: str):
    if not info_path.exists():
        return
    with open(info_path, "r") as f:
        info = json.load(f)
    if "features" not in info:
        info["features"] = {}
    if column_name not in info["features"]:
        info["features"][column_name] = {
            "dtype": "string",
            "shape": [1],
            "description": description,
        }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)


@dataclasses.dataclass
class InferredValueCache:
    flat_values: list[float] = dataclasses.field(default_factory=list)
    values_by_episode: dict[int, list[float]] = dataclasses.field(default_factory=dict)
    values_by_episode_frame: dict[int, dict[int, float]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PendingLabelRequest:
    parquet_path: Path
    episode_id: int
    reward_labels: np.ndarray
    human_mask: np.ndarray
    pending_positions: np.ndarray
    future_positions: np.ndarray
    step_indices: np.ndarray


class IndexedDataset:
    def __init__(self, dataset, indices: list[int]):
        self._dataset = dataset
        self._indices = indices

    def __getitem__(self, index: int):
        return self._dataset[self._indices[index]]

    def __len__(self) -> int:
        return len(self._indices)


def _get_prefetch_factor(num_workers: int) -> int | None:
    if num_workers <= 0:
        return None
    prefetch_env = os.getenv("OPENPI_VALUE_PREFETCH_FACTOR")
    if prefetch_env:
        try:
            return max(1, int(prefetch_env))
        except ValueError:
            LOG.warning(console.warn(f"OPENPI_VALUE_PREFETCH_FACTOR='{prefetch_env}' 无效，使用默认值 4"))
    return 4


def _build_inference_dataset(
    *,
    data_dir: Path,
    model_config: ValueModelConfig,
    tokenizer_path: str | None,
    instruction_col: str | None,
    base_image_col: str,
    wrist_image_col: str | None,
    right_wrist_image_col: str | None,
    copy_wrist_to_right: bool,
    tasks_map: dict[int, str] | None,
):
    from openpi.models.value_model_config import ValueModelConfig
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    data_config = _config.DataConfig(
        local_data_dir=str(data_dir),
        prompt_from_task=False,
        data_transforms=_transforms.Group(
            inputs=[
                LabelAdvantageInputs(
                    base_image_col=base_image_col,
                    wrist_image_col=wrist_image_col,
                    right_wrist_image_col=right_wrist_image_col,
                    copy_wrist_to_right=copy_wrist_to_right,
                    instruction_col=instruction_col,
                    tasks_map=tasks_map,
                    camera_names=model_config.camera_names,
                )
            ]
        ),
        model_transforms=_transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    GemmaValueTokenizer(model_config.max_token_len, tokenizer_path=tokenizer_path)
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ]
        ),
    )
    dataset = _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)
    return _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=True)


def _pop_batch_key(batch: dict[str, Any], key: str) -> np.ndarray | None:
    value = batch.pop(key, None)
    if value is None:
        return None
    return _to_int_array(value)


def _compute_values_with_dataloader(
    *,
    dataset,
    model,
    supports,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> InferredValueCache:
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    import openpi.training.data_loader as _data_loader

    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
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
    prefetch_factor = _get_prefetch_factor(num_workers)
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    torch_loader = torch.utils.data.DataLoader(**loader_kwargs)
    cache = InferredValueCache()
    rng = jax.random.key(0)

    @jax.jit
    def infer_fn(observation: _model.Observation) -> jax.Array:
        logits = model(observation, train=False)
        probs = jax.nn.softmax(logits, axis=-1)
        return jnp.sum(probs * supports, axis=-1)

    pbar = tqdm(
        torch_loader,
        total=math.ceil(len(dataset) / batch_size),
        desc="VLM value 推理",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    progress.sync_pbar_color(pbar)
    for batch in pbar:
        progress.sync_pbar_color(pbar)
        episode_indices = _pop_batch_key(batch, "episode_index")
        frame_indices = _pop_batch_key(batch, "frame_index")
        if frame_indices is None:
            frame_indices = _pop_batch_key(batch, "index")

        observation = _model.Observation.from_dict(batch)
        available_keys = list(observation.images.keys())
        observation = _model.preprocess_observation(rng, observation, train=False, image_keys=available_keys)
        values_batch = np.asarray(jax.device_get(infer_fn(observation)), dtype=np.float32)
        cache.flat_values.extend(values_batch.tolist())

        if episode_indices is None:
            continue

        if frame_indices is None:
            for episode_idx, value in zip(episode_indices, values_batch, strict=True):
                cache.values_by_episode.setdefault(int(episode_idx), []).append(float(value))
            continue

        for episode_idx, frame_idx, value in zip(episode_indices, frame_indices, values_batch, strict=True):
            cache.values_by_episode_frame.setdefault(int(episode_idx), {})[int(frame_idx)] = float(value)

    return cache


def _extract_episode_id(df: pd.DataFrame, fallback: int) -> int:
    if "episode_index" not in df.columns or df.empty:
        return fallback
    return _to_scalar_int(df["episode_index"].iloc[0])


def _resolve_values_for_episode(
    cache: InferredValueCache,
    *,
    episode_id: int,
    episode_len: int,
    flat_offset: int,
) -> np.ndarray:
    if episode_id in cache.values_by_episode_frame:
        frame_map = cache.values_by_episode_frame[episode_id]
        if len(frame_map) != episode_len:
            raise ValueError(
                f"Episode {episode_id} 的 value 数量与 parquet 长度不匹配: {len(frame_map)} vs {episode_len}"
            )
        values = np.zeros((episode_len,), dtype=np.float32)
        for frame_idx, value in frame_map.items():
            if frame_idx < 0 or frame_idx >= episode_len:
                raise ValueError(f"Episode {episode_id} 存在越界 frame_index={frame_idx}, 长度={episode_len}")
            values[frame_idx] = np.float32(value)
        return values

    if episode_id in cache.values_by_episode:
        values = np.asarray(cache.values_by_episode[episode_id], dtype=np.float32)
        if len(values) != episode_len:
            raise ValueError(
                f"Episode {episode_id} 的 value 数量与 parquet 长度不匹配: {len(values)} vs {episode_len}"
            )
        return values

    end = flat_offset + episode_len
    if end > len(cache.flat_values):
        raise ValueError(
            f"推理得到的总 value 数量不足: 需要切片到 {end}, 实际只有 {len(cache.flat_values)}"
        )
    return np.asarray(cache.flat_values[flat_offset:end], dtype=np.float32)


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    requested = [column for column in columns if column]
    try:
        return pd.read_parquet(path, columns=requested)
    except Exception:
        return pd.read_parquet(path)


def _compute_advantages(values: np.ndarray, reward_labels: np.ndarray, lookahead: int) -> np.ndarray:
    T = len(values)
    advantages = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        if t + lookahead < T:
            reward_sum = float(np.sum(reward_labels[t : t + lookahead]))
            future_value = float(values[t + lookahead])
        else:
            reward_sum = float(np.sum(reward_labels[t:]))
            future_value = 0.0
        advantages[t] = reward_sum + future_value - float(values[t])
    return advantages


def _lookup_inferred_value(cache: InferredValueCache, *, episode_id: int, step_index: int) -> float:
    if episode_id not in cache.values_by_episode_frame:
        raise KeyError(f"Episode {episode_id} 缺少按帧索引缓存的 value 推理结果")

    frame_map = cache.values_by_episode_frame[episode_id]
    if step_index not in frame_map:
        raise KeyError(f"Episode {episode_id} 缺少 step_index={step_index} 的 value 推理结果")
    return float(frame_map[step_index])


def _build_pending_label_requests(
    parquet_files: list[Path],
    *,
    reward_col: str | None,
    human_col: str | None,
    lookahead: int,
) -> tuple[list[PendingLabelRequest], list[int], int, int, int]:
    requests: list[PendingLabelRequest] = []
    selected_dataset_indices: set[int] = set()
    total_rows = 0
    demo_rows = 0
    rollout_rows = 0
    flat_offset = 0

    pbar = tqdm(
        parquet_files,
        desc="扫描 rollout 帧",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    progress.sync_pbar_color(pbar)
    for fallback_episode_id, parquet_path in enumerate(pbar):
        progress.sync_pbar_color(pbar)
        columns = list(
            dict.fromkeys(
                [
                    "episode_index",
                    "frame_index",
                    "index",
                    reward_col,
                    *REWARD_COLUMN_CANDIDATES,
                    human_col,
                ]
            )
        )
        df = _read_parquet_columns(parquet_path, columns)
        total_rows += len(df)

        reward_col_name = _select_reward_column(df, reward_col)
        reward_labels = _series_to_scalar_float_array(df[reward_col_name], name=reward_col_name)
        human_mask = _extract_human_mask(df, human_col)

        if np.all(human_mask):
            demo_rows += len(df)
            flat_offset += len(df)
            continue

        rollout_rows += len(df)
        episode_id = _extract_episode_id(df, fallback_episode_id)
        step_indices = _resolve_step_indices(df)
        dataset_row_indices = _resolve_dataset_row_indices(df, flat_offset)

        pending_positions = np.arange(len(df), dtype=np.int64)
        future_positions = np.minimum(pending_positions + lookahead, len(df) - 1).astype(np.int64)

        selected_dataset_indices.update(dataset_row_indices[pending_positions].tolist())
        selected_dataset_indices.update(dataset_row_indices[future_positions].tolist())

        requests.append(
            PendingLabelRequest(
                parquet_path=parquet_path,
                episode_id=episode_id,
                reward_labels=reward_labels,
                human_mask=human_mask,
                pending_positions=pending_positions,
                future_positions=future_positions,
                step_indices=step_indices,
            )
        )
        flat_offset += len(df)

    return requests, sorted(selected_dataset_indices), total_rows, demo_rows, rollout_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Label advantage indicators using the VLM value model")
    parser.add_argument("--data_dir", type=str, nargs="+", required=True, help="LeRobot 数据集或 AgileX HDF5 根目录")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint directory containing step_*")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="Specific checkpoint folder name")
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True, help="Use EMA params if present")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional local Gemma3 tokenizer.model path")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for value inference")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader worker 数量")
    parser.add_argument("--decode_workers", type=int, default=None, help="兼容旧参数，等同于 --num_workers")
    parser.add_argument("--lookahead", type=int, default=50, help="N-step lookahead for Advantage")
    parser.add_argument(
        "--top_percent",
        type=float,
        default=30.0,
        help="Percentage of non-intervention rollout samples with highest advantage to label positive (default: 30.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="DataLoader 随机种子")
    parser.add_argument(
        "--reward_col",
        type=str,
        default="reward_label",
        help="Reward label column name (default: reward_label)",
    )
    parser.add_argument("--human_col", type=str, default="intervention", help="Human intervention flag column name (default: 'intervention')")
    parser.add_argument("--instruction_col", type=str, default=None, help="Instruction/prompt column name to read")
    parser.add_argument("--adv_col", type=str, default="adv_ind", help="Advantage indicator column name to create")
    parser.add_argument("--camera_names", nargs="+", choices=("head", "left", "right"), default=None,
                        help="1-3 cameras; read checkpoint camera metadata, otherwise legacy head+left")
    parser.add_argument("--base_image_col", type=str, default="image", help="Base image column name")
    parser.add_argument("--wrist_image_col", type=str, default="wrist_image", help="Wrist image column name")
    parser.add_argument("--right_wrist_image_col", type=str, default=None, help="Right wrist image column name")
    parser.add_argument(
        "--copy_wrist_to_right",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true and right wrist is missing, copy wrist image",
    )
    parser.add_argument("--num_shards", type=int, default=None, help="HDF5 推理使用的 GPU 数")
    parser.add_argument("--infer_shard", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frame_start", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frame_end", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help="沿用已有 values.dat，只补未完成的帧并写 adv_ind")
    parser.add_argument("--work_dir", type=str, default="/tmp/adv_ind_hdf5", help="HDF5 推理中间文件目录")
    args = parser.parse_args()
    if not 0.0 < args.top_percent <= 100.0:
        parser.error("--top_percent must be in (0, 100]")

    from openpi.training.agilex_hdf5_value_dataset import looks_like_agilex_hdf5_dirs

    if looks_like_agilex_hdf5_dirs(args.data_dir):
        import label_advantage_hdf5

        label_advantage_hdf5.run(args)
        return
    if len(args.data_dir) != 1:
        raise ValueError("LeRobot 模式只接受一个 data_dir")
    args.data_dir = args.data_dir[0]

    import jax
    import jax.numpy as jnp
    from openpi.models.value_model_config import ValueModelConfig

    logging.basicConfig(level=logging.INFO, force=True)
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir:
        jax.config.update("jax_compilation_cache_dir", cache_dir)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"数据目录不存在: {data_dir}")
    parquet_dir = data_dir / "data"
    if not parquet_dir.exists():
        raise ValueError(f"找不到 data 目录: {parquet_dir}")
    parquet_files = sorted(parquet_dir.rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"找不到 parquet 文件: {parquet_dir}")

    pending_requests, selected_dataset_indices, total_rows, demo_rows, rollout_rows = _build_pending_label_requests(
        parquet_files,
        reward_col=args.reward_col,
        human_col=args.human_col,
        lookahead=args.lookahead,
    )

    LOG.info("数据集总帧数: %d, 跳过 demo 帧数: %d, 待更新 rollout 帧数: %d", total_rows, demo_rows, rollout_rows)
    if rollout_rows == 0:
        print(console.ok("完成: 未发现 rollout 帧，无需执行 VLM 推理"))
        return

    context_frames = len(selected_dataset_indices) - rollout_rows
    LOG.info("实际参与 VLM value 推理的帧数: %d (含 %d 帧 lookahead 终点帧)", len(selected_dataset_indices), context_frames)

    checkpoint_path = _resolve_checkpoint_path(Path(args.checkpoint_dir), args.checkpoint_name)
    LOG.info("Using checkpoint: %s", checkpoint_path)

    from openpi.shared.value_cameras import resolve_camera_names
    names = resolve_camera_names(args.camera_names, checkpoint_path,
        default=("head", "left", "right") if args.right_wrist_image_col or args.copy_wrist_to_right else ("head", "left"))
    config = ValueModelConfig(camera_names=names)
    if "right" in names and args.right_wrist_image_col is None:
        args.right_wrist_image_col = "right_wrist_image"
    LOG.info("Value cameras: %s", names)
    params = _load_checkpoint_params(checkpoint_path, use_ema=args.use_ema)
    model = config.load(params, remove_extra_params=True)

    supports = jnp.linspace(-1.0, 0.0, 201, dtype=jnp.float32)

    max_workers = args.num_workers
    if max_workers is None:
        max_workers = args.decode_workers if args.decode_workers is not None else 2
    max_workers = max(0, max_workers)

    env_workers = os.getenv("OPENPI_VALUE_NUM_WORKERS")
    if env_workers is not None:
        try:
            requested_workers = int(env_workers)
            if requested_workers < 0:
                raise ValueError
            cpu_count = os.cpu_count() or 1
            max_workers = min(requested_workers, cpu_count)
            if requested_workers > cpu_count:
                LOG.info(console.info(f"OPENPI_VALUE_NUM_WORKERS={requested_workers} 超过CPU核数，已裁剪为 {max_workers}"))
        except ValueError:
            LOG.warning(console.warn(f"OPENPI_VALUE_NUM_WORKERS='{env_workers}' 无效，保持 num_workers={max_workers}"))

    resolved_tokenizer_path = _resolve_local_gemma_tokenizer_path(args.tokenizer_path)
    if resolved_tokenizer_path is not None:
        LOG.info(console.info(f"使用本地 Gemma3 tokenizer: {resolved_tokenizer_path}"))
    elif max_workers > 0:
        LOG.warning(
            console.warn(
                "未找到本地 Gemma3 tokenizer；多进程 DataLoader 中会在 worker 内触发默认远程加载，"
                "容易报错。已自动将 num_workers 降为 0，请尽量显式传入 --tokenizer_path。"
            )
        )
        max_workers = 0

    _validate_gemma_tokenizer(resolved_tokenizer_path)

    tasks_map: dict[int, str] | None = None
    tasks_file = data_dir / "meta" / "tasks.jsonl"
    if tasks_file.exists():
        tasks_map = _load_tasks_map(data_dir)

    dataset = _build_inference_dataset(
        data_dir=data_dir,
        model_config=config,
        tokenizer_path=resolved_tokenizer_path,
        instruction_col=args.instruction_col,
        base_image_col=args.base_image_col,
        wrist_image_col=args.wrist_image_col,
        right_wrist_image_col=args.right_wrist_image_col,
        copy_wrist_to_right=args.copy_wrist_to_right,
        tasks_map=tasks_map,
    )
    dataset = IndexedDataset(dataset, selected_dataset_indices)
    LOG.info(console.info(f"待推理数据集大小: {len(dataset)} 帧"))
    value_cache = _compute_values_with_dataloader(
        dataset=dataset,
        model=model,
        supports=supports,
        batch_size=args.batch_size,
        num_workers=max_workers,
        seed=args.seed,
    )

    advs_autonomous: list[np.ndarray] = []
    advantage_cache: dict[Path, np.ndarray] = {}

    pbar = tqdm(
        pending_requests,
        desc="计算待更新优势",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    progress.sync_pbar_color(pbar)
    for request in pbar:
        progress.sync_pbar_color(pbar)
        advantages = np.zeros((len(request.pending_positions),), dtype=np.float32)

        for i, row_pos in enumerate(request.pending_positions):
            row_pos = int(row_pos)
            reward_end = min(row_pos + args.lookahead, len(request.reward_labels))
            reward_sum = float(np.sum(request.reward_labels[row_pos:reward_end]))
            current_value = _lookup_inferred_value(
                value_cache,
                episode_id=request.episode_id,
                step_index=int(request.step_indices[row_pos]),
            )
            if row_pos + args.lookahead < len(request.reward_labels):
                future_pos = int(request.future_positions[i])
                future_value = _lookup_inferred_value(
                    value_cache,
                    episode_id=request.episode_id,
                    step_index=int(request.step_indices[future_pos]),
                )
            else:
                future_value = 0.0
            advantages[i] = reward_sum + future_value - current_value

        advantage_cache[request.parquet_path] = advantages

        autonomous_mask = ~request.human_mask[request.pending_positions]
        if np.any(autonomous_mask):
            advs_autonomous.append(advantages[autonomous_mask])

    if len(value_cache.flat_values) != len(selected_dataset_indices):
        LOG.warning(
            console.warn(
                f"value 推理总数与目标帧数不完全一致: values={len(value_cache.flat_values)}, target_frames={len(selected_dataset_indices)}"
            )
        )

    threshold: float | None = None
    total_pending = sum(len(request.pending_positions) for request in pending_requests)
    total_human = sum(int(np.sum(request.human_mask[request.pending_positions])) for request in pending_requests)
    total_autonomous = total_pending - total_human
    LOG.info(
        "待更新 rollout 拆分: intervention=%d, non-intervention=%d",
        total_human,
        total_autonomous,
    )
    if advs_autonomous:
        all_autonomous = np.concatenate(advs_autonomous, axis=0)
        threshold_percentile = 100.0 - args.top_percent
        threshold = float(np.percentile(all_autonomous, threshold_percentile))
        selected_autonomous = int(np.sum(all_autonomous >= threshold))
        LOG.info(
            "Advantage threshold for top %.2f%% rollout non-intervention samples "
            "(percentile=%.2f): %.6f (selected=%d/%d)",
            args.top_percent,
            threshold_percentile,
            threshold,
            selected_autonomous,
            len(all_autonomous),
        )
    else:
        LOG.info("待更新样本中没有 intervention=0 的帧，跳过 top %.2f%% 阈值计算", args.top_percent)

    pbar = tqdm(
        pending_requests,
        desc="写回 Advantage 标签",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    progress.sync_pbar_color(pbar)
    updated_positive = 0
    updated_negative = 0
    for request in pbar:
        progress.sync_pbar_color(pbar)
        df = pd.read_parquet(request.parquet_path)
        if args.adv_col in df.columns:
            adv_labels = [_to_scalar_str(v, name=args.adv_col) for v in df[args.adv_col].tolist()]
        else:
            adv_labels = ["none"] * len(df)

        pending_human_mask = request.human_mask[request.pending_positions]
        pending_advantages = advantage_cache[request.parquet_path]
        positive_mask = pending_human_mask.copy()
        if threshold is not None:
            positive_mask |= (~pending_human_mask) & (pending_advantages >= threshold)

        updated_positive += int(np.sum(positive_mask))
        updated_negative += len(positive_mask) - int(np.sum(positive_mask))

        for row_pos, is_positive in zip(request.pending_positions.tolist(), positive_mask.tolist(), strict=True):
            adv_labels[int(row_pos)] = "positive" if bool(is_positive) else "negative"

        df[args.adv_col] = adv_labels
        df.to_parquet(request.parquet_path, index=False)

    info_path = data_dir / "meta" / "info.json"
    _update_info_json(
        info_path,
        args.adv_col,
        description="Advantage indicator label (positive/negative).",
    )

    print(console.ok(f"完成: 已更新 {rollout_rows} 帧，其中 positive={updated_positive}, negative={updated_negative}"))


if __name__ == "__main__":
    main()
