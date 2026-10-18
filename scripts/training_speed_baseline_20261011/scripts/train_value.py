"""Value function 训练脚本。

用法:
    python scripts/train_value.py \
        --data_dir /path/to/lerobot_dataset \
        --checkpoint_dir /path/to/save/checkpoints \
        --batch_size 32 \
        --num_train_steps 10000 \
        --load_pretrained  # 加载 PaliGemma 预训练权重
"""

import os

if os.getenv("OPENPI_SILENCE_TF", "1") != "0":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")
    os.environ.setdefault("ABSL_LOG_SEVERITY_THRESHOLD", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")

import argparse
from collections.abc import Sequence
import dataclasses
import functools
import json
import logging
from pathlib import Path
import platform
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm

from openpi.models import model as _model
from openpi.models.value_model_config import ValueModelConfig
from openpi.shared import array_typing as at
from openpi.shared import console
from openpi.shared import progress
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
from openpi.training.weight_loaders import ValueModelWeightLoader
import openpi.transforms as _transforms
import openpi.policies.value_policy as value_policy
from openpi.shared import nnx_utils


VALUE_LABEL_COLUMN = "value_label"
LEGACY_VALUE_LABEL_COLUMN = "value_lable"


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
    """Tokenizer wrapper aligned with the Gemma3 embedder used by ValueModel."""

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

    def tokenize(self, prompt: str, state: jnp.ndarray | None = None,
                 adv_ind: str | None = None, *, adv_ind_dropout: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        del state, adv_ind_dropout
        if adv_ind is not None:
            raise ValueError("Value-model tokenizer does not support advantage-indicator prompts")

        tokenizer = self._get_tokenizer()
        text = f"{str(prompt).rstrip()}\nValue:"
        tokens = tokenizer.encode(text, add_bos=True, add_eos=False)
        if len(tokens) > self._max_len:
            tokens = tokens[: self._max_len]
        else:
            tokens = tokens + [0] * (self._max_len - len(tokens))

        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        mask = tokens != 0
        return tokens, mask


@dataclasses.dataclass(frozen=True)
class RemapValueLabelKey(_transforms.DataTransformFn):
    source_keys: tuple[str, ...] = (VALUE_LABEL_COLUMN, LEGACY_VALUE_LABEL_COLUMN)
    target_key: str = "value"

    def __call__(self, data: dict) -> dict:
        source_key = next((key for key in self.source_keys if key in data), None)
        if source_key is None:
            raise KeyError(f"Missing value label column. Tried: {self.source_keys}")
        remapped = dict(data)
        remapped[self.target_key] = remapped[source_key]
        return remapped


def normalize_data_dirs(local_data_dir: str | Sequence[str]) -> str | tuple[str, ...]:
    dirs = (local_data_dir,) if isinstance(local_data_dir, str) else tuple(local_data_dir)
    if not dirs:
        raise ValueError("data_dir 为空")
    return dirs[0] if len(dirs) == 1 else dirs


def build_value_data_config(
    local_data_dir: str | Sequence[str],
    config: ValueModelConfig,
    *,
    tokenizer_path: str | None,
    default_prompt: str | None = None,
    filter_if_success: tuple[str, ...] | None = None,
) -> _config.DataConfig:
    from openpi.training.agilex_hdf5_value_dataset import looks_like_agilex_hdf5_dir

    dirs = (local_data_dir,) if isinstance(local_data_dir, str) else tuple(local_data_dir)
    if not dirs:
        raise ValueError("data_dir 为空")
    hdf5_flags = [looks_like_agilex_hdf5_dir(path) for path in dirs]
    if any(hdf5_flags) and not all(hdf5_flags):
        raise ValueError("data_dir 里同时包含 AgileX HDF5 和 LeRobot 数据集，不能放在同一次训练里")
    is_hdf5 = all(hdf5_flags)
    return _config.DataConfig(
        local_data_dir=normalize_data_dirs(dirs),
        prompt_from_task=not is_hdf5,
        agilex_default_prompt=default_prompt,
        agilex_filter_if_success=filter_if_success,
        data_transforms=_transforms.Group(
            inputs=[RemapValueLabelKey(), value_policy.ValueInputs()],
        ),
        model_transforms=_transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    GemmaValueTokenizer(
                        config.max_token_len,
                        tokenizer_path=tokenizer_path,
                    )
                ),
                _transforms.PadStatesAndActions(config.action_dim),
            ]
        ),
    )


def init_logging():
    """Custom logging format."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)


def _wandb_sanitize_config(config: dict) -> dict:
    sanitized = {}
    for key, value in config.items():
        if isinstance(value, Path):
            sanitized[key] = str(value)
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            sanitized[key] = value
    return sanitized


@dataclasses.dataclass
class TrainState:
    step: int
    params: nnx.State
    model_def: nnx.GraphDef
    opt_state: optax.OptState
    ema_params: nnx.State | None = None  


FREEZE_MODES = ("none", "siglip_only", "all_backbones")


def _is_siglip_path(path, _value) -> bool:
    joined_path = "/".join(str(part) for part in path)
    return joined_path == "img" or joined_path.startswith("img/")


def _is_llm_path(path, _value) -> bool:
    joined_path = "/".join(str(part) for part in path)
    return joined_path == "llm" or joined_path.startswith("llm/")


def _make_trainable_filter(freeze_mode: str):
    if freeze_mode == "none":
        return nnx.Param
    if freeze_mode == "siglip_only":
        return nnx.All(nnx.Param, nnx.Not(_is_siglip_path))
    if freeze_mode == "all_backbones":
        return nnx.All(nnx.Param, nnx.Not(_is_siglip_path), nnx.Not(_is_llm_path))
    raise ValueError(f"Unsupported freeze_mode: {freeze_mode}")


def _make_group_filter(group: str):
    if group == "siglip":
        return nnx.All(nnx.Param, _is_siglip_path)
    if group == "llm":
        return nnx.All(nnx.Param, _is_llm_path)
    if group == "head":
        return nnx.All(nnx.Param, nnx.Not(_is_siglip_path), nnx.Not(_is_llm_path))
    raise ValueError(f"Unsupported param group: {group}")


def _kernel_param_norm(params: nnx.State, param_filter) -> jnp.ndarray:
    kernel_filter = nnx.All(
        param_filter,
        nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
        lambda _, x: getattr(x.value, "ndim", 0) > 1,
    )
    filtered = params.filter(kernel_filter)
    if not filtered.flat_state():
        return jnp.asarray(0.0, dtype=jnp.float32)
    return optax.global_norm(filtered)


def create_train_state(
    config: ValueModelConfig,
    num_train_steps: int,
    rng: at.KeyArrayLike,
    load_pretrained: bool = False,
    ema_decay: float = 0.99,
    peak_lr: float = 5e-6,
    decay_lr: float = 5e-7,
    warmup_steps: int | None = None,
    grad_clip_norm: float = 1.0,
    freeze_mode: str = "none",
) -> tuple[TrainState, optax.GradientTransformation, optax.Schedule]:
    """创建训练状态。"""
    model = config.create(rng)
    params = nnx.state(model)

    if load_pretrained:
        logging.info(console.info("加载 SigLIP + Gemma3-270M 预训练权重..."))
        loader = ValueModelWeightLoader()
        params_dict = params.to_pure_dict()
        loaded_params = loader.load(params_dict)
        params.replace_by_pure_dict(loaded_params)
        logging.info(console.ok("预训练权重加载完成"))

    model_def = nnx.graphdef(model)

    if warmup_steps is None:
        warmup_steps = min(1000, num_train_steps // 10)
    decay_steps = max(num_train_steps - warmup_steps, 1)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=peak_lr / (warmup_steps + 1),
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=decay_lr,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(
            lr_schedule, 
            b1=0.9,        # π0优化：更保守的momentum
            b2=0.95,       # π0优化：更保守的二阶momentum
            eps=1e-8,      # π0优化：数值稳定性
            weight_decay=1e-10  # π0优化：极小权重衰减，避免OOM
        )
    )
    trainable_filter = _make_trainable_filter(freeze_mode)
    trainable_params = params.filter(trainable_filter)
    if not trainable_params.flat_state():
        raise ValueError(f"freeze_mode={freeze_mode} 没有任何可训练参数")
    opt_state = tx.init(trainable_params)

    return TrainState(
        step=0,
        params=params,
        model_def=model_def,
        opt_state=opt_state,
        ema_params=jax.tree.map(lambda x: x, params) if ema_decay is not None else None,  # π0优化：初始化EMA
    ), tx, lr_schedule


def train_step(
    state: TrainState,
    tx: optax.GradientTransformation,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    target: jnp.ndarray,
    ema_decay: float = 0.99,
    freeze_mode: str = "none",
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    """单步训练。"""
    model = nnx.merge(state.model_def, state.params)
    model.train()
    train_rng = jax.random.fold_in(rng, state.step)
    trainable_filter = _make_trainable_filter(freeze_mode)

    def loss_fn(model):
        return model.compute_loss(train_rng, observation, target, train=True)

    diff_state = nnx.DiffState(0, trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model)

    params = state.params.filter(trainable_filter)
    updates, new_opt_state = tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    full_params = nnx.state(model)

    # π0优化：EMA更新
    new_ema_params = None
    if state.ema_params is not None:
        new_ema_params = jax.tree.map(
            lambda old, new: ema_decay * old + (1 - ema_decay) * new, 
            state.ema_params, 
            full_params
        )

    new_state = TrainState(
        step=state.step + 1,
        params=full_params,
        model_def=state.model_def,
        opt_state=new_opt_state,
        ema_params=new_ema_params,
    )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "siglip_param_norm": _kernel_param_norm(full_params, _make_group_filter("siglip")),
        "trainable_param_norm": _kernel_param_norm(full_params, trainable_filter),
        "llm_param_norm": _kernel_param_norm(full_params, _make_group_filter("llm")),
        "head_param_norm": _kernel_param_norm(full_params, _make_group_filter("head")),
    }

    return new_state, info


def eval_step(
    params: nnx.State,
    model_def: nnx.GraphDef,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    target: jnp.ndarray,
) -> jnp.ndarray:
    model = nnx.merge(model_def, params)
    model.eval()
    return model.compute_loss(rng, observation, target, train=False)


def load_checkpoint(checkpoint_path: Path, state: TrainState) -> TrainState:
    """从 checkpoint 恢复训练状态。"""
    import orbax.checkpoint as ocp
    
    # 确保使用绝对路径
    checkpoint_path = checkpoint_path.resolve()
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint 不存在: {checkpoint_path}")
    
    with ocp.PyTreeCheckpointer() as ckptr:
        restored = ckptr.restore(str(checkpoint_path))
    
    # 恢复参数
    state.params.replace_by_pure_dict(restored["params"])
    
    # 恢复 EMA 参数（如果存在）
    if "ema_params" in restored and state.ema_params is not None:
        state.ema_params.replace_by_pure_dict(restored["ema_params"])
    
    # 恢复步数
    restored_step = int(restored["step"])
    state = TrainState(
        step=restored_step,
        params=state.params,
        model_def=state.model_def,
        opt_state=state.opt_state,
        ema_params=state.ema_params,
    )
    
    logging.info(console.ok(f"从 checkpoint 恢复: {checkpoint_path}, step={restored_step}"))
    return state


def save_checkpoint(state: TrainState, checkpoint_dir: Path, step: int):
    """保存 checkpoint。"""
    import orbax.checkpoint as ocp

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 确保使用绝对路径
    ckpt_path = (checkpoint_dir / f"step_{step:08d}").resolve()
    if ckpt_path.exists():
        logging.warning(console.warn(f"Checkpoint already exists, skip save: {ckpt_path}"))
        return

    with ocp.PyTreeCheckpointer() as ckptr:
        save_dict = {
            "params": state.params.to_pure_dict(), 
            "step": step
        }
        # π0优化：同时保存EMA参数
        if state.ema_params is not None:
            save_dict["ema_params"] = state.ema_params.to_pure_dict()
        
        ckptr.save(str(ckpt_path), save_dict)

    logging.info(console.ok(f"保存 checkpoint: {ckpt_path}"))


def main():
    parser = argparse.ArgumentParser(description="训练 Value Function", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="+",
        required=True,
        help="一个或多个数据集路径。LeRobot 只能传一个；AgileX HDF5 可传多个根目录，每个目录单独扫描 episode.hdf5",
    )
    parser.add_argument(
        "--default_prompt",
        type=str,
        default=None,
        help="AgileX HDF5 两行条码 prompt。扫码帧及之前用第一行，之后按条码填 item/color/basket。只传一行时改用 pi0.5 默认两行模板",
    )
    parser.add_argument(
        "--filter_if_success",
        nargs="+",
        default=None,
        metavar="STATUS",
        help="只使用这些 if_success 的 AgileX episode，例如: --filter_if_success take_over fail",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        nargs="+",
        default=None,
        help="可选的验证集路径，AgileX HDF5 可传多个根目录",
    )
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="未指定 val_data_dir 时按完整 episode 留出的验证比例，0 表示关闭自动划分")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="ValueHead 训练 dropout 概率，验证时关闭")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint 保存路径")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_train_steps", type=int, default=30000, help="训练步数")
    parser.add_argument("--log_interval", type=int, default=100, help="日志间隔")
    parser.add_argument("--val_interval", type=int, default=1000, help="验证间隔步数，<=0 表示禁用验证")
    parser.add_argument("--save_interval", type=int, default=1000, help="保存间隔")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker 数量（参考 openpi 默认值）")
    parser.add_argument("--wandb_project", type=str, default="openpi", help="Weights & Biases project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"], help="Weights & Biases mode")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gemma_variant", type=str, default="gemma3_270m", help="Gemma 变体")
    parser.add_argument("--siglip_variant", type=str, default="So400m/14", help="SigLIP 变体")
    parser.add_argument("--fsdp_devices", type=int, default=1, help="FSDP设备数量，>1启用模型并行")
    parser.add_argument("--load_pretrained", action="store_true", help="加载 PaliGemma 预训练权重")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从指定checkpoint恢复训练（例如：step_00001000）")
    parser.add_argument("--pyarrow_num_threads", type=int, default=0, help="PyArrow 读取并行线程数，0表示不设置")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="可选的 Gemma3 tokenizer.model 本地路径")
    parser.add_argument("--peak_lr", type=float, default=5e-6, help="学习率峰值")
    parser.add_argument("--decay_lr", type=float, default=5e-7, help="余弦衰减结束学习率")
    parser.add_argument("--warmup_steps", type=int, default=None, help="warmup 步数，默认取 min(1000, num_train_steps//10)")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="全局梯度裁剪阈值")
    parser.add_argument(
        "--freeze_mode",
        type=str,
        choices=FREEZE_MODES,
        default="none",
        help="冻结策略: none=全量训练, siglip_only=只冻结SigLIP, all_backbones=冻结SigLIP和Gemma backbone",
    )
    args = parser.parse_args()
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error("val_fraction must be in [0, 1)")
    if not 0.0 <= args.dropout_rate < 1.0:
        parser.error("dropout_rate must be in [0, 1)")

    init_logging()
    logging.info(f"\033[1;34;46mRunning on:\033[0m \033[1;33;40m{platform.node()}\033[0m")

    # 配置JAX
    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")  # 优先使用CUDA，fallback到CPU

    if args.pyarrow_num_threads and args.pyarrow_num_threads > 0:
        os.environ.setdefault("PYARROW_NUM_THREADS", str(args.pyarrow_num_threads))
        logging.info(console.info(f"PYARROW_NUM_THREADS={args.pyarrow_num_threads}"))

    # 设置JAX配置
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    try:
        logging.info(f"JAX devices: {jax.devices()}")
        logging.info(f"JAX default backend: {jax.default_backend()}")
    except Exception as e:
        logging.warning(console.warn(f"JAX设备检测失败: {e}"))
        logging.info(console.warn("继续使用CPU进行训练"))

    rng = jax.random.key(args.seed)
    train_rng, init_rng = jax.random.split(rng)

    # π0风格多GPU配置 - 自动使用所有可用GPU
    available_devices = jax.device_count()
    # 如果用户没有指定fsdp_devices，自动使用所有GPU
    if args.fsdp_devices == 1 and available_devices > 1:
        args.fsdp_devices = available_devices
        logging.info(console.info(f"自动调整：使用所有 {available_devices} 个GPU进行FSDP"))
    
    logging.info(console.info(f"可用设备数: {available_devices}, FSDP设备数: {args.fsdp_devices}"))
    if available_devices % args.fsdp_devices != 0:
        raise ValueError(f"设备数 {available_devices} 必须能被FSDP设备数 {args.fsdp_devices} 整除")
    
    mesh = sharding.make_mesh(num_fsdp_devices=args.fsdp_devices)
    logging.info(console.info(f"Mesh形状: {mesh.shape}, 轴: {mesh.axis_names}"))
    
    # 优化后的sharding配置
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    config = ValueModelConfig(
        gemma_variant=args.gemma_variant,
        siglip_variant=args.siglip_variant,
        dropout_rate=args.dropout_rate,
    )

    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb is not installed. Install it or set --wandb_mode disabled") from exc

        wandb_config = _wandb_sanitize_config({**vars(args), **dataclasses.asdict(config)})
        tags = [t for t in args.wandb_tags.split(",") if t] if args.wandb_tags else None
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=wandb_config,
            tags=tags,
        )
        wandb.define_metric("step")
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("val/*", step_metric="step")

    logging.info("\033[1;36m初始化数据加载器...\033[0m")
    effective_batch_size = args.batch_size
    logging.info(f"全局batch size: {effective_batch_size}")
    logging.info(console.info(f"freeze_mode={args.freeze_mode}"))
    logging.info(console.info(f"训练标签列: {VALUE_LABEL_COLUMN} (兼容旧列 {LEGACY_VALUE_LABEL_COLUMN})"))

    # 参考 openpi：默认使用多进程 DataLoader
    max_workers = max(0, args.num_workers)
    env_workers = os.getenv("OPENPI_VALUE_NUM_WORKERS")
    if env_workers is not None:
        try:
            requested_workers = int(env_workers)
            if requested_workers < 0:
                raise ValueError
            cpu_count = os.cpu_count() or 1
            max_workers = min(requested_workers, cpu_count)
            if requested_workers > cpu_count:
                logging.info(console.info(f"OPENPI_VALUE_NUM_WORKERS={requested_workers} 超过CPU核数，已裁剪为 {max_workers}"))
        except ValueError:
            logging.warning(console.warn(f"OPENPI_VALUE_NUM_WORKERS='{env_workers}' 无效，保持 num_workers=0"))

    resolved_tokenizer_path = _resolve_local_gemma_tokenizer_path(args.tokenizer_path)
    if resolved_tokenizer_path is not None:
        logging.info(console.info(f"使用本地 Gemma3 tokenizer: {resolved_tokenizer_path}"))
    elif max_workers > 0:
        logging.warning(
            console.warn(
                "未找到本地 Gemma3 tokenizer；多进程 DataLoader 中会在 worker 内触发默认远程加载，"
                "容易报错。已自动将 num_workers 降为 0，请尽量显式传入 --tokenizer_path。"
            )
        )
        max_workers = 0

    _validate_gemma_tokenizer(resolved_tokenizer_path)

    filter_if_success = tuple(args.filter_if_success) if args.filter_if_success else None
    data_config = build_value_data_config(
        args.data_dir,
        config,
        tokenizer_path=resolved_tokenizer_path,
        default_prompt=args.default_prompt,
        filter_if_success=filter_if_success,
    )

    from openpi.training.agilex_hdf5_value_dataset import (
        AgilexHdf5ValueDataset,
        looks_like_agilex_hdf5_dirs,
    )

    raw_train = None
    raw_val = None
    val_data_config = data_config
    if looks_like_agilex_hdf5_dirs(args.data_dir):
        raw_train = AgilexHdf5ValueDataset(
            args.data_dir, default_prompt=args.default_prompt, filter_if_success=filter_if_success,
        )
    if args.val_interval > 0:
        if args.val_data_dir:
            val_data_config = build_value_data_config(
                args.val_data_dir, config, tokenizer_path=resolved_tokenizer_path,
                default_prompt=args.default_prompt, filter_if_success=filter_if_success,
            )
            if looks_like_agilex_hdf5_dirs(args.val_data_dir):
                raw_val = AgilexHdf5ValueDataset(
                    args.val_data_dir, default_prompt=args.default_prompt, filter_if_success=filter_if_success,
                )
            else:
                raw_val = _data_loader.create_torch_dataset(val_data_config, config.action_horizon, config)
        elif args.val_fraction > 0:
            if raw_train is None:
                raise ValueError("Automatic episode validation split requires AgileX HDF5; "
                                 "provide --val_data_dir or explicitly set --val_fraction 0")
            previous_split_path = Path(args.checkpoint_dir) / "validation_split.json"
            validation_paths = None
            if previous_split_path.exists():
                previous = json.loads(previous_split_path.read_text())
                known_paths = set(previous["train_episodes"]) | set(previous["val_episodes"])
                current_paths = {str(ep["hdf5_path"].resolve()) for ep in raw_train.episodes}
                if (previous.get("seed") == args.seed and previous.get("val_fraction") == args.val_fraction
                        and current_paths <= known_paths):
                    validation_paths = previous["val_episodes"]
            raw_train, raw_val = raw_train.split_by_episode(args.val_fraction, args.seed, validation_paths)
    else:
        logging.info(console.warn("val_interval<=0: validation disabled"))

    if isinstance(raw_train, AgilexHdf5ValueDataset) and isinstance(raw_val, AgilexHdf5ValueDataset):
        train_paths = [str(ep["hdf5_path"].resolve()) for ep in raw_train.episodes]
        val_paths = [str(ep["hdf5_path"].resolve()) for ep in raw_val.episodes]
        overlap = set(train_paths) & set(val_paths)
        if overlap:
            raise ValueError(f"Training/validation episode overlap: {len(overlap)} (sample: {sorted(overlap)[:3]})")
        manifest = {"seed": args.seed, "val_fraction": args.val_fraction,
                    "train_episodes": train_paths, "val_episodes": val_paths}
        manifest_path = Path(args.checkpoint_dir) / "validation_split.json"
        excluded = {item["path"]: item for dataset in (raw_train, raw_val)
                    for item in dataset.excluded_prompt_episodes}
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous != manifest:
                removed = ((set(previous["train_episodes"]) - set(train_paths))
                           | (set(previous["val_episodes"]) - set(val_paths)))
                safe_exclusion = (
                    previous.get("seed") == args.seed and previous.get("val_fraction") == args.val_fraction
                    and set(train_paths) <= set(previous["train_episodes"])
                    and set(val_paths) <= set(previous["val_episodes"])
                    and removed <= set(excluded)
                )
                if not safe_exclusion:
                    raise ValueError("Existing validation split differs; use a separate checkpoint_dir for this experiment")
                backup_manifest = manifest_path.with_name("validation_split.before-barcode-repair-20261011.json")
                if not backup_manifest.exists():
                    backup_manifest.write_text(manifest_path.read_text())
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if excluded:
            exclusion_path = manifest_path.with_name("excluded_prompt_episodes.json")
            exclusion_path.write_text(json.dumps(list(excluded.values()), indent=2) + "\n")
            logging.warning(console.warn(f"Excluded {len(excluded)} episodes with invalid prompt metadata; report: {exclusion_path}"))
        logging.info(console.info(f"Episode split manifest: {manifest_path}"))

    data_loader = _data_loader.create_value_data_loader(
        data_config, model_config=config, batch_size=effective_batch_size,
        sharding=data_sharding, shuffle=True, num_workers=max_workers,
        seed=args.seed, skip_norm_stats=True, framework="jax", raw_dataset=raw_train,
    )
    if len(data_loader.dataset) < effective_batch_size:
        raise ValueError("Training split has fewer frames than batch_size")
    logging.info(console.info(f"训练集大小: {len(data_loader.dataset)} 帧; dropout_rate={args.dropout_rate}"))
    if wandb_run is not None:
        wandb_run.config.update({"dataset_size": len(data_loader.dataset),
                                "effective_batch_size": effective_batch_size}, allow_val_change=True)

    val_loader = None
    val_num_batches = 0
    if raw_val is not None:
        val_dataset_size = len(raw_val)
        if val_dataset_size < effective_batch_size:
            raise ValueError(f"验证集太小: {val_dataset_size} 帧，小于 batch_size={effective_batch_size}")
        val_num_batches = val_dataset_size // effective_batch_size
        val_remainder = val_dataset_size % effective_batch_size
        val_loader = _data_loader.create_value_data_loader(
            val_data_config, model_config=config, batch_size=effective_batch_size,
            sharding=data_sharding, shuffle=False, num_batches=val_num_batches,
            num_workers=max_workers, seed=args.seed, skip_norm_stats=True,
            framework="jax", raw_dataset=raw_val,
        )
        logging.info(console.info(
            f"验证集大小: {val_dataset_size} 帧, 验证batch数: {val_num_batches}, "
            f"drop_last丢弃: {val_remainder} 帧, val_interval={args.val_interval}"
        ))
        if wandb_run is not None:
            wandb_run.config.update({"val_dataset_size": val_dataset_size,
                                    "val_num_batches": val_num_batches}, allow_val_change=True)

    logging.info("\033[1;36m初始化模型...\033[0m")
    train_state, tx, lr_schedule = create_train_state(
        config,
        args.num_train_steps,
        init_rng,
        args.load_pretrained,
        peak_lr=args.peak_lr,
        decay_lr=args.decay_lr,
        warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        freeze_mode=args.freeze_mode,
    )
    logging.info("\033[1;32m模型初始化完成\033[0m")
    
    # 从检查点恢复（如果指定）
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.checkpoint_dir) / args.resume_from_checkpoint
        train_state = load_checkpoint(checkpoint_path, train_state)
        logging.info(console.info(f"从 step {train_state.step} 继续训练"))

    # 将模型参数分片到多GPU
    if args.fsdp_devices > 1:
        logging.info("\033[1;36m应用FSDP分片到模型参数...\033[0m")
        with sharding.set_mesh(mesh):
            train_state = jax.tree.map(
                lambda x: sharding.apply_fsdp_sharding(mesh, x) if hasattr(x, 'shape') else x,
                train_state,
                is_leaf=lambda x: hasattr(x, 'shape')
            )
        logging.info("\033[1;32mFSDP分片完成\033[0m")
    else:
        # 单卡：复制到所有设备
        train_state = jax.tree.map(
            lambda x: jax.device_put(x, replicated_sharding), 
            train_state,
            is_leaf=lambda x: hasattr(x, 'shape')
        )

    @functools.partial(
        jax.jit,
        in_shardings=(
            replicated_sharding,  # params
            replicated_sharding,  # model_def
            replicated_sharding,  # opt_state
            replicated_sharding,  # ema_params
            replicated_sharding,  # step
            replicated_sharding,  # rng
            data_sharding,        # observation
            data_sharding,        # target
        ),
        out_shardings=(
            replicated_sharding,  # new_params
            replicated_sharding,  # new_model_def
            replicated_sharding,  # new_opt_state
            replicated_sharding,  # new_ema_params
            replicated_sharding,  # new_step
            replicated_sharding,  # info
        ),
        # 移除donate_argnums，避免缓冲区重复使用错误
    )
    def jit_train_step(params, model_def, opt_state, ema_params, step, rng, observation, target):
        state = TrainState(
            step=step, 
            params=params, 
            model_def=model_def, 
            opt_state=opt_state,
            ema_params=ema_params
        )
        new_state, info = train_step(state, tx, rng, observation, target, freeze_mode=args.freeze_mode)
        return new_state.params, new_state.model_def, new_state.opt_state, new_state.ema_params, new_state.step, info

    @functools.partial(
        jax.jit,
        in_shardings=(
            replicated_sharding,  # params
            replicated_sharding,  # model_def
            replicated_sharding,  # rng
            data_sharding,        # observation
            data_sharding,        # target
        ),
        out_shardings=replicated_sharding,
    )
    def jit_eval_step(params, model_def, rng, observation, target):
        return eval_step(params, model_def, rng, observation, target)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()  # 确保绝对路径
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 从恢复的步数开始训练
    start_step = train_state.step
    total_steps = args.num_train_steps
    pbar = tqdm.tqdm(
        range(start_step, total_steps),
        initial=start_step,
        total=total_steps,
        dynamic_ncols=True,
        desc="训练进度",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    progress.sync_pbar_color(pbar)
    infos = []

    # π0风格：数据预取和GPU利用率优化
    data_iter = iter(data_loader)
    
    # 预取多个batch，减少GPU等待
    logging.info("\033[1;36m预取前几个batch以优化GPU利用率...\033[0m")
    prefetch_batches = []
    for i in range(min(3, len(data_loader))):  # 预取3个batch
        try:
            batch = next(data_iter)
            prefetch_batches.append(batch)
        except StopIteration:
            break

    if not prefetch_batches:
        logging.error("\033[1;31m无法预取任何batch，检查数据集\033[0m")
        return

    logging.info(f"\033[1;32m成功预取 {len(prefetch_batches)} 个batch\033[0m")

    # JIT预热：避免首次编译的GPU空闲
    logging.info("\033[1;36mJIT编译预热...\033[0m")
    observation, value = prefetch_batches[0]
    with sharding.set_mesh(mesh):
        _ = jit_train_step(
            train_state.params,
            train_state.model_def,
            train_state.opt_state,
            train_state.ema_params,
            train_state.step,
            train_rng,
            observation,
            value,
        )
    logging.info("\033[1;32mJIT编译完成，开始训练...\033[0m")
    
    # 重新初始化数据迭代器
    data_iter = iter(data_loader)
    prefetch_idx = 0

    def run_validation(step: int) -> None:
        if val_loader is None:
            return

        params_for_eval = train_state.ema_params if train_state.ema_params is not None else train_state.params
        val_losses: list[float] = []
        val_iter = iter(val_loader)

        logging.info(console.info(f"开始验证: step={step}, batches={val_num_batches}"))
        for val_batch_idx in range(val_num_batches):
            val_observation, val_value = next(val_iter)
            eval_rng = jax.random.fold_in(train_rng, step * 10_000 + val_batch_idx)
            with sharding.set_mesh(mesh):
                val_loss = jit_eval_step(
                    params_for_eval,
                    train_state.model_def,
                    eval_rng,
                    val_observation,
                    val_value,
                )
            val_losses.append(float(jax.device_get(val_loss)))

        mean_val_loss = float(np.mean(val_losses))
        pbar.write(f"Step {step}: val_cross_entropy={mean_val_loss:.4f}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "step": int(step),
                    "val/cross_entropy": mean_val_loss,
                }
            )

    for step in pbar:
        progress.sync_pbar_color(pbar)
        # 使用预取的batch或获取新batch
        if prefetch_idx < len(prefetch_batches):
            observation, value = prefetch_batches[prefetch_idx]
            prefetch_idx += 1
        else:
            try:
                observation, value = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                observation, value = next(data_iter)
        with sharding.set_mesh(mesh):
            new_params, new_model_def, new_opt_state, new_ema_params, new_step, info = jit_train_step(
                train_state.params,
                train_state.model_def,
                train_state.opt_state,
                train_state.ema_params,
                train_state.step,
                train_rng,
                observation,
                value,
            )
            train_state = TrainState(
                step=new_step,
                params=new_params,
                model_def=new_model_def,
                opt_state=new_opt_state,
                ema_params=new_ema_params,
            )

        infos.append(info)

        if step % args.log_interval == 0 and step > 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            
            # 获取当前学习率
            current_lr = float(lr_schedule(step)) if lr_schedule is not None else None
            
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}, lr={current_lr:.2e}" if isinstance(current_lr, (int, float)) else f"Step {step}: {info_str}")
            if wandb_run is not None:
                log_payload = {"step": int(step), "train/freeze_mode": args.freeze_mode}
                for key, value in reduced_info.items():
                    log_payload[f"train/{key}"] = float(value)
                if current_lr is not None:
                    log_payload["train/lr"] = float(current_lr)
                wandb_run.log(log_payload)
            infos = []

        if val_loader is not None and args.val_interval > 0 and step % args.val_interval == 0 and step > 0:
            run_validation(step)

        if step % args.save_interval == 0 and step > 0:
            # 计算当前损失
            current_loss = jax.device_get(info["loss"]).item()
            logging.info(f"\033[1;44m保存检查点 - Step {step}, Loss: {current_loss:.4f}\033[0m")
            save_checkpoint(train_state, checkpoint_dir, step)

    save_checkpoint(train_state, checkpoint_dir, args.num_train_steps)
    if wandb_run is not None:
        wandb_run.finish()
    logging.info("\033[1;32m训练完成!\033[0m")


if __name__ == "__main__":
    main()
