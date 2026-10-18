"""Configuration for ValueModel."""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from typing import Literal
from openpi.shared import array_typing as at

# 添加 Gemma 配置导入
import sys
import os
gemma_path = os.path.join(os.path.dirname(__file__), '../../../gemma')
sys.path.insert(0, gemma_path)
from gemma.gm.nn._gemma import Gemma3_270M

if TYPE_CHECKING:
    from openpi.models.value_model import ValueModel


@dataclasses.dataclass(frozen=True)
class ValueModelConfig(_model.BaseModelConfig):
    """Configuration for ValueModel combining SigLIP and Gemma3 270M."""

    # 基础配置（都有默认值）
    dtype: str = "bfloat16"
    gemma_variant: Literal["gemma3_270m"] = "gemma3_270m"
    siglip_variant: str = "So400m/14"
    
    # Value function 特定配置
    action_dim: int = 32
    action_horizon: int = 1
    
    # 在 __post_init__ 中设置的字段，需要有默认值
    embed_dim: int = 640  # 将在 __post_init__ 中覆盖
    vocab_size: int = 262144  # 将在 __post_init__ 中覆盖  
    max_token_len: int = 256  # 将在 __post_init__ 中覆盖

    def __post_init__(self):
        """在初始化后设置 Gemma 相关配置"""
        # 使用 Gemma3_270M 的完整配置（借鉴 Pi0 方式但使用 Gemma3）
        gemma_config = Gemma3_270M().config
        
        object.__setattr__(self, '_gemma_config', gemma_config)
        object.__setattr__(self, 'embed_dim', gemma_config.embed_dim)  # 640
        object.__setattr__(self, 'vocab_size', gemma_config.num_embed)  # 262144 - Gemma3 完整词汇表
        object.__setattr__(self, 'max_token_len', 48)  # 合理的默认值

    @property
    def gemma_config(self):
        """获取 Gemma 配置"""
        return getattr(self, '_gemma_config', Gemma3_270M().config)

    @property
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.VALUE  # 改为 VALUE 类型而不是 PI0

    def create(self, rng: at.KeyArrayLike) -> "ValueModel":
        from openpi.models.value_model import ValueModel

        return ValueModel(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, jax.Array]:
        """为 Value Model 创建输入规格"""
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        
        # Value model 输出价值而不是动作
        value_spec = jax.ShapeDtypeStruct([batch_size], jnp.float32)

        return observation_spec, value_spec
