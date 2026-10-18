"""Value function model combining SigLIP and Gemma for VLM-based value estimation."""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import model as _model
import sys
import os
# 添加 gemma 目录到 Python 路径
gemma_path = os.path.join(os.path.dirname(__file__), '../../../gemma')
sys.path.insert(0, gemma_path)
from gemma.gm.nn._gemma import Gemma3_270M
from gemma.gm.nn._modules import Embedder
from gemma.gm.nn import _transformer
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

NUM_ATOMS = 201
V_MIN = -1.0
V_MAX = 0.0
DELTA_Z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)


def get_supports():
    """Get the support values for C51 distributional RL."""
    return jnp.linspace(V_MIN, V_MAX, NUM_ATOMS)


def make_attn_mask(input_mask, mask_ar):
    """Creates attention mask from input mask and autoregressive mask."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def target_to_twohot(target_values: jnp.ndarray) -> jnp.ndarray:
    """Convert target values to two-hot distribution using linear interpolation.

    Args:
        target_values: Target values in range [V_MIN, V_MAX], shape [batch].

    Returns:
        Two-hot target distribution, shape [batch, NUM_ATOMS].
    """
    target_values = jnp.clip(target_values, V_MIN, V_MAX)

    b_left = jnp.floor((target_values - V_MIN) / DELTA_Z).astype(jnp.int32)
    b_left = jnp.clip(b_left, 0, NUM_ATOMS - 2)
    b_right = b_left + 1

    z_left = V_MIN + b_left.astype(jnp.float32) * DELTA_Z

    weight_right = (target_values - z_left) / DELTA_Z
    weight_left = 1.0 - weight_right

    batch_size = target_values.shape[0]
    target_dist = jnp.zeros((batch_size, NUM_ATOMS))

    batch_idx = jnp.arange(batch_size)
    target_dist = target_dist.at[batch_idx, b_left].set(weight_left)
    return target_dist.at[batch_idx, b_right].set(weight_right)


def normalize_value_target_shape(target: jnp.ndarray) -> jnp.ndarray:
    """Accept scalar targets shaped [batch] or [batch, 1], or full distributions [batch, num_atoms]."""
    target = jnp.asarray(target, dtype=jnp.float32)
    if target.ndim == 0:
        return target[None]
    if target.ndim == 1:
        return target
    if target.ndim == 2 and target.shape[-1] == 1:
        return jnp.squeeze(target, axis=-1)
    if target.ndim == 2 and target.shape[-1] == NUM_ATOMS:
        return target
    raise ValueError(
        f"Unsupported target shape {target.shape}; expected [batch], [batch, 1], or [batch, {NUM_ATOMS}]"
    )



def dist_to_value(logits: jnp.ndarray) -> jnp.ndarray:
    """Convert distribution logits to expected value.

    Args:
        logits: Distribution logits, shape [batch, NUM_ATOMS].

    Returns:
        Expected value, shape [batch].
    """
    supports = get_supports()
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    return jnp.sum(probs * supports, axis=-1)


class ValueHead(nnx.Module):
    """Two-layer MLP value head with LayerNorm and GELU."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float, rngs: nnx.Rngs):
        self.layer_norm = nnx.LayerNorm(input_dim, rngs=rngs)
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = nnx.gelu(x)
        # 简化：训练时不使用 dropout，避免状态管理复杂性
        # if train and self.dropout_rate > 0:
        #     x = nnx.dropout(x, rate=self.dropout_rate, deterministic=False)
        return self.linear2(x)


class GemmaBlockRunner(Gemma3_270M):
    """Extends Gemma3_270M to allow running the backbone with continuous embeddings."""
    
    def __call__(self, 
                 embeddings: jax.Array | None = None, 
                 mask: jax.Array | None = None,
                 tokens: jax.Array | None = None):
        """
        Args:
            embeddings: [Batch, SeqLen, Dim] - For running Backbone
            mask: [Batch, SeqLen] - For running Backbone
            tokens: [Batch, SeqLen] - For initializing Embedder
        """
        # 1. Force Embedder initialization if tokens are provided
        if tokens is not None:
            self.embedder.encode(tokens)

        # 2. Run Backbone if embeddings are provided
        if embeddings is not None:
            batch_size, seq_len, _ = embeddings.shape
            
            # Create positions [Batch, SeqLen]
            positions = jnp.arange(seq_len)[None, :] + jnp.zeros((batch_size, 1), dtype=jnp.int32)
            
            # Create attention mask [Batch, SeqLen, SeqLen]
            if mask is None:
                mask = jnp.ones((batch_size, seq_len), dtype=bool)
                
            attn_mask = mask[:, None, :] * mask[:, :, None]
            
            inputs = _transformer._Inputs(
                embeddings=embeddings,
                positions=positions,
                attention_mask=attn_mask,
                inputs_mask=mask
            )
            
            # Run the transformer blocks
            # cache is None for training/simple inference
            x, _ = self._apply_attention(inputs, cache=None)
            return x
        
        return None


class ValueModel(nnx.Module):
    """Value function model using SigLIP (400M) + Gemma (270M) with C51-style distributional output."""

    def __init__(self, config: "ValueModelConfig", rngs: nnx.Rngs):
        self.config = config
        # 直接使用配置中的 Gemma 配置
        gemma_config = config.gemma_config

        self.img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=None,  # 直接使用 SigLIP backbone token，避免依赖未加载的分类头
                variant=config.siglip_variant,
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )

        # 添加投影层将SigLIP的1152维投影到Gemma3的640维
        self.img_projection = nnx.Linear(1152, gemma_config.embed_dim, rngs=rngs)

        # 使用自定义的 GemmaBlockRunner 替代原始 Gemma3_270M
        self.llm = nnx_bridge.ToNNX(
            GemmaBlockRunner()
        )

        fake_image = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
        self.img.lazy_init(fake_image, train=False, rngs=rngs)
        
        # 初始化 LLM - 需要同时提供 tokens (初始化 Embedder) 和 embeddings (初始化 Backbone)
        fake_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
        fake_embeds = jnp.zeros((1, 1, gemma_config.embed_dim), dtype=jnp.float32)
        fake_mask = jnp.ones((1, 1), dtype=bool)
        
        self.llm.lazy_init(embeddings=fake_embeds, mask=fake_mask, tokens=fake_tokens, rngs=rngs)

        # 添加交叉注意力层：让文本特征关注图像特征
        self.cross_attention = nnx.MultiHeadAttention(
            num_heads=8,                        # 8个注意力头
            in_features=gemma_config.embed_dim, # 640维
            qkv_features=gemma_config.embed_dim,
            out_features=gemma_config.embed_dim,
            decode=False,
            rngs=rngs,
        )
        self.cross_attn_norm = nnx.LayerNorm(gemma_config.embed_dim, rngs=rngs)

        hidden_dim = gemma_config.embed_dim // 2
        self.value_head = ValueHead(
            input_dim=gemma_config.embed_dim,
            hidden_dim=hidden_dim,
            output_dim=NUM_ATOMS,
            dropout_rate=0.1,
            rngs=rngs,
        )

        self.deterministic = True

    @at.typecheck
    def embed_tokens(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed images and text into a single token sequence with cross-attention and deep fusion."""
        input_mask = []
        ar_mask = []
        image_tokens_list = []
        image_mask_list = []

        # 处理图像特征
        for name in obs.images:
            image_tokens, _ = self.img(obs.images[name], train=False)
            image_tokens = self.img_projection(image_tokens)
            image_tokens_list.append(image_tokens)
            image_mask_list.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )

        # 拼接所有图像特征
        all_image_tokens = jnp.concatenate(image_tokens_list, axis=1) if image_tokens_list else None
        all_image_masks = jnp.concatenate(image_mask_list, axis=1) if image_mask_list else None

        # 准备 Text Embedder 参数
        _, llm_state = nnx.split(self.llm)
        llm_params = llm_state.to_pure_dict()
        
        embedder_params = None
        if 'embedder' in llm_params:
            embedder_params = llm_params['embedder']
        else:
            def find_embedder(d, path=""):
                if isinstance(d, dict):
                    for key, value in d.items():
                        current_path = f"{path}.{key}" if path else key
                        if key == 'embedder':
                            return value
                        elif isinstance(value, dict):
                            result = find_embedder(value, current_path)
                            if result is not None:
                                return result
                return None
            embedder_params = find_embedder(llm_params)
        
        if embedder_params is None:
            raise ValueError(f"Could not find embedder params in LLM state. Available keys: {list(llm_params.keys())}")
        
        # 实例化 Embedder
        vocab_size = self.config.vocab_size
        embed_dim = self.config.embed_dim
        embedder = Embedder(vocab_size=vocab_size, embed_dim=embed_dim)

        if obs.tokenized_prompt is not None and all_image_tokens is not None:
            # 获取文本 Embedding
            text_tokens = embedder.apply({"params": embedder_params}, 
                                       obs.tokenized_prompt, 
                                       method=embedder.encode)
            
            # 【关键修改】冻结 Embedder，防止梯度回传破坏预训练语义
            text_tokens = jax.lax.stop_gradient(text_tokens)
            
            # 应用交叉注意力：让文本特征关注图像特征
            text_tokens_normed = self.cross_attn_norm(text_tokens)
            attended_text = self.cross_attention(
                text_tokens_normed,   # Query: 文本特征
                all_image_tokens,     # Key: 图像特征
                all_image_tokens,     # Value: 图像特征
            )
            # 残差连接
            text_tokens = text_tokens + attended_text
            
            # 拼接图像和融合后的文本特征
            tokens = jnp.concatenate([all_image_tokens, text_tokens], axis=1)
            input_mask = jnp.concatenate([all_image_masks, obs.tokenized_prompt_mask], axis=1)
            # 对于 Value Function，我们通常不需要 Causal Mask (AR mask)，这里保留结构但稍后 Backbone 使用双向 Attention
            ar_mask = jnp.array([False] * all_image_tokens.shape[1] + [False] * text_tokens.shape[1])
            
        elif obs.tokenized_prompt is not None:
            # 只有文本
            text_tokens = embedder.apply({"params": embedder_params}, 
                                       obs.tokenized_prompt, 
                                       method=embedder.encode)
            text_tokens = jax.lax.stop_gradient(text_tokens) # 冻结
            tokens = text_tokens
            input_mask = obs.tokenized_prompt_mask
            ar_mask = jnp.array([False] * text_tokens.shape[1])
            
        elif all_image_tokens is not None:
            # 只有图像
            tokens = all_image_tokens
            input_mask = all_image_masks
            ar_mask = jnp.array([False] * all_image_tokens.shape[1])
        else:
            raise ValueError("No valid input: both images and text are None")

        # 【关键修改】启用 Backbone (Deep Fusion)
        # 将融合后的 tokens 送入 Gemma Transformer Block 进行 18 层推理
        # 使用 self.llm (即 GemmaBlockRunner) 运行 Backbone
        tokens = self.llm(embeddings=tokens, mask=input_mask)

        return tokens, input_mask, ar_mask

    def __call__(self, observation: _model.Observation, *, train: bool = False) -> at.Float[at.Array, "b num_atoms"]:
        """Forward pass to compute value distribution logits.

        Args:
            observation: Input observation containing images and optional text.
            train: Whether in training mode.

        Returns:
            Value distribution logits of shape [batch, NUM_ATOMS].
        """
        tokens, input_mask, ar_mask = self.embed_tokens(observation)

        # 对所有 token 进行平均池化（JAX兼容版本）
        # 使用加权平均池化（根据 mask），添加小的epsilon避免除零
        mask_expanded = input_mask[..., None].astype(tokens.dtype)
        masked_tokens = tokens * mask_expanded
        denom = jnp.maximum(jnp.sum(input_mask, axis=1, keepdims=True), 1)
        pooled = jnp.sum(masked_tokens, axis=1) / denom.astype(tokens.dtype)

        return self.value_head(pooled, train=train)

    def compute_value(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, " b"]:
        """Compute expected value from distribution.

        Args:
            rng: Random key for preprocessing.
            observation: Input observation.
            train: Whether in training mode.

        Returns:
            Expected value of shape [batch].
        """
        # Value model 使用数据中实际可用的图像键
        available_keys = list(observation.images.keys())
        observation = _model.preprocess_observation(rng, observation, train=train, image_keys=available_keys)
        logits = self(observation, train=train)
        return dist_to_value(logits)

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        target: at.Float[at.Array, "..."],
        *,
        train: bool = False,
    ) -> at.Float[at.Array, ""]:
        """Compute cross-entropy loss between predicted distribution and two-hot target.

        Args:
            rng: Random key for preprocessing.
            observation: Input observation.
            target: Either scalar values [batch] or two-hot distribution [batch, 201].
            train: Whether in training mode.

        Returns:
            Scalar cross-entropy loss.
        """
        # Value model 使用数据中实际可用的图像键
        available_keys = list(observation.images.keys())
        observation = _model.preprocess_observation(rng, observation, train=train, image_keys=available_keys)
        logits = self(observation, train=train).astype(jnp.float32)
        target = normalize_value_target_shape(target)
        target_dist = target_to_twohot(target) if target.ndim == 1 else target
        target_dist = target_dist.astype(jnp.float32)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(target_dist * log_probs, axis=-1)

        return jnp.mean(loss)
