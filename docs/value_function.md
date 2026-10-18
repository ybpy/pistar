# 价值函数模型

## 概述

本文档描述了结合 SigLIP (400M) 和 Gemma3 (270M) 的价值函数 VLM（视觉语言模型）实现，用于强化学习中的价值估计。采用 C51 分布式强化学习的方法进行价值预测，并使用**交叉注意力机制**实现图像和文本的深度融合。

## 架构

### 整体架构图

```
                    +------------------+
                    |     输入图像      |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  SigLIP So400m   |
                    |  (4亿参数)        |
                    |  输出: 1152维     |
                    +--------+---------+
                             |
                             | img_projection
                             | (1152 → 640)
                             v
                    +------------------+
                    |   图像特征        |
                    |   [B, seq_img, 640]
                    +--------+---------+
                             |
                             | (作为 Key/Value)
                             |
+------------------+         v
|   任务文本        |   +------------------+
+--------+---------+   |   交叉注意力层     |
         |             |   8 heads, 640维  |
         v             +--------+---------+
+------------------+         |
| Gemma3 Embedder  |         | (Query来自文本)
| (冻结权重)        |         |
+--------+---------+         v
         |             +------------------+
         |             |  增强的文本特征   |
         +------------>|  (残差连接)      |
                       +--------+---------+
                                |
                                v
                       +------------------+
                       | 拼接图像+文本特征 |
                       | [B, seq_total, 640]
                       +--------+---------+
                                |
                                | 输入 Backbone (Deep Fusion)
                                v
                       +------------------+
                       | Gemma Transformer|
                       |     (18 Layers)  |
                       | [B, seq_total, 640]
                       +--------+---------+
                                |
                                | 加权平均池化
                                v
                       +------------------+
                       |   Value Head     |
                       | LayerNorm + MLP  |
                       | 640 → 320 → 201  |
                       +--------+---------+
                                |
                                v
                       +------------------+
                       |  价值分布 (201)   |
                       |  Softmax → 期望值 |
                       +------------------+
```

### 交叉注意力与深层融合详解

**1. 交叉注意力 (Cross-Attention)**:
- 文本特征作为 Query，图像特征作为 Key/Value。
- 让文本 Token 能够"关注"并聚合相关的图像区域特征。
- 此时特征仅进行了初步的模态交互。

**2. 深层融合 (Deep Fusion via Backbone)**:
- 将拼接后的 [图像 + 文本] 序列送入 **Gemma Transformer Backbone**。
- 经过 **18 层** Transformer Block 的深层处理。
- 允许图像和文本信息在深层语义空间中进行复杂的推理和交互。
- 相比仅使用 Embedder 的浅层模型，具有更强的理解和泛化能力。

**3. 冻结 Embedder (Frozen Embedder)**:
- 文本 Embedder 权重被**冻结** (Stop Gradient)。
- 防止在小数据集上微调时破坏预训练的词汇语义空间。
- 保持大模型的通用语言理解能力。

## C51 分布式价值函数

### 核心思想

预测价值的**分布**，然后通过期望得到价值估计。

### 支架设置

| 参数 | 值 |
|------|-----|
| NUM_ATOMS | 201 |
| V_MIN | -1.0 |
| V_MAX | 0.0 |
| DELTA_Z | 0.005 |
| SUPPORTS | [-1.0, -0.995, -0.99, ..., 0.0] |

### Two-hot 编码

由于真实目标值往往不会精准落在某个支架中心，采用线性插值投影：

```
目标值 y = -0.503

找到左右支架:
  b_left = 99  (对应 z = -0.505)
  b_right = 100 (对应 z = -0.500)

计算权重:
  weight_left = 0.4
  weight_right = 0.6

生成 201 维目标分布:
  P_target[99] = 0.4
  P_target[100] = 0.6
  其余位置 = 0
```

### 损失函数

交叉熵损失：

$$Loss = -\sum_{i=0}^{200} P_{target}^{(i)} \cdot \log(\text{Softmax}(L_{pred})^{(i)})$$

### 价值计算

期望值：

$$Value = \sum_{i=0}^{200} \text{Softmax}(L_{pred})^{(i)} \cdot z_i$$



## 添加/修改的文件

### 新增文件

1. **`src/openpi/models/value_model.py`**
   - `GemmaBlockRunner` 类：继承 Gemma3，支持 Continuous Embeddings 输入以运行 Backbone。
   - `ValueModel` 类：集成了 SigLIP, Gemma Embedder (Frozen), Cross-Attention, Gemma Backbone, Value Head。
   - `NUM_ATOMS, V_MIN, V_MAX`等：C51 分布式 RL 常量。
   - 核心逻辑：`embed_tokens` 中实现了 `Stop Gradient` (冻结) 和 `Backbone` 调用。

2. **`src/openpi/models/value_model_config.py`**
   - `ValueModelConfig`：配置类

### 修改的文件

1. **`src/openpi/training/weight_loaders.py`**
   - 修复参数加载正则 BUG：`missing_regex=".*(value_head|img_projection|cross_att).*"`，确保新层权重不被丢弃。

2. **`src/openpi/models/gemma.py`**
   - 添加 `gemma_270m` 配置。

## 模型参数

### Gemma 270M
| 参数 | 值 |
|------|-----|
| Width | 640 |
| Depth | 18 |
| MLP Dim | 2048 |
| Num Heads | 4 |
| Head Dim | 256 |
| 总参数量 | ~2.7亿 |

### SigLIP So400m/14
| 参数 | 值 |
|------|-----|
| Width | 1152 |
| Depth | 27 |
| Patch Size | 14x14 |
| 总参数量 | ~4亿 |

### Value Head
| 参数 | 值 |
|------|-----|
| 输入维度 | 640 |
| 输出维度 | 201 (NUM_ATOMS) |

## 使用方法

```python
from openpi.models.value_model import ValueModel
from openpi.models.value_model_config import ValueModelConfig
import jax

# 创建配置
config = ValueModelConfig(
    dtype="bfloat16",
    gemma_variant="gemma_270m",
    siglip_variant="So400m/14",
)

# 初始化模型
model = config.create(jax.random.key(0))

# 计算价值（返回期望值）
value = model.compute_value(rng, observation, train=False)
# value shape: [batch]，范围 [-1, 0]

# 获取完整分布 logits
logits = model(observation, train=False)
# logits shape: [batch, 201]

# 计算损失（target_values 需在 [-1, 0] 范围内）
loss = model.compute_loss(rng, observation, target_values, train=True)
```
## 训练

### 步骤 1：添加价值标签

```bash
python3 openpi/scripts/add_value_labels.py \
    --data_dir /public/home/wangsenbao_it/litianheng/lerobot_datasets/piper_plug_libero
```

生成的 `value_label` 定义为：

$$
value\_label = -(T - t) / T
$$

- 标签范围固定为 `[-1, 0]`
- 当前训练脚本默认假设标签已经在这个范围内

### 步骤 2：仅训练集训练

```bash
python3 /public/home/wangsenbao_it/litianheng/openpi/scripts/train_value.py \
    --data_dir /public/home/wangsenbao_it/litianheng/lerobot_datasets/piper_plug_libero \
    --checkpoint_dir ./checkpoints/value_model \
    --batch_size 32 \
    --num_train_steps 10000 \
    --load_pretrained
```

### 步骤 3：训练时启用验证集

```bash
python3 /public/home/wangsenbao_it/litianheng/openpi/scripts/train_value.py \
    --data_dir /public/home/wangsenbao_it/litianheng/lerobot_datasets/piper_plug_libero_train \
    --val_data_dir /public/home/wangsenbao_it/litianheng/lerobot_datasets/piper_plug_libero_val \
    --checkpoint_dir ./checkpoints/value_model \
    --batch_size 64 \
    --num_train_steps 10000 \
    --val_interval 5000 \
    --load_pretrained
```

新增参数说明：

- `--val_data_dir`：验证集目录。若不传，则不会跑验证。
- `--val_interval`：每隔多少个训练 step 运行一次验证。`<=0` 表示禁用验证。

建议：

- 验证集按 `episode` 切分，不要随机按帧切分。
- 尽量让验证集 batch 数不少于 1，即 `val_dataset_size >= batch_size`。

### 验证集 Cross-Entropy

训练脚本会在验证集上计算与训练完全相同的损失函数，即 201 维 two-hot 分布交叉熵：

$$
Loss = -\sum_{i=0}^{200} P_{target}^{(i)} \cdot \log(\text{Softmax}(L_{pred})^{(i)})
$$

验证流程：

1. 在 `val_data_dir` 上前向推理。
2. 对每个验证 batch 计算一次 `model.compute_loss(..., train=False)`。
3. 对所有验证 batch 的 loss 取平均，得到 `val/cross_entropy`。

输出位置：

- 终端日志：

```text
Step 100: val_cross_entropy=2.1374
```

- W&B：

```text
val/cross_entropy
```

解释：

- `val/cross_entropy` 越低越好。
- 它表示模型在未参与训练的数据上，对价值分布预测得有多准确。
- 它比单独看训练损失更适合拿来选 checkpoint 和判断过拟合。

注意：

- 当前验证 loader 使用 `drop_last=True`，所以如果验证集样本数不能被 `batch_size` 整除，最后一个不满 batch 的尾部样本会被丢弃。
- 日志会打印验证集总帧数、验证 batch 数以及被丢弃的尾部帧数。

## VLM 价值推理与 Advantage 打标流程

### 价值推理 (Value Inference)

对当前批次所有 episodes 逐帧执行价值推理：

1. 提取每一帧的相机图像与原始任务指令。+2. 输入到 VLM 价值函数，得到 201 维 logits。+3. 对 logits 做 Softmax 得到分布概率 $p_i$。+4. 与支持集 $z_i \in [-1.0, 0.0]$ 做期望：

$$
V_t = \sum_{i=0}^{200} p_i \cdot z_i
$$

### N 步前瞻优势 (Advantage)

固定 lookahead $N=50$，对轨迹中每个时间步 $t$ 计算：

$$
A_t = \sum_{k=0}^{N-1} r_{t+k} + V_{t+N} - V_t
$$

若 $t+N \ge T$（轨迹不足 50 步），则：

- 累积奖励取到最后一帧
- $V_{t+N}$ 取 $V_{T-1}$

### 全局阈值与打标

1. 汇总该批次所有“机器人自主执行步骤”的优势值。+2. 计算 70 百分位数作为阈值：

```
threshold = np.percentile(advs, 70)
```

3. 再次遍历所有帧，执行指令文本修改：

在 LeRobot 数据集中，`intervention` 字段格式为：

```
"intervention": {
    "dtype": "int64",
    "shape": [1]
}
```

若 `intervention = 1`，代表人工干预，直接标注为 `positive`。+
- 若该帧为人类专家接管（Human Intervention），或 $A_t > threshold$：

  ```
  <原始指令>. Advantage: positive
  ```

- 否则：

  ```
  <原始指令>. Advantage: negative
  ```

最终将更新后的指令文本覆盖写回原数据集，用于下一轮 VLA 微调训练。+
### 脚本参考

已提供批量处理脚本：

```
python scripts/label_advantage_from_vlm.py \
  --data_dir /path/to/lerobot_dataset \
  --checkpoint_dir /path/to/value_ckpt_dir
```

**重要参数说明：**
- `--human_col`: 人工干预列名（默认为 `intervention`）
  - 当 `intervention = 1` 时，该帧直接标记为 `positive`
  - 当 `intervention = 0` 时，使用 advantage 阈值判断
- `--reward_col`: 奖励列名（默认会自动检测 `reward_label`）
  - 当前脚本约定数据集奖励字段名为 `reward_label`
- `--lookahead`: N步前瞻计算advantage（默认为50）
- `--batch_size`: 批处理大小（默认为8）

脚本会按 episode 自动区分 demo 和 rollout：
- 若 episode 内所有帧 `intervention=1`，视为 demo，整条跳过。
- 若 episode 内存在任意 `intervention=0`，视为 rollout，整条全量推理并覆盖写回 `adv_ind`。

**LeRobot 数据集格式匹配：**
- 自动检测 `intervention`、`value`、`image`、`wrist_image` 列
- 支持 `task_index` 来从 `meta/tasks.jsonl` 加载任务指令
- 指令列支持：`prompt`、`instruction`、`task`、`language_instruction`、`text`

**完整使用示例：**
```bash
# 使用默认参数（匹配 LeRobot 标准格式）
python scripts/label_advantage_from_vlm.py \
  --data_dir /path/to/lerobot_dataset \
  --checkpoint_dir /path/to/value_ckpt_dir

# 或者显式指定所有参数
python scripts/label_advantage_from_vlm.py \
  --data_dir /path/to/lerobot_dataset \
  --checkpoint_dir /path/to/value_ckpt_dir \
  --human_col intervention \
  --reward_col reward_label \
  --instruction_col prompt \
  --lookahead 50 \
  --batch_size 8

# 多轮 rollout：同一个命令会跳过 demo，并重算所有 rollout。
python scripts/label_advantage_from_vlm.py \
  --data_dir /path/to/mixed_lerobot_dataset \
  --checkpoint_dir /path/to/value_ckpt_dir \
  --lookahead 50 \
  --batch_size 8
```


## 1-3 camera value training and inference

The value backbone shares the same SigLIP weights for each view. Camera selection
adds no learned parameters. The input adapter now retains left and right wrist
images independently, and cross-attention respects missing-view masks.

Select cameras using `--camera_names` in both `scripts/train_value.py` and
`scripts/label_advantage_from_vlm.py` (including its HDF5 multi-GPU mode):

```bash
# Add one of these to the existing value training/inference command:
--camera_names head                 # one camera
--camera_names head left            # two; unchanged default for legacy checkpoints
--camera_names head right           # head plus right wrist
--camera_names head left right      # three
```

Any 1-3 distinct names from `head`, `left`, `right` are accepted. Their order is
canonicalized. `left` retains the historical model key `wrist_0_rgb`; `right` uses
`right_wrist_0_rgb`. A selected missing stream is zero-filled with mask=False;
no available selected images is an error. The legacy wrist-only-to-head fallback
is retained when head is requested.

Python configuration and input transformation:

```python
from openpi.models.value_model_config import ValueModelConfig
from openpi.policies.value_policy import ValueInputs
config = ValueModelConfig(camera_names=("head", "left", "right"))
inputs = ValueInputs(camera_names=config.camera_names)
# Inputs accept cam_high / cam_left_wrist / cam_right_wrist, or
# image / wrist_image / right_wrist_image, with nested or flattened keys.
```

Training writes `value_camera_config.json` to the experiment directory and each
saved step directory. Inference reads it automatically unless explicitly
overridden with `--camera_names`; old checkpoints without it default to head+left.
LeRobot inference can specify `--base_image_col`, `--wrist_image_col`, and
`--right_wrist_image_col` for custom column names. The default right column when
right is selected is `right_wrist_image`. Never substitute a copied wrist image
for a real third camera unless the explicit legacy `--copy_wrist_to_right` option
is intentional.

Use a new checkpoint directory when changing camera layout. The input schema is
recorded to prevent mixing different layouts in one training run or resumed
HDF5 inference cache. Existing model weights remain loadable, but a two-view
checkpoint has not been trained to use the additional right view; evaluate or
fine-tune with the chosen layout before relying on its scores. Existing dataset
labels and model checkpoints are not rewritten by this code change.

Focused checks: `python -m pytest -q scripts/test_value_multicamera.py`.
