# Pistar 数据处理流程文档

## 概述

这套脚本实现了 Pistar 数据处理流程，包括 reward 转换、value 计算、advantage 计算和分类。

## 处理逻辑

### 1. Reward 转换

```python
def transform_reward(original_reward, is_terminal, is_last, episode_length):
    if is_terminal or is_last:
        # 至少一个为 True
        if original_reward == 1.0:
            return 0.0
        else:  # original_reward == 0.0
            return -1.0
    else:
        # 都为 False (中间步骤): reward = -1 / episode_length
        return -1.0 / episode_length
```

### 2. Advantage 计算

```
adv[t] = Σ(reward[t:t+N]) + value[t+N] - value[t]
```

- N-step return + bootstrap value - current value
- 需要先计算所有 value 才能计算 adv

### 3. Epsilon 计算

- 对于每个 task，收集所有相同 task 的 value
- epsilon = 该 task 所有 value 的 70% 分位数（上30%）

### 4. Adv_ind 分类

```python
adv_ind = "positive" if adv > epsilon else "negative"
```

## 三遍处理流程

### Pass 1: 数据加载与预处理
- 加载所有 RLDS 数据
- 转换 reward
- 计算或加载 value（通过模型或预计算文件）

### Pass 2: 统计与计算 Epsilon
- 按 task 分组统计所有 value
- 计算每个 task 的 epsilon（70% 分位数）

### Pass 3: 计算 Advantage 并写入
- 使用 N-step return 计算 advantage
- 与 epsilon 比较得到 adv_ind
- 写入 LeRobot 数据集

## 脚本文件

### 1. pistar_data_processing.py

Pistar 数据处理主脚本，完整的三遍处理流程。

**使用方法：**

```bash
# 使用默认 value 0.0 和默认 adv_ind（测试）
python3 examples/libero/pistar_data_processing.py \
  --data_dir /public/home/chenyuyao1/dataset/modified_libero_rlds \
  --default_value 0.0 \
  --default_adv_ind positive

# 使用预计算的 value
python3 examples/libero/pistar_data_processing.py \
  --data_dir /public/home/chenyuyao1/dataset/modified_libero_rlds \
  --precomputed_values /path/to/values.npz \
  --n_steps 10

# 自定义参数
python3 examples/libero/pistar_data_processing.py \
  --data_dir /public/home/chenyuyao1/dataset/modified_libero_rlds \
  --precomputed_values values.npz \
  --n_steps 5 \
  --epsilon_percentile 70.0 \
  --repo_name "your_name/libero_processed" \
  --push_to_hub
```

**参数说明：**
- `--data_dir`: RLDS 数据集路径
- `--n_steps`: N-step advantage 窗口大小
- `--value_model_path`: Value 模型路径（待实现）
- `--precomputed_values`: 预计算的 value 文件（.npz）
- `--use_random_values`: 使用随机 value（仅测试）
- `--epsilon_percentile`: Epsilon 分位数（默认 70.0）
- `--repo_name`: 输出数据集名称
- `--push_to_hub`: 是否推送到 HuggingFace Hub

### 2. test_pistar_processing.py

快速测试脚本，仅处理 3 个 episodes。

**使用方法：**

```bash
python3 examples/libero/test_pistar_processing.py
```

### 3. pistar_value_utils.py

Value 计算和保存工具。

**使用方法：**

```bash
# 计算并保存 value（使用随机值测试）
python3 examples/libero/pistar_value_utils.py compute \
  --data_dir /public/home/chenyuyao1/dataset/modified_libero_rlds \
  --output_path values_random.npz \
  --use_random

# 使用实际模型（待实现）
python3 examples/libero/pistar_value_utils.py compute \
  --data_dir /public/home/chenyuyao1/dataset/modified_libero_rlds \
  --output_path values_model.npz \
  --value_model_path /path/to/model.pth

# 检查保存的 value 文件
python3 examples/libero/pistar_value_utils.py inspect \
  --values_path values_random.npz
```

## 完整工作流程

### 场景 1: 使用预计算的 Value

```bash
# Step 1: 用你的模型计算 value 并保存
# TODO: 修改 pistar_value_utils.py 中的模型加载代码
python3 examples/libero/pistar_value_utils.py compute \
  --data_dir /public/home/chenyuyao1/dataset/modified_libero_rlds \
  --output_path my_values.npz \
  --value_model_path /path/to/your/model.pth

# Step 2: 检查保存的 value
python3 examples/libero/pistar_value_utils.py inspect \
  --values_path my_values.npz

# Step 3: 运行完整转换
python3 examples/libero/pistar_data_processing.py \
  --data_dir /public/home/chenyuyao1/dataset/modified_libero_rlds \
  --precomputed_values my_values.npz \
  --n_steps 10 \
  --repo_name "your_name/libero_processed"
```

### 场景 2: 快速测试（使用默认值）

```bash
# 运行测试脚本
python3 examples/libero/test_pistar_processing.py
```

## 输出数据集字段

转换后的 LeRobot 数据集包含以下字段：

| 字段 | 类型 | Shape | 说明 |
|------|------|-------|------|
| `image` | image | (256,256,3) | 主相机图像 |
| `wrist_image` | image | (256,256,3) | 手腕相机图像 |
| `state` | float32 | (8,) | 机器人状态 |
| `actions` | float32 | (7,) | 机器人动作 |
| `reward` | float32 | (1,) | **转换后的** reward |
| `value` | float32 | (1,) | 状态价值 |
| `adv` | float32 | (1,) | Advantage |
| `epsilon` | float32 | (1,) | 当前 task 的阈值 |
| `adv_ind` | string | (1,) | "positive" 或 "negative" |
| `task` | string | - | 任务描述 |

## 注意事项

### 关于 Value 计算

当前脚本中的 `compute_value_placeholder` 函数返回默认值 0.0。实际使用时需要：

1. 训练一个 value 网络
2. 修改 `pistar_value_utils.py` 中的模型加载代码
3. 使用 `pistar_value_utils.py compute` 预计算所有 value
4. 在转换时使用 `--precomputed_values` 加载

### 关于 Epsilon

- 默认使用 70% 分位数（上 30%）
- 可以通过 `--epsilon_percentile` 调整
- 每个 task 有独立的 epsilon 值

### 关于 N-step

- N-step 窗口大小影响 advantage 计算
- 较大的 N 考虑更长期的回报
- 需要根据任务长度调整

## 待实现功能

1. **Value 模型集成**
   - 加载训练好的 value 模型
   - 批量推理优化
   - GPU 加速

2. **并行处理**
   - 多进程数据加载
   - 并行 value 计算

3. **断点续传**
   - 保存处理进度
   - 支持中断恢复

## 示例输出

```
================================================================================
🚀 Pistar 数据处理流程
================================================================================
N-step window: 10
Epsilon percentile: 70.0%
Output repo: ybpy/libero_advanced

================================================================================
📊 Pass 1: 加载数据并计算 reward/value
================================================================================
⚠️  Using random values for testing

🔄 Processing: libero_10_no_noops
   Processed 50 episodes
   ...

✅ Pass 1 complete: 379 episodes loaded

================================================================================
📈 Pass 2: 计算每个 task 的 epsilon
================================================================================
Task: put the white mug on the left plate and put...
  Values count: 5478
  Epsilon (70.0%): 0.4523
...

✅ Pass 2 complete: 40 unique tasks

================================================================================
💾 Pass 3: 计算 advantage 并写入数据集
================================================================================
   Written 50/379 episodes
   ...

✅ Pass 3 complete!
   Total episodes: 379
   Total steps: 106234
   Output path: ~/.cache/huggingface/lerobot/ybpy/libero_advanced

================================================================================
🎉 All processing complete!
================================================================================
```
