"""
Pistar 数据处理流程脚本，支持复杂的 reward/value/adv/epsilon 计算

处理流程：
1. Pass 1: 加载所有数据，转换 reward，计算 value
2. Pass 2: 计算 advantages，按 task 统计 advantages 并计算 epsilon (70% 分位数)
3. Pass 3: 基于 epsilon 计算 adv_ind，写入 LeRobot 数据集

注意：
- Value 取值范围: [-1.0, 0.0]
- 所有 value 使用 --default_value 参数设置（默认 0.0）
- value_model_path 功能待实现
- 可通过 --default_adv_ind 跳过 adv 计算，直接设置所有 adv_ind

Usage:
# 计算 adv 和 adv_ind
python examples/libero/pistar_data_processing.py \
    --data_dir /path/to/modified_libero_rlds \
    --default_value 0.0 \
    --n_steps 10

# 跳过 adv 计算，直接设置 adv_ind
python examples/libero/pistar_data_processing.py \
    --data_dir /path/to/modified_libero_rlds \
    --default_value 0.0 \
    --default_adv_ind positive

# 使用 value 模型（待实现）
python examples/libero/pistar_data_processing.py \
    --data_dir /path/to/modified_libero_rlds \
    --value_model_path /path/to/value_model.pth \
    --n_steps 10

# unbuffered 输出日志（实时查看）
python -u examples/libero/pistar_data_processing.py ... 
"""

import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import numpy as np
import tyro


REPO_NAME = "ybpy/libero_pistar"
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]


def transform_reward(original_reward: float, is_terminal: bool, is_last: bool, episode_length: int) -> float:
    """
    转换 reward 的规则：
    1. 如果 is_terminal 或 is_last 为 True:
       - 原始 reward = 1.0 → 0.0
       - 原始 reward = 0.0 → -1.0
    2. 如果 is_terminal 和 is_last 都为 False (中间步骤):
       - reward = -1 / episode_length
    """
    if is_terminal or is_last:
        # 至少一个为 True
        if original_reward == 1.0:
            return 0.0
        else:  # original_reward == 0.0
            return -1.0
    else:
        # 都为 False (中间步骤): reward = -1 / episode_length
        return -1.0 / episode_length


def compute_value_placeholder(step_data: dict) -> float:
    """
    占位函数：计算 value
    TODO: 替换为实际的模型推理
    
    Args:
        step_data: 包含 observation, action 等的字典
    
    Returns:
        value: 预测的状态价值 (范围: [-1.0, 0.0])
    """
    # 当前使用 0.0 作为默认值
    # Value 的取值范围预定在 [-1.0, 0.0] 之间
    return 0.0


def compute_advantage(
    rewards: np.ndarray,
    values: np.ndarray,
    n_steps: int,
    gamma: float = 1.0
) -> np.ndarray:
    """
    计算 advantage
    
    adv[t] = sum(rewards[t:t+N]) + value[t+N] - value[t]
    
    注意：对于 episode 的最后 N 个 step：
    - 窗口会截断到 episode 结束
    - 使用 episode 最后一个 step 的 value 作为 bootstrap
    - 例如：t=95, T=100, N=10 时，actual_steps=5
      adv[95] = sum(rewards[95:100]) + gamma^5 * value[99] - value[95]
    
    Args:
        rewards: shape (T,) 的 reward 数组
        values: shape (T,) 的 value 数组
        n_steps: N-step 窗口大小
        gamma: 折扣因子 (默认 1.0)
    
    Returns:
        advantages: shape (T,) 的 advantage 数组
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    
    for t in range(T):
        # 计算 N-step return (窗口会自动截断到 episode 结束)
        n_step_return = 0.0
        actual_steps = min(n_steps, T - t)  # 实际能看到的步数
        
        for i in range(actual_steps):
            n_step_return += (gamma ** i) * rewards[t + i]
        
        # 添加 bootstrap value
        if t + n_steps < T:
            # 正常情况：加上 N 步后的 value
            n_step_return += (gamma ** n_steps) * values[t + n_steps]
        else:
            # 最后 N 个 step：加上 episode 最后一个 step 的 value
            n_step_return += (gamma ** actual_steps) * values[T - 1]
        
        # advantage = n_step_return - value[t]
        advantages[t] = n_step_return - values[t]
    
    return advantages


def main(
    data_dir: str,
    *,
    n_steps: int = 10,
    value_model_path: str | None = None,
    default_value: float = 0.0,
    default_adv_ind: str | None = None,
    epsilon_percentile: float = 70.0,
    repo_name: str = REPO_NAME,
    push_to_hub: bool = False,
):
    """
    Pistar 数据处理和转换
    
    Args:
        data_dir: RLDS 数据集路径
        n_steps: N-step advantage 计算的窗口大小
        value_model_path: Value 模型路径 (可选，待实现)
        default_value: 默认 value 值 (范围 [-1.0, 0.0]，默认 0.0)
        default_adv_ind: 默认 adv_ind 值 ("positive" 或 "negative")，
                        如果设置，将跳过 adv 和 epsilon 计算
        epsilon_percentile: epsilon 的分位数 (默认 70.0)
        repo_name: 输出数据集名称
        push_to_hub: 是否推送到 HuggingFace Hub
    """
    
    print("=" * 80)
    print("🚀 Pistar 数据处理流程")
    print("=" * 80)
    print(f"N-step window: {n_steps}")
    print(f"Default value: {default_value}")
    if default_adv_ind:
        print(f"Default adv_ind: {default_adv_ind} (跳过 adv 计算)")
    else:
        print(f"Epsilon percentile: {epsilon_percentile}%")
    print(f"Output repo: {repo_name}")
    
    # ========================================================================
    # Pass 1: 加载所有数据，转换 reward，计算 value
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 Pass 1: 加载数据并计算 reward/value")
    print("=" * 80)
    
    # 检查 value 模型
    if value_model_path:
        print(f"⚠️  Value model loading not yet implemented")
        print(f"    Using default value: {default_value}")
    else:
        print(f"📌 Using default value: {default_value}")
    
    # 存储所有 episodes 的数据
    all_episodes_data = []
    global_episode_idx = 0
    
    for dataset_name in RAW_DATASET_NAMES:
        print(f"\n🔄 Processing: {dataset_name}")
        raw_dataset = tfds.load(dataset_name, data_dir=data_dir, split="train")
        
        for episode in raw_dataset:
            episode_data = {
                'steps': [],
                'task': None,
                'dataset_name': dataset_name,
                'global_episode_idx': global_episode_idx,
            }
            
            steps_list = list(episode['steps'].as_numpy_iterator())
            episode_length = len(steps_list)
            
            # 获取 task
            task = steps_list[0]['language_instruction']
            task = task.decode() if isinstance(task, bytes) else task
            episode_data['task'] = task
            
            for step_idx, step in enumerate(steps_list):
                # 转换 reward
                original_reward = float(step['reward'])
                is_terminal = bool(step['is_terminal'])
                is_last = bool(step['is_last'])
                
                transformed_reward = transform_reward(
                    original_reward, is_terminal, is_last, episode_length
                )
                
                # 获取或计算 value
                if value_model_path:
                    # TODO: 使用实际模型计算 value
                    value = compute_value_placeholder(step)
                else:
                    value = default_value
                
                step_data = {
                    'observation': step['observation'],
                    'action': step['action'],
                    'original_reward': original_reward,
                    'transformed_reward': transformed_reward,
                    'value': value,
                    'is_terminal': is_terminal,
                    'is_last': is_last,
                    'step_idx': step_idx,
                }
                
                episode_data['steps'].append(step_data)
            
            all_episodes_data.append(episode_data)
            global_episode_idx += 1
            
            if global_episode_idx % 50 == 0:
                print(f"   Processed {global_episode_idx} episodes")
    
    print(f"\n✅ Pass 1 complete: {global_episode_idx} episodes loaded")
    
    # ========================================================================
    # Pass 2: 计算 advantages，统计每个 task 的 epsilon
    # ========================================================================
    task_epsilon = {}
    
    if not default_adv_ind:
        # 只有在需要计算 adv_ind 时才计算 epsilon
        print("\n" + "=" * 80)
        print("📈 Pass 2: 计算 advantages 并统计 epsilon")
        print("=" * 80)
        
        # 按 task 收集所有 advantages
        task_advantages = defaultdict(list)
        
        for episode_data in all_episodes_data:
            task = episode_data['task']
            steps = episode_data['steps']
            
            # 提取 rewards 和 values
            rewards = np.array([s['transformed_reward'] for s in steps], dtype=np.float32)
            values = np.array([s['value'] for s in steps], dtype=np.float32)
            
            # 计算 advantages
            advantages = compute_advantage(rewards, values, n_steps)
            
            # 存储 advantages 到 episode_data 中，供 Pass 3 使用
            episode_data['advantages'] = advantages
            
            # 收集到 task_advantages 中
            task_advantages[task].extend(advantages.tolist())
        
        # 计算每个 task 的 epsilon (基于 advantages)
        for task, advantages in task_advantages.items():
            epsilon = np.percentile(advantages, epsilon_percentile)
            task_epsilon[task] = epsilon
            print(f"Task: {task[:50]}...")
            print(f"  Advantages count: {len(advantages)}")
            print(f"  Epsilon ({epsilon_percentile}%): {epsilon:.4f}")
        
        print(f"\n✅ Pass 2 complete: {len(task_epsilon)} unique tasks")
    else:
        print("\n" + "=" * 80)
        print(f"⏭️  Pass 2: 跳过 (使用默认 adv_ind: {default_adv_ind})")
        print("=" * 80)
    
    # ========================================================================
    # Pass 3: 计算 adv_ind，写入数据集
    # ========================================================================
    print("\n" + "=" * 80)
    print("💾 Pass 3: 计算 adv_ind 并写入数据集")
    print("=" * 80)
    
    # 清理已存在的数据集
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"🗑️  Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # 创建 LeRobot 数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            "reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"],
            },
            "value": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["value"],
            },
            "adv": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["adv"],
            },
            "epsilon": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["epsilon"],
            },
            "adv_ind": {
                "dtype": "string",
                "shape": (1,),
                "names": ["adv_ind"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # 写入数据
    total_steps = 0
    for ep_idx, episode_data in enumerate(all_episodes_data):
        task = episode_data['task']
        steps = episode_data['steps']
        
        if default_adv_ind:
            # 使用默认 adv_ind，跳过 adv 计算
            advantages = np.zeros(len(steps), dtype=np.float32)  # adv 全部为 0
            epsilon = 0.0  # epsilon 也设为 0
        else:
            # 使用 Pass 2 中计算好的 advantages 和 epsilon
            epsilon = task_epsilon[task]
            advantages = episode_data['advantages']
        
        # 写入每个 step
        for step_idx, step_data in enumerate(steps):
            adv = advantages[step_idx]
            
            if default_adv_ind:
                adv_ind = default_adv_ind
            else:
                adv_ind = "positive" if adv > epsilon else "negative"
            
            dataset.add_frame({
                "image": step_data['observation']['image'],
                "wrist_image": step_data['observation']['wrist_image'],
                "state": step_data['observation']['state'],
                "actions": step_data['action'],
                "task": task,
                "reward": np.array([step_data['transformed_reward']], dtype=np.float32),
                "value": np.array([step_data['value']], dtype=np.float32),
                "adv": np.array([adv], dtype=np.float32),
                "epsilon": np.array([epsilon], dtype=np.float32),
                "adv_ind": adv_ind,
            })
            total_steps += 1
        
        dataset.save_episode()
        
        if (ep_idx + 1) % 50 == 0:
            print(f"   Written {ep_idx + 1}/{len(all_episodes_data)} episodes")
    
    print(f"\n✅ Pass 3 complete!")
    print(f"   Total episodes: {len(all_episodes_data)}")
    print(f"   Total steps: {total_steps}")
    print(f"   Output path: {output_path}")
    
    # 推送到 Hub
    if push_to_hub:
        print(f"\n📤 Pushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds", "advanced", "value", "advantage"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"✅ Successfully pushed to Hub!")
    
    print("\n" + "=" * 80)
    print("🎉 All processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    tyro.cli(main)
