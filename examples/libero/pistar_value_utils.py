"""
工具脚本：用于保存和加载预计算的 value 值

当你有了训练好的 value 模型后，使用这个脚本：
1. 过一遍所有数据，用模型计算 value
2. 保存为 .npz 文件
3. 在转换脚本中加载使用
"""

import numpy as np
import tensorflow_datasets as tfds
from pathlib import Path
import tyro


def compute_and_save_values(
    data_dir: str,
    output_path: str,
    value_model_path: str | None = None,
    use_random: bool = False,
):
    """
    计算并保存所有数据的 value
    
    Args:
        data_dir: RLDS 数据集路径
        output_path: 输出的 .npz 文件路径
        value_model_path: Value 模型路径
        use_random: 是否使用随机值（用于测试）
    """
    print("=" * 80)
    print("💾 计算并保存 Value 值")
    print("=" * 80)
    
    if use_random:
        print("⚠️  使用随机值（仅用于测试）")
    elif value_model_path:
        print(f"📦 加载 value 模型: {value_model_path}")
        # TODO: 加载实际模型
        # model = load_value_model(value_model_path)
        print("⚠️  模型加载功能待实现，暂时使用随机值")
        use_random = True
    else:
        print("❌ 请提供 --value_model_path 或使用 --use_random")
        return
    
    dataset_names = [
        "libero_10_no_noops",
        "libero_goal_no_noops",
        "libero_object_no_noops",
        "libero_spatial_no_noops",
    ]
    
    episode_indices = []
    step_indices = []
    values = []
    
    global_episode_idx = 0
    
    for dataset_name in dataset_names:
        print(f"\n🔄 处理: {dataset_name}")
        raw_dataset = tfds.load(dataset_name, data_dir=data_dir, split="train")
        
        for episode in raw_dataset:
            steps_list = list(episode['steps'].as_numpy_iterator())
            
            for step_idx, step in enumerate(steps_list):
                # 计算 value
                if use_random:
                    value = float(np.random.randn())
                else:
                    # TODO: 使用实际模型
                    # value = model.predict(step['observation'])
                    value = 0.0
                
                episode_indices.append(global_episode_idx)
                step_indices.append(step_idx)
                values.append(value)
            
            global_episode_idx += 1
            
            if global_episode_idx % 50 == 0:
                print(f"   处理了 {global_episode_idx} episodes, {len(values)} steps")
    
    # 保存
    episode_indices = np.array(episode_indices, dtype=np.int32)
    step_indices = np.array(step_indices, dtype=np.int32)
    values = np.array(values, dtype=np.float32)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_file,
        episode_indices=episode_indices,
        step_indices=step_indices,
        values=values,
    )
    
    print(f"\n✅ 保存完成!")
    print(f"   文件: {output_file}")
    print(f"   Episodes: {global_episode_idx}")
    print(f"   Total steps: {len(values)}")
    print(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def load_and_inspect_values(values_path: str):
    """
    加载并检查保存的 value 文件
    
    Args:
        values_path: .npz 文件路径
    """
    print("=" * 80)
    print("🔍 检查 Value 文件")
    print("=" * 80)
    
    data = np.load(values_path)
    
    episode_indices = data['episode_indices']
    step_indices = data['step_indices']
    values = data['values']
    
    print(f"\n文件: {values_path}")
    print(f"总数据点: {len(values)}")
    print(f"Episodes: {episode_indices.max() + 1}")
    
    print(f"\nValue 统计:")
    print(f"  均值: {values.mean():.4f}")
    print(f"  标准差: {values.std():.4f}")
    print(f"  最小值: {values.min():.4f}")
    print(f"  最大值: {values.max():.4f}")
    print(f"  中位数: {np.median(values):.4f}")
    
    print(f"\n前10个数据点:")
    for i in range(min(10, len(values))):
        print(f"  Episode {episode_indices[i]}, Step {step_indices[i]}: value = {values[i]:.4f}")
    
    print("\n✅ 检查完成!")


import sys

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pistar Value 计算工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # compute 子命令
    parser_compute = subparsers.add_parser("compute", help="计算并保存 value 值")
    parser_compute.add_argument("--data-dir", type=str, required=True, help="RLDS 数据集路径")
    parser_compute.add_argument("--output-path", type=str, required=True, help="输出的 .npz 文件路径")
    parser_compute.add_argument("--value-model-path", type=str, default=None, help="Value 模型路径")
    parser_compute.add_argument("--use-random", action="store_true", help="是否使用随机值（用于测试）")
    
    # inspect 子命令
    parser_inspect = subparsers.add_parser("inspect", help="检查保存的 value 文件")
    parser_inspect.add_argument("--values-path", type=str, required=True, help=".npz 文件路径")
    
    args = parser.parse_args()
    
    if args.command == "compute":
        compute_and_save_values(
            data_dir=args.data_dir,
            output_path=args.output_path,
            value_model_path=args.value_model_path,
            use_random=args.use_random,
        )
    elif args.command == "inspect":
        load_and_inspect_values(values_path=args.values_path)
    else:
        parser.print_help()
