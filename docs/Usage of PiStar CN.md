# Usage of PiStar

## Base Environment Setup

Using uv to manage virtual environment.

```bash
git clone https://github.com/ybpy/pistar.git

git submodule update --init --recursive

uv venv --python 3.11.9 /path/to/create/pistar/venv

source /path/to/your/pistar/venv/bin/activate

cd /path/to/pistar

GIT_LFS_SKIP_SMUDGE=1 uv sync --active

GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

uv pip install -r pistar_requirements.txt
```

## PiStar 数据闭环

PiStar 不是只训练一次的流程。一般先用 demo 数据训练一个初始策略，再用策略 rollout 生成新数据，然后把 demo 和 rollout 合并；VLM 价值模型会根据 `value_label` 学习价值函数，并对 rollout 数据重新写入 `adv_ind`，最后用合并且打标后的数据继续训练 PiStar。

推荐顺序：

1. 把 demo 数据转换成 PiStar 使用的 LeRobot schema。
2. 配置 LIBERO 仿真 client 环境，用于后续 evaluation 和 rollout。
3. 用 demo 数据训练一个初始 PiStar checkpoint。
4. 启动 policy server，用初始 checkpoint 做 evaluation / rollout。
5. 用 `scripts/merge_datasets.py` 合并 demo 数据和 rollout 数据。
6. 用合并后的数据训练 VLM value model。
7. 用 VLM 推理并覆盖 rollout 数据里的 `adv_ind`。
8. 用合并且完成 `adv_ind` 打标的数据继续 fine-tune PiStar。

### LeRobot 必需字段


| 字段 | 说明 |
| --- | --- |
| `image` | 主视角相机图像。 |
| `wrist_image` | 腕部相机图像。 |
| `state` | 策略输入的机器人状态。 |
| `actions` | 动作监督目标。 |
| `intervention` | `1` 表示人工/demo/intervention 帧，`0` 表示策略自主 rollout 帧。 |
| `value_label` | VLM value model 的训练监督。 |
| `reward` | 稀疏成功奖励，成功 episode 通常只有最后一帧为 `1`。 |
| `reward_label` | VLM 计算 advantage 时使用的 reward 信号。 |
| `adv_ind` | PiStar 的 advantage 条件，通常为 `positive`、`negative` 或 `none`。 |

`scripts/merge_datasets.py` 只保留上述数据字段，以及 `timestamp`、`frame_index`、`episode_index`、`index`、`task_index`。它只是纯合并脚本，不会补字段、不重算标签、不缩放图像、不转换图像 layout，也不会判断一个 episode 是 demo 还是 rollout。如果某个源数据集缺字段，需要先重新转换或补齐，再进行 merge。

## 数据准备与合并

### 1. 转换 demo 数据

LIBERO demo 数据使用 PiStar 专用转换脚本。该脚本会补齐训练 VLM 和 PiStar 所需字段：

```bash
python examples/libero/pistar_rlds_demo_processing.py \
  --data_dir /path/to/modified_libero_rlds \
  --output_dir /path/to/lerobot_datasets \
  --repo_name libero_demo_pistar
```

对于 demo 数据，转换脚本默认把每条轨迹当作成功的专家轨迹：

- 每一帧 `intervention = 1`。
- 每一帧 `adv_ind = positive`。
- 转换器提供初始标签。合并后在整个数据集上重算 `value_label` 和 `reward_label`。

如果设置了 `--output_dir`，输出路径是 `/path/to/lerobot_datasets/libero_demo_pistar`。如果不设置，LeRobot 会写到 `HF_LEROBOT_HOME` 下。

### 2. 创建 LIBERO client 环境

建议为仿真单独创建虚拟环境，把 MuJoCo/LIBERO 依赖和 PiStar 训练依赖隔离开。

```bash
uv venv --python 3.10 /path/to/create/libero/venv
source /path/to/your/libero/venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
uv pip install --no-deps git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

如果要确保依赖安装到这个环境，可以在 `uv pip install` 命令后追加 `--python /path/to/your/libero/venv/bin/python`。

### 3. 用 demo 数据训练初始 PiStar

回到 PiStar 环境，用转换好的 demo 数据训练初始 checkpoint。训练 config 必须指向 demo 数据集；默认 LIBERO config `pi05_star_libero` 使用 `src/openpi/training/config.py` 中配置的数据源，训练前需要确认它指向刚转换出的 demo dataset。

先计算 normalization statistics：

```bash
source /path/to/your/pistar/venv/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/compute_norm_stats.py --config-name pi05_star_libero
```

再启动训练：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_star_libero --exp-name=demo_init --overwrite
```

如果要继续训练已有实验，把 `--overwrite` 换成 `--resume`：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_star_libero --exp-name=demo_init --resume
```

### 4. 启动 policy server

初始 PiStar checkpoint 训练完成后，在加载 checkpoint 的机器上使用 PiStar 环境启动 server：

```bash
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_star_libero_infer \
  --policy.dir=checkpoints/pi05_star_libero/demo_init/10000
```

`--policy.config` 必须和 checkpoint 对应的 infer config 一致。`--policy.dir` 应该指向某一个具体 step 的 checkpoint 目录。

### 5. 只评估，不保存 rollout

```bash
source /path/to/your/libero/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python examples/libero/main.py \
  --args.adv_ind_input positive \
  --args.task_suite_name libero_10 \
  --args.num_trials_per_task 5
```

如果 EGL 渲染报错，先安装系统依赖后重试：

```bash
sudo -E apt-get update
sudo -E apt-get install -y libegl1 libgl1 libglvnd0 libgles2 libdrm2 libgbm1
```

如果 EGL 仍然失败，可以用 Xvfb + GLX：

```bash
export MUJOCO_GL=glx
xvfb-run -a python examples/libero/main.py \
  --args.adv_ind_input positive \
  --args.task_suite_name libero_10 \
  --args.num_trials_per_task 5
```

### 6. 保存仿真 rollout 数据

使用同一个 client 脚本，但打开 LeRobot rollout 导出：

```bash
source /path/to/your/libero/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python examples/libero/main.py \
  --args.adv_ind_input positive \
  --args.task_suite_name libero_10 \
  --args.num_trials_per_task 5 \
  --args.save_lerobot_rollout true \
  --args.rollout_output_dir /path/to/lerobot_datasets \
  --args.rollout_repo_id libero_rollout_round1 \
  --args.rollout_overwrite true
```

这里有一个容易混淆的点：

- `--args.adv_ind_input positive` 是推理时送进 PiStar policy 的条件。
- 保存下来的 rollout 帧一开始会写 `adv_ind = none`，这只是占位符，后续需要由 `scripts/label_advantage_from_vlm.py` 覆盖成 `positive` 或 `negative`。

仿真 rollout 输出字段和 demo 数据一致，但因为 LIBERO rollout 没有人工接管，所以 `intervention = 0`。成功 episode 会按成功规则生成 `value_label` / `reward` / `reward_label`，失败 episode 会按失败规则生成这些字段。

### 7. 合并 demo 和 rollout 数据

demo 和 rollout 都已经对齐到 PiStar LeRobot schema 后，再进行合并：

```bash
python scripts/merge_datasets.py \
  --sources \
    /path/to/lerobot_datasets/libero_demo_pistar \
    /path/to/lerobot_datasets/libero_rollout_round1 \
  --output /path/to/lerobot_datasets/libero_mixed_round1 \
  --overwrite
```

合并后的数据集会用于 VLM value 训练、VLM advantage 打标，以及下一轮 PiStar fine-tuning。后续多轮迭代时，可以把 demo、round 1 rollout、round 2 rollout 等需要进入训练的数据一起 merge。

## VLM 训练与 Advantage 打标

VLM value model 输入图像观测和任务文本，输出当前帧的价值估计。训练时使用 `value_label` 作为监督；推理时对 rollout 帧计算 value，再结合 `reward_label` 计算 N-step advantage，最后写回 `adv_ind`。

合并后、训练前重算两列：

```bash
python scripts/add_value_labels.py --data_dir /path/to/lerobot_datasets/libero_mixed_round1
```

`Tmax` 是所有输入轨迹的最大帧数，在划分训练集和验证集之前统一统计。`T` 是本条轨迹的末帧下标，`t` 从 0 开始。默认 `C=0.3*Tmax`，成功标签为 `clip(-(T-t)/Tmax, -1, 0)`，失败标签为 `clip((-(T-t)-C)/Tmax, -1, 0)`。配套奖励为普通帧 `-1/Tmax`、成功末帧 `0`、失败末帧 `-0.3`。两列原地更新，`--dry-run` 可先检查且不写入。改过标签后请重新训练 VLM。

### 1. 训练 VLM value model

VLM 基础权重下载链接：

- [AliPan: VLM base weights](https://www.alipan.com/s/h31AF5CBWwA)

下载后把基础权重和 `tokenizer.model` 放到训练机器可访问的位置。训练时 `--load_pretrained` 会加载基础权重；`--tokenizer_path` 需要指向本地的 Gemma tokenizer 文件。

```bash
python scripts/train_value.py \
  --data_dir /path/to/lerobot_datasets/libero_mixed_round1 \
  --checkpoint_dir checkpoints/value_model/libero_round1 \
  --batch_size 32 \
  --num_train_steps 10000 \
  --save_interval 1000 \
  --load_pretrained \
  --tokenizer_path /path/to/gemma/tokenizer.model
```


训练脚本读取 `value_label`，内部会把它映射成 value target。旧数据里如果存在拼写错误的 `value_lable`，脚本仍然兼容；新数据统一使用 `value_label`。

### 2. VLM 推理并打 `adv_ind`

```bash
python scripts/label_advantage_from_vlm.py \
  --data_dir /path/to/lerobot_datasets/libero_mixed_round1 \
  --checkpoint_dir checkpoints/value_model/libero_round1 \
  --lookahead 50 \
  --top_percent 30 \
  --batch_size 8
```



## VLM 打标后继续训练 PiStar

VLM 完成 `adv_ind` 打标后，再回到 PiStar 环境继续训练下一轮策略。训练 config 必须指向已经 merge 且完成 `adv_ind` 打标的数据集，而不是只包含 demo 的初始数据集。



## 真机数据采集与部署

真机脚本在 `control_your_robot` 目录下。建议从该目录运行，避免相对路径和本地 import 出问题：

```bash
cd /path/to/pistar/control_your_robot
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
```

真机采集脚本依赖 `robot.data.collect_lerobot_rl`。运行前需要确认当前 runtime checkout 中存在这个 collector 模块，并且 `PYTHONPATH` 已包含 `control_your_robot/src`。

### 1. 采集真机 demo 数据

使用软件主从遥操作采集 demo：

```bash
python example/collect/collect_lerobot_master_slave_teleop.py
```

运行前先修改脚本底部的配置：

- `REPO_ID`：输出 LeRobot 数据集名称。
- `OUTPUT_DIR`：输出数据集的父目录。
- `TASK_NAME`：任务文本指令。
- `MASTER_CAN` 和 `SLAVE_CAN`：主臂、从臂的 CAN 接口。
- `FPS`、`NUM_EPISODES`、reset joint positions、camera settings 等硬件和采集参数。

该脚本保存的是 demo-style 数据。因为每帧都是人工遥操作，后续应当作为 positive expert data 处理。保存后的数据集可以和 rollout 数据一起用 `scripts/merge_datasets.py` 合并。

### 2. 真机 DAgger rollout 与数据采集

如果需要策略自主执行，同时允许人工接管并保存 rollout 数据，使用 DAgger 部署脚本：

```bash
python example/deploy/piper_dagger_on_PI0.py \
  --model-path /path/to/checkpoint/step_dir \
  --task-name "put the white plug into the two-hole socket" \
  --train-config pi05_star_white_plug_infer \
  --repo-id white_plug_rollout_round1 \
  --output-dir /path/to/lerobot_datasets \
  --num-episode 50 \
  --fps 10 \
  --penalty-value -1.0 \
  --adv-ind positive
```


### 3. 真机 PiStar 纯推理

如果只想跑训练好的 checkpoint，不需要采集 DAgger rollout 数据，使用单臂推理脚本：

```bash
python example/deploy/piper_single_on_PI0.py \
  --model-path /path/to/checkpoint/step_dir \
  --task-name "put the white plug into the two-hole socket" \
  --train-config pi05_star_white_plug_infer \
  --max-step 160 \
  --num-episode 10 \
  --adv-ind positive
```

普通 `pi05` checkpoint 可以不传 `--adv-ind`。PiStar checkpoint 需要传入训练 config 期望的条件，常见值是 `positive` 或 `negative`。
