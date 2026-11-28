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

sudo -E apt update && sudo -E apt install screen -y
```

## PiStar Data Closed Loop

PiStar is not a one-time training pipeline. In general, you first train an initial policy with demo data, then use the policy to generate new data through rollout, and then merge the demo and rollout data. The VLM value model learns the value function according to `value_label`, rewrites `adv_ind` for the rollout data, and finally continues training PiStar with the merged and labeled data.

Recommended order:

1. Convert the demo data into the LeRobot schema used by PiStar.
2. Configure the LIBERO simulation client environment for later evaluation and rollout.
3. Train an initial PiStar checkpoint using the demo data.
4. Start the policy server and use the initial checkpoint for evaluation / rollout.
5. Merge the demo data and rollout data with `scripts/merge_datasets.py`.
6. Train the VLM value model with the merged data.
7. Run VLM inference and overwrite `adv_ind` in the rollout data.
8. Continue fine-tuning PiStar with the merged data whose `adv_ind` labels have been completed.

### Required LeRobot Fields

| Field | Description |
| --- | --- |
| `image` | Main-view camera image. |
| `wrist_image` | Wrist camera image. |
| `state` | Robot state used as policy input. |
| `actions` | Action supervision target. |
| `intervention` | `1` indicates human/demo/intervention frames, and `0` indicates autonomous policy rollout frames. |
| `value_label` | Training supervision for the VLM value model. |
| `reward` | Sparse success reward. In a successful episode, usually only the last frame is `1`. |
| `reward_label` | Reward signal used when the VLM computes advantage. |
| `adv_ind` | PiStar's advantage condition, usually `positive`, `negative`, or `none`. |

`scripts/merge_datasets.py` only keeps the data fields above, as well as `timestamp`, `frame_index`, `episode_index`, `index`, and `task_index`. It is only a pure merging script: it does not fill missing fields, recompute labels, rescale images, convert image layout, or determine whether an episode is demo or rollout. If a source dataset is missing some fields, it needs to be re-converted or completed before merging.

## Data Preparation and Merging

### 1. Convert demo data

You can download the raw LIBERO dataset (rlds format, filtering failure episodes) from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds). 

LIBERO demo data uses the PiStar-specific conversion script. This script fills in the fields required for training both the VLM and PiStar:

```bash
python examples/libero/pistar_rlds_demo_processing.py \
  --data_dir /path/to/modified_libero_rlds \
  --output_dir /path/to/lerobot_datasets \
  --repo_name libero_demo_pistar
```

For demo data, the conversion script treats each trajectory as a successful expert trajectory by default:

- Every frame has `intervention = 1`.
- Every frame has `adv_ind = positive`.
- The converter's initial `value_label` and `reward_label` are replaced after merging by the labeling step below.

If `--output_dir` is set, the output path is `/path/to/lerobot_datasets/libero_demo_pistar`. If it is not set, LeRobot writes the dataset under `HF_LEROBOT_HOME`.

### 2. Create the LIBERO client environment

It is recommended to create a separate virtual environment for simulation so that MuJoCo/LIBERO dependencies are isolated from PiStar training dependencies.

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

If you want to make sure the dependencies are installed into this environment, append `--python /path/to/your/libero/venv/bin/python` to the `uv pip install` command.

### 3. Train the initial PiStar with demo data

Return to the PiStar environment and train the initial checkpoint with the converted demo data. The training config must point to the demo dataset. The default LIBERO config `pi05_star_libero` uses the data source configured in `src/openpi/training/config.py`, so before training you need to confirm that it points to the newly converted demo dataset.

First compute the normalization statistics:

```bash
source /path/to/your/pistar/venv/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/compute_norm_stats.py --config-name pi05_star_libero
```

Then start training:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_star_libero --exp-name=demo_init --overwrite
```

If loading the `pi05_base` checkpoint from the cloud (`gs://openpi-assets/checkpoints/pi05_base`) fails, you can manually download the model weights to a local path before training:

```bash
pip install gsutil

mkdir -p /path/to/pi05_base

gsutil -m rsync -r \
  gs://openpi-assets/checkpoints/pi05_base \
  /path/to/pi05_base
```

After the download is complete, modify the `weight_loader` path of the corresponding `TrainConfig` in `pistar/src/openpi/training/config.py` to the local checkpoint path `/path/to/pi05_base/params`, then you can start training.

If you want to continue training an existing experiment, replace `--overwrite` with `--resume`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_star_libero --exp-name=demo_init --resume
```

### 4. Start the policy server

After the initial PiStar checkpoint is trained, start the server in the PiStar environment on the machine that loads the checkpoint:

```bash
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_star_libero_infer \
  --policy.dir=checkpoints/pi05_star_libero/demo_init/10000
```

`--policy.config` must match the infer config corresponding to the checkpoint. `--policy.dir` should point to the checkpoint directory of one specific step.

### 5. Evaluation only, without saving rollout

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

If EGL rendering reports an error, install the system dependencies first and try again:

```bash
sudo -E apt-get update
sudo -E apt-get install -y libegl1 libgl1 libglvnd0 libgles2 libdrm2 libgbm1
```

If EGL still fails, use Xvfb + GLX:

```bash
export MUJOCO_GL=glx
xvfb-run -a python examples/libero/main.py \
  --args.adv_ind_input positive \
  --args.task_suite_name libero_10 \
  --args.num_trials_per_task 5
```

### 6. Save simulation rollout data

Use the same client script, but enable LeRobot rollout export:

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

There is one point that can easily be confusing here:

- `--args.adv_ind_input positive` is the condition fed into the PiStar policy during inference.
- The saved rollout frames are initially written with `adv_ind = none`. This is only a placeholder and needs to be overwritten as `positive` or `negative` later by `scripts/label_advantage_from_vlm.py`.

LIBERO rollouts have `intervention = 0`. The sparse `reward` records success (1 on a successful last frame, 0 otherwise). Recompute the value and reward labels on the full mixed dataset after merging.

### 7. Merge demo and rollout data

After both demo and rollout data have been aligned to the PiStar LeRobot schema, merge them:

```bash
python scripts/merge_datasets.py \
  --sources \
    /path/to/lerobot_datasets/libero_demo_pistar \
    /path/to/lerobot_datasets/libero_rollout_round1 \
  --output /path/to/lerobot_datasets/libero_mixed_round1 \
  --overwrite
```

The merged dataset will be used for VLM value training, VLM advantage labeling, and the next round of PiStar fine-tuning. In later multi-round iterations, you can merge the demo data, round 1 rollout, round 2 rollout, and any other data that needs to enter training together.

## VLM Training and Advantage Labeling

The VLM value model takes image observations and task text as input and outputs a value estimate for the current frame. During training, it uses `value_label` as supervision. During inference, it computes value for rollout frames, then combines it with `reward_label` to compute N-step advantage, and finally writes the result back to `adv_ind`.

Before training, replace both labels on the merged dataset:

```bash
python scripts/add_value_labels.py --data_dir /path/to/lerobot_datasets/libero_mixed_round1
```

`Tmax` is the longest episode in frames across all supplied data, before the train/validation split. Let `G = -(T-t) - C * failed` be the raw return, with frame `t` and last frame `T` both zero-based.

- `--normalization tmax-clip` (default): `value_label = clip(G / Tmax, -1, 0)`, with `C = 0.3 * Tmax` by default. Non-terminal rewards are `-1 / Tmax`; the terminal reward is `0` for success and `-0.3` by default for failure.
- `--normalization plus-c`: `value_label = G / abs(min(G))` with no clipping and `C = 0.3 * Tmax` by default. The minimum is taken over the whole corpus. If the longest episode fails, the divisor is `Tmax - 1 + C`. The rewards use that same divisor.

Both the LeRobot script above and `scripts/add_hdf5_value_labels.py` accept these modes. Use `--failure-cost` for an absolute penalty or `--failure-cost-ratio` for a multiple of `Tmax`. Each run replaces `value_label` and `reward_label`; use separate dataset copies to compare modes, and use `--dry-run` to inspect the divisor before writing. Use a VLM checkpoint trained with the same labels when computing advantages.

### 1. Train the VLM value model

The VLM base checkpoint weights are available from two sources: [ybpy/vlm_ckpt · Hugging Face](https://huggingface.co/ybpy/vlm_ckpt) and [Google Drive](https://drive.google.com/drive/folders/1pS6J82pvEwqUJt16n1uKKFm_MmyyKuuu?usp=drive_link). You may download the checkpoint from either source as you like before running training or inference.

After downloading, place the base weights and `tokenizer.model` somewhere accessible from the training machine. During training, `--load_pretrained` loads the base weights; `--tokenizer_path` needs to point to the local Gemma tokenizer file.

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

The training script reads `value_label` and internally maps it to the value target. If old data contains the misspelled `value_lable`, the script is still compatible with it; new data should consistently use `value_label`.

### 2. Run VLM inference and label `adv_ind`

```bash
python scripts/label_advantage_from_vlm.py \
  --data_dir /path/to/lerobot_datasets/libero_mixed_round1 \
  --checkpoint_dir checkpoints/value_model/libero_round1 \
  --lookahead 50 \
  --top_percent 30 \
```

## Continue Training PiStar after VLM Labeling

After the VLM completes `adv_ind` labeling, return to the PiStar environment and continue training the next-round policy. The training config must point to the dataset that has already been merged and labeled with `adv_ind`, rather than the initial dataset that only contains demo data.

## Real-Robot Data Collection and Deployment

The real-robot scripts are under the `control_your_robot` directory. It is recommended to run them from this directory to avoid relative path and local import issues:

```bash
cd /path/to/pistar/control_your_robot
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
```

### 1. Collect real-robot demo data

Use software master-slave teleoperation to collect demos:

```bash
python example/collect/collect_lerobot_master_slave_teleop.py
```

Before running, modify the configuration at the bottom of the script:

- `REPO_ID`: Output LeRobot dataset name.
- `OUTPUT_DIR`: Parent directory of the output dataset.
- `TASK_NAME`: Task text instruction.
- `MASTER_CAN` and `SLAVE_CAN`: CAN interfaces of the master arm and slave arm.
- `FPS`, `NUM_EPISODES`, reset joint positions, camera settings, and other hardware and collection parameters.

This script saves demo-style data. Because every frame is manually teleoperated, it should be treated as positive expert data later. The saved dataset can be merged with rollout data using `scripts/merge_datasets.py`.

### 2. Real-robot DAgger rollout and data collection

If you need the policy to execute autonomously while allowing human intervention and saving rollout data, use the DAgger deployment script:

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

### 3. Real-robot PiStar pure inference

If you only want to run a trained checkpoint and do not need to collect DAgger rollout data, use the single-arm inference script:

```bash
python example/deploy/piper_single_on_PI0.py \
  --model-path /path/to/checkpoint/step_dir \
  --task-name "put the white plug into the two-hole socket" \
  --train-config pi05_star_white_plug_infer \
  --max-step 160 \
  --num-episode 10 \
  --adv-ind positive
```

For a regular `pi05` checkpoint, `--adv-ind` can be omitted. For a PiStar checkpoint, you need to pass in the condition expected by the training config; common values are `positive` or `negative`.

## Questions and Feedback

If you have any questions or encounter any issues, feel free to join our WeChat group for discussion.

<img src=".\wechat_group.jpg" style="zoom:30%;" />
