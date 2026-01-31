# Usage of PiStar

## Environment Setup

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

## Fine-Tuning

Before executing following command, make sure the pistar venv have been activated. 

### 1. Convert your data to a LeRobot dataset

```bash
python examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

### 2. Compute the normalization statistics for the training data

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/compute_norm_stats.py --config-name pi05_star_libero
```

### 3. Run training

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_star_libero --exp-name=my_experiment --overwrite
```

You can use `--resume` to replace `--overwrite` in above command to restore latest checkpoint for continuation training. 

## Evaluation

### 1. Client

```bash
# Create virtual environment
uv venv --python 3.8 /path/to/create/libero/venv
source /path/to/your/libero/venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
# To insure installing these package in your libero venv, you can add --python /path/to/your/libero/venv/bin/python behind above command, like uv pip install -e third_party/libero --python /path/to/your/libero/venv/bin/python
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py --args.adv_ind_input positive
```

### 2. Server

```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_star_libero_infer --policy.dir=checkpoints/pi05_star_libero/my_experiment/10000
```

