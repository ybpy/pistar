"""Measure host loader throughput without starting model training."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import ast
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.train_value import build_value_data_config
from openpi.models.value_model_config import ValueModelConfig
from openpi.training.agilex_hdf5_value_dataset import AgilexHdf5ValueDataset
from openpi.training.data_loader import create_value_data_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=0)
    args = parser.parse_args()
    old_log = (ROOT / "wandb/offline-run-20261011_052504-srsp424q/logs/debug.log").read_text()
    cfg = ast.literal_eval(next(line[8:] for line in old_log.splitlines() if line.startswith("config: ")))
    split = json.loads((Path(cfg["checkpoint_dir"]) / "validation_split.json").read_text())
    paths = split["train_episodes"]
    sample = [str(Path(p).parent) for p in paths[::max(1, len(paths) // 128)][:128]]
    dataset_kwargs = {}
    if "camera_names" in inspect.signature(AgilexHdf5ValueDataset).parameters:
        dataset_kwargs["camera_names"] = ("head", "left")
    dataset = AgilexHdf5ValueDataset(sample, default_prompt=cfg["default_prompt"], **dataset_kwargs)
    model_config = ValueModelConfig()
    data_cfg = build_value_data_config(sample, model_config, tokenizer_path=cfg["tokenizer_path"],
                                       default_prompt=cfg["default_prompt"])
    kwargs = dict(raw_dataset=dataset, batch_size=args.batch_size, num_workers=args.workers,
                  shuffle=True, seed=42, framework="pytorch", skip_norm_stats=True)
    if "episode_block_size" in inspect.signature(create_value_data_loader).parameters:
        kwargs["episode_block_size"] = args.block_size
    loader = create_value_data_loader(data_cfg, model_config, **kwargs)
    raw_loader = loader._data_loader.torch_loader
    start = time.perf_counter()
    iterator = iter(raw_loader)
    for _ in range(4): next(iterator)
    warmup_seconds = time.perf_counter() - start
    waits = []
    for _ in range(args.batches):
        start = time.perf_counter()
        batch = next(iterator)
        waits.append(time.perf_counter() - start)
    result = {"batch_size": args.batch_size, "workers": args.workers, "block_size": args.block_size,
              "measured_batches": len(waits), "warmup_seconds": warmup_seconds,
              "mean_batch_seconds": float(np.mean(waits)), "p95_batch_seconds": float(np.percentile(waits, 95)),
              "frames_per_second": args.batch_size / float(np.mean(waits))}
    print("BENCHMARK_RESULT", json.dumps(result), flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    iterator._shutdown_workers()


if __name__ == "__main__":
    main()
