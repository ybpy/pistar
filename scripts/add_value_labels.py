"""Recompute LeRobot value_label and reward_label with a shared corpus Tmax."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.training.recap_labels import DEFAULT_FAILURE_COST_RATIO
from openpi.training.recap_labels import REWARD_LABEL
from openpi.training.recap_labels import VALUE_LABEL
from openpi.training.recap_labels import RecapLabelScale
from openpi.training.recap_labels import parse_success

LEGACY = "value_lable"
SIGNALS = ("success", "if_success", "reward")


def scalar(value):
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def scan(roots, assume_success):
    jobs = []
    lengths = {}
    for root in roots:
        files = sorted((root / "data").rglob("*.parquet"))
        if not files:
            raise ValueError(f"No Parquet files under {root / 'data'}")
        episodes = defaultdict(list)
        for path in files:
            pf = pq.ParquetFile(path)
            names = pf.schema_arrow.names
            signal = next((col for col in SIGNALS if col in names), None)
            if signal is None and not assume_success:
                raise ValueError(f"No outcome or binary success reward in {path}; use --assume-success for known demos")
            columns = [col for col in ("episode_index", "frame_index", signal) if col in names]
            table = pf.read(columns=list(dict.fromkeys(columns))).to_pydict()
            count = pf.metadata.num_rows
            if "episode_index" not in table and not path.stem.startswith("episode_"):
                raise ValueError(f"{path}: missing episode_index in a multi-episode file")
            keys, rows = [], []
            for i in range(count):
                key = table["episode_index"][i] if "episode_index" in table else str(path)
                frame = table["frame_index"][i] if "frame_index" in table else len(episodes[key])
                observed = None if signal is None else parse_success(scalar(table[signal][i]))
                episodes[key].append((frame, observed, signal))
                keys.append(key)
                rows.append(frame)
            jobs.append((root, path, keys, np.asarray(rows, dtype=np.int64)))
        for key, entries in episodes.items():
            entries.sort(key=lambda record: record[0])
            n = len(entries)
            if any(pos != entry[0] for pos, entry in enumerate(entries)):
                raise ValueError(f"{root}: episode {key} must contain each frame index from 0 exactly once")
            explicit = {success for _, success, sig in entries if sig not in (None, "reward")}
            if len(explicit) > 1:
                raise ValueError(f"{root}: conflicting success flags for episode {key}")
            reward = [success for _, success, sig in entries if sig == "reward"]
            if reward and len(reward) != n:
                raise ValueError(f"{root}: inconsistent outcome columns for episode {key}")
            if reward and any(reward[:-1]):
                raise ValueError(f"{root}: episode {key} has non-terminal sparse success reward")
            success = explicit.pop() if explicit else (reward[-1] if reward else True)
            lengths[root, key] = (n, success)
    return jobs, lengths


def rewrite(path, value, reward):
    table = pq.read_table(path)
    if LEGACY in table.column_names:
        table = table.drop([LEGACY])
    for key, data in ((VALUE_LABEL, value), (REWARD_LABEL, reward)):
        idx = table.schema.get_field_index(key)
        typ = table.schema.field(idx).type if idx >= 0 else pa.float32()
        if pa.types.is_list(typ) or pa.types.is_large_list(typ) or pa.types.is_fixed_size_list(typ):
            if pa.types.is_fixed_size_list(typ) and typ.list_size != 1:
                raise ValueError(f"Unexpected {key} shape in {path}: {typ}")
            arr = pa.array(data.reshape(-1, 1).tolist(), type=typ)
        else:
            arr = pa.array(data, type=pa.float32())
        table = table.append_column(key, arr) if idx < 0 else table.set_column(idx, key, arr)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        pq.write_table(table, tmp)
        tmp.chmod(path.stat().st_mode & 0o777)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", "--data-dir", required=True, nargs="+", type=Path)
    parser.add_argument("--failure-cost-ratio", type=float, default=DEFAULT_FAILURE_COST_RATIO)
    parser.add_argument("--assume-success", action="store_true", help="Treat missing outcomes as successful demos")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    roots = tuple(dict.fromkeys(p.resolve(strict=True) for p in args.data_dir))
    jobs, episodes = scan(roots, args.assume_success)
    scale = RecapLabelScale.from_lengths((length for length, _ in episodes.values()), args.failure_cost_ratio)
    print(json.dumps(dict(scale.metadata(), episodes=len(episodes), files=len(jobs),
                          frames=sum(n for n, _ in episodes.values()), dry_run=args.dry_run)), flush=True)
    if args.dry_run:
        return
    for root, path, keys, frames in jobs:
        values, rewards = np.empty(len(keys), np.float32), np.empty(len(keys), np.float32)
        positions = defaultdict(list)
        for i, key in enumerate(keys):
            positions[key].append(i)
        for key, pos in positions.items():
            n, success = episodes[root, key]
            values[pos], rewards[pos] = scale.labels_at(n, success, frames[pos])
        rewrite(path, values, rewards)
    for root in roots:
        info_path = root / "meta" / "info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            features = info.setdefault("features", {})
            features.pop(LEGACY, None)
            for key in (VALUE_LABEL, REWARD_LABEL):
                features[key] = dict(features.get(key, {}), dtype="float32", shape=[1])
            info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        meta_dir = root / "meta"
        meta_dir.mkdir(exist_ok=True)
        (meta_dir / "recap_value_labels.json").write_text(json.dumps(scale.metadata(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
