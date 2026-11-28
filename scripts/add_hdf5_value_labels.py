"""Replace HDF5 value_label and reward_label using one corpus normalization."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import h5py

from openpi.training.recap_labels import REWARD_LABEL
from openpi.training.recap_labels import VALUE_LABEL
from openpi.training.recap_labels import RecapLabelScale
from openpi.training.recap_labels import add_normalization_args
from openpi.training.recap_labels import parse_success


SEGMENT_DATASETS = frozenset((
    "260728_v1", "260729_v1", "260730_v1", "260804_v1", "260805_v1", "260807_v1", "260810_v1",
))
FRAME_KEYS = (
    "action", "observations/qpos/left_arm", "observations/qpos",
    "observations/camera/cam_head", "observation/camera/head", "source_index",
)


@dataclass(frozen=True)
class Episode:
    path: Path
    length: int
    success: bool


def collect(inputs):
    files = {}
    for item in inputs:
        candidate = item.resolve(strict=True)
        if candidate.suffix == ".jsonl" and candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                path = (candidate.parent / row["episode_file"]).resolve(strict=True)
                if path in files and files[path] not in (None, row):
                    raise ValueError(f"Conflicting index records: {path}")
                files[path] = row
        elif candidate.is_dir():
            for path in candidate.rglob("*.hdf5"):
                files.setdefault(path.resolve(strict=True), None)
        elif candidate.suffix == ".hdf5":
            files.setdefault(candidate, None)
        else:
            raise ValueError(f"Expected an HDF5 file, directory or episode JSONL: {candidate}")
    if not files:
        raise ValueError("No HDF5 episodes found")
    return files


def read_episode(path, row, assume_success):
    if SEGMENT_DATASETS.intersection(path.parts):
        raise ValueError(f"{path} uses protected segment labels; use relabel_hdf5_segment_values.py")
    with h5py.File(path, "r") as h:
        if "value_label_segments_source" in h.attrs:
            raise ValueError(f"{path} uses segment labels; use relabel_hdf5_segment_values.py")
        lengths = {len(h[key]) for key in FRAME_KEYS if isinstance(h.get(key), h5py.Dataset)}
        if len(lengths) != 1:
            raise ValueError(f"Missing or inconsistent episode length: {path}: {lengths}")
        length = lengths.pop()
        if row and int(row["episode_length"]) != length:
            raise ValueError(f"{path}: indexed length does not match stored frames")
        for key in (VALUE_LABEL, REWARD_LABEL):
            item = h.get(key)
            if item is not None and not isinstance(item, h5py.Dataset):
                raise ValueError(f"{path}: {key} is not a dataset")
        status = next((h.attrs[k] for k in ("success", "if_success", "value_label_if_success") if k in h.attrs), None)
        source = h["meta"].attrs.get("camera_source_episode_dir") if "meta" in h else None
        if isinstance(source, bytes):
            source = source.decode()
    if row and "success" in row:
        indexed_success = parse_success(row["success"])
        if status is not None and parse_success(status) != indexed_success:
            raise ValueError(f"{path}: index and HDF5 outcomes disagree")
        status = indexed_success
    if status is None:
        for meta in [path.with_name("meta.json")] + ([Path(source) / "meta.json"] if source else []):
            if meta.is_file():
                fields = json.loads(meta.read_text(encoding="utf-8"))
                status = fields.get("if_success", fields.get("success"))
                if status is not None:
                    break
    if status is None and not assume_success:
        raise ValueError(f"Missing episode outcome: {path}; use --assume-success only for known demos")
    return Episode(path, length, parse_success(status) if status is not None else True)


def write_episode(episode, scale):
    values, rewards = scale.labels(episode.length, episode.success)
    with h5py.File(episode.path, "r+") as h:
        for key, values_to_write in ((VALUE_LABEL, values), (REWARD_LABEL, rewards)):
            if key in h:
                del h[key]
            h.create_dataset(key, data=values_to_write)
        h.attrs.setdefault("value_label_if_success", "success" if episode.success else "fail")
        h.attrs.update({f"value_label_{key}": value for key, value in scale.metadata().items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="HDF5 episode files, directories, or JSONL indices")
    parser.add_argument("--data_dir", "--data-dir", nargs="+", type=Path, default=[])
    add_normalization_args(parser)
    parser.add_argument("--assume-success", action="store_true", help="Treat missing outcomes as successful demos")
    parser.add_argument("--dry-run", action="store_true", help="Read and print the shared scale without writing")
    args = parser.parse_args()
    if not (args.inputs or args.data_dir):
        parser.error("Provide all input datasets together so they share one normalization")
    episodes = [read_episode(p, row, args.assume_success) for p, row in collect(args.inputs + args.data_dir).items()]
    scale = RecapLabelScale.from_episodes(((ep.length, ep.success) for ep in episodes),
                                         normalization=args.normalization, failure_cost=args.failure_cost,
                                         failure_cost_ratio=args.failure_cost_ratio)
    print(json.dumps(dict(scale.metadata(), episodes=len(episodes), frames=sum(ep.length for ep in episodes),
                          failures=sum(not ep.success for ep in episodes), dry_run=args.dry_run)), flush=True)
    if not args.dry_run:
        for episode in episodes:
            write_episode(episode, scale)


if __name__ == "__main__":
    main()
