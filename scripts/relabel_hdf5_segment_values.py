"""Relabel the seven early success datasets using their episode.json segments.

Each package and explicit return-home subtask has its own remaining-fraction
value: -(end - frame) / (end - start), with the tail exactly 0. A one-frame subtask is 0. Run without --apply for an audit.
All original labels are backed up before any source dataset is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path('/ML-vePFS/shared/world_action_model_data/self_produced/agilex_project/sort_product')
DATASETS = ('260728_v1', '260729_v1', '260730_v1', '260804_v1', '260805_v1', '260807_v1', '260810_v1')
SCHEMA = 'episode_json_subtask_endpoint_zero_v1'


def labels_from_segments(length: int, segments: list[dict]) -> np.ndarray:
    if length < 1 or not segments:
        raise ValueError('A nonempty episode and segments are required')
    labels = np.empty(length, dtype=np.float32)
    next_start = 0
    packages = homes = 0
    for segment in segments:
        start, end = segment['start_idx'], segment['end_idx']
        if type(start) is not int or type(end) is not int:
            raise ValueError('Segment boundaries must be integers')
        if start != next_start or end < start or end >= length:
            raise ValueError(f'Invalid segment {start}:{end}; expected start {next_start}, frames={length}')
        barcode = segment.get('barcode_idx')
        if segment.get('barcode_value') is not None:
            if type(barcode) is not int or not start <= barcode <= end:
                raise ValueError('Package barcode frame must lie inside its segment')
            packages += 1
        elif any(word in segment.get('subtask_text', '').lower() for word in ('home', 'reset')):
            homes += 1
        else:
            raise ValueError('Unknown non-package subtask')
        status = segment.get('if_success', 'success')
        if status not in ('success', 'take_over'):
            raise ValueError(f'Unexpected segment outcome in early success data: {status}')
        count = end - start + 1
        labels[start:end + 1] = np.linspace(-1.0, 0.0, count, dtype=np.float32) if count > 1 else 0.0
        next_start = end + 1
    if next_start != length:
        raise ValueError('Segments do not cover the full episode')
    if packages != 5 or homes != 1:
        raise ValueError(f'Expected 5 packages + 1 home, got {packages} + {homes}')
    return labels


def digest(values: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(values.dtype.str.encode()); h.update(str(values.shape).encode())
    h.update(values.tobytes())
    return h.hexdigest()


def json_value(value):
    if isinstance(value, bytes): return value.decode()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def episode_files(root: Path):
    if not root.is_dir(): raise FileNotFoundError(root)
    for first in sorted(p for p in root.iterdir() if p.is_dir()):
        direct = first / 'episode.hdf5'
        if direct.is_file():
            yield direct
        else:
            for second in sorted(p for p in first.iterdir() if p.is_dir()):
                if (second / 'episode.hdf5').is_file(): yield second / 'episode.hdf5'


def describe(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    path.resolve().relative_to(ROOT.resolve())
    annotation = Path(str(path.parent).replace('/self_produced/', '/self_produced_label/', 1)) / 'episode.json'
    source = annotation.read_bytes()
    document = json.loads(source)
    referenced = document.get('episode_file')
    if referenced and Path(referenced).resolve() != path.resolve():
        raise ValueError('Annotation references a different episode')
    with h5py.File(path, 'r') as f:
        length = len(f['action']) if 'action' in f else len(f['observation/camera/head'])
        if len(f['observation/camera/head']) != length:
            raise ValueError('Camera and action frame counts differ')
        old = np.asarray(f['value_label'])
        if old.shape != (length,) or old.dtype != np.dtype('float32') or not np.isfinite(old).all():
            raise ValueError('Unexpected existing label shape, dtype, or nonfinite values')
        attrs = {k: json_value(v) for k, v in f.attrs.items() if k.startswith('value_label')}
        if attrs.get('value_label_if_success') != 'success':
            raise ValueError('Only the early success episodes can be relabeled by this script')
    new = labels_from_segments(length, document['segments'])
    record = {'path': str(path), 'annotation': str(annotation), 'frames': length,
              'segments': document['segments'], 'old_label_sha256': digest(old),
              'new_label_sha256': digest(new), 'annotation_sha256': hashlib.sha256(source).hexdigest(),
              'old_value_label_attributes': attrs,
              'mtime_ns': path.stat().st_mtime_ns, 'size_bytes': path.stat().st_size}
    return record, old, new


def apply_record(record: dict, backup_dir: Path, created_at: str) -> None:
    path = Path(record['path'])
    relative = path.relative_to(ROOT)
    backup = backup_dir / relative.with_suffix('.npz')
    with np.load(backup, allow_pickle=False) as saved:
        original = saved['value_label']
        if digest(original) != record['old_label_sha256']:
            raise ValueError(f'Backup integrity mismatch: {path}')
    current, old, new = describe(path)
    for key in ('mtime_ns', 'size_bytes', 'old_label_sha256', 'annotation_sha256', 'new_label_sha256'):
        if current[key] != record[key]: raise RuntimeError(f'Concurrent modification: {path} ({key})')
    with h5py.File(path, 'r+') as f:
        if digest(np.asarray(f['value_label'])) != record['old_label_sha256']:
            raise RuntimeError(f'Labels changed before write: {path}')
        f['value_label'][...] = new
        f.attrs['value_label_schema'] = SCHEMA
        f.attrs['value_label_segments_source'] = record['annotation']
        f.attrs['value_label_segments_sha256'] = record['annotation_sha256']
        f.attrs['value_label_segment_count'] = len(record['segments'])
        f.attrs['value_label_home_policy'] = 'independent_success_subtask'
        f.attrs['value_label_status_source'] = 'existing_episode_success; explicit segment status if present'
        f.attrs['value_label_relabelled_at'] = created_at
        f.flush()
    with h5py.File(path, 'r') as f:
        if not np.array_equal(np.asarray(f['value_label']), new):
            raise RuntimeError(f'Post-write verification failed: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError('Use a new report/backup directory')
    args.output_dir.mkdir(parents=True)
    records, errors, counts = [], [], Counter()
    for dataset in DATASETS:
        for path in episode_files(ROOT / dataset):
            try:
                record, _, _ = describe(path)
                records.append(record); counts[dataset] += 1
            except Exception as exc:
                errors.append({'path': str(path), 'error': str(exc)})
        print('PREFLIGHT', dataset, counts[dataset], flush=True)
    created_at = datetime.now(timezone.utc).isoformat()
    report = {'schema': SCHEMA, 'created_at_utc': created_at, 'apply_requested': args.apply,
              'datasets': dict(counts), 'episodes': len(records), 'frames': sum(r['frames'] for r in records),
              'package_segments': len(records) * 5, 'home_segments': len(records),
              'home_policy': 'independent_success_subtask', 'errors': errors, 'records': records}
    (args.output_dir / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    if errors or any(not counts[name] for name in DATASETS):
        raise RuntimeError(f'Preflight failed: {len(errors)} bad episodes or empty dataset; no data modified')
    if not args.apply:
        print('AUDIT_ONLY', len(records), flush=True); return
    backups = args.output_dir / 'original_labels'
    for i, record in enumerate(records, 1):
        current, old, _ = describe(Path(record['path']))
        if current != record: raise RuntimeError('Concurrent modification before backup')
        backup = backups / Path(record['path']).relative_to(ROOT).with_suffix('.npz')
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open('xb') as f:
            np.savez_compressed(f, value_label=old,
                               original_attributes_json=np.asarray(json.dumps(record['old_value_label_attributes'])))
        with np.load(backup, allow_pickle=False) as saved:
            if not np.array_equal(saved['value_label'], old): raise RuntimeError('Backup verification failed')
        if i % 200 == 0: print('BACKUP', i, '/', len(records), flush=True)
    print('ALL_BACKUPS_VERIFIED', len(records), flush=True)
    with (args.output_dir / 'applied.jsonl').open('x') as log:
        for i, record in enumerate(records, 1):
            apply_record(record, backups, created_at)
            log.write(json.dumps({'path': record['path'], 'new_label_sha256': record['new_label_sha256']}) + '\n')
            log.flush()
            if i % 100 == 0: print('APPLIED_AND_VERIFIED', i, '/', len(records), flush=True)
    result = {k: v for k, v in report.items() if k != 'records'}
    result.update(applied_episodes=len(records), backup_directory=str(backups))
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()