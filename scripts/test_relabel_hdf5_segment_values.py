"""Checks for inclusive segment bounds and preservation of unrelated HDF5 data."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import h5py
import numpy as np
import pytest
import relabel_hdf5_segment_values as relabel


def make_segments():
    segments = [{'start_idx':2*i,'end_idx':2*i+1,'barcode_idx':2*i,'barcode_value':'item-01-001','subtask_text':''} for i in range(5)]
    return segments + [{'start_idx':10,'end_idx':11,'barcode_idx':None,'barcode_value':None,'subtask_text':'reset arms to home'}]


def test_segments_restart_and_include_last_frame():
    labels = relabel.labels_from_segments(12, make_segments())
    np.testing.assert_array_equal(labels, np.tile([-1., 0.], 6).astype(np.float32))


@pytest.mark.parametrize('case', ['gap','overlap','out_of_range','uncovered','bad_barcode','unknown_subtask'])
def test_invalid_boundaries_are_rejected(case):
    segments = make_segments()
    if case == 'gap': segments[1]['start_idx'] = 3
    if case == 'overlap': segments[1]['start_idx'] = 1
    if case == 'out_of_range': segments[-1]['end_idx'] = 12
    if case == 'uncovered': segments[-1]['end_idx'] = 10
    if case == 'bad_barcode': segments[0]['barcode_idx'] = 2
    if case == 'unknown_subtask': segments[-1]['subtask_text'] = ''
    with pytest.raises(ValueError): relabel.labels_from_segments(12, segments)


def test_backup_and_write_leave_images_actions_and_status_unchanged(tmp_path):
    root = tmp_path/'self_produced'/'sort_product'
    path = root/'260728_v1'/'episode_0'/'episode.hdf5'
    path.parent.mkdir(parents=True)
    annotation = Path(str(path.parent).replace('/self_produced/','/self_produced_label/'))/'episode.json'
    annotation.parent.mkdir(parents=True)
    annotation.write_text(json.dumps({'episode_file':str(path),'segments':make_segments()}))
    original = -(12-np.arange(12,dtype=np.float32))/12
    with h5py.File(path,'w') as f:
        f['value_label'] = original
        f['action'] = np.arange(24).reshape(12,2)
        f['observation/camera/head'] = np.arange(12)
        f.attrs['value_label_if_success'] = 'success'
        f.attrs['other_attribute'] = 'preserve'
    with patch.object(relabel,'ROOT',root):
        record,old,new = relabel.describe(path)
        backup = tmp_path/'backups'/path.relative_to(root).with_suffix('.npz')
        backup.parent.mkdir(parents=True)
        np.savez_compressed(backup,value_label=old)
        relabel.apply_record(record,tmp_path/'backups','test-time')
        with h5py.File(path,'r') as f:
            np.testing.assert_array_equal(f['value_label'][:],new)
            np.testing.assert_array_equal(f['action'][:],np.arange(24).reshape(12,2))
            np.testing.assert_array_equal(f['observation/camera/head'][:],np.arange(12))
            assert f.attrs['value_label_if_success']=='success'
            assert f.attrs['other_attribute']=='preserve'
        np.testing.assert_array_equal(np.load(backup)['value_label'],original)

def test_single_frame_subtask_has_terminal_zero():
    segments = make_segments()
    segments[-1]['end_idx'] = 10
    labels = relabel.labels_from_segments(11, segments)
    assert labels[-1] == 0.0
    for segment in segments:
        assert labels[segment['end_idx']] == 0.0