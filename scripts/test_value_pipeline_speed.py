"""Correctness checks for batched HDF5 loading and continuous prefetch."""
import base64
import threading
import cv2
import h5py
import numpy as np
import pytest
from openpi.training.agilex_hdf5_value_dataset import AgilexHdf5ValueDataset, decode_hdf5_camera_frame
from openpi.training.data_loader import EpisodeBlockSampler, TransformedDataset
from openpi.training.prefetch import PrefetchIterator
from scripts.train_value import GemmaValueTokenizer


def image_episodes(root):
    for episode, length in enumerate((3, 4)):
        folder = root / f"episode_{episode:03d}";folder.mkdir()
        with h5py.File(folder / "episode.hdf5", "w") as f:
            f.create_dataset("action", shape=(length, 1))
            f.create_dataset("value_label", data=np.linspace(-1, 0, length, dtype=np.float32))
            for camera, offset in (("head", 20), ("left", 80), ("right", 140)):
                frames = []
                for frame in range(length):
                    image = np.full((8, 12, 3), offset + episode * 10 + frame, dtype=np.uint8)
                    ok, jpeg = cv2.imencode(".jpg", image);assert ok
                    frames.append(base64.b64encode(jpeg.tobytes()).decode())
                f.create_dataset(f"observation/camera/{camera}", data=frames, dtype=h5py.string_dtype())
    return AgilexHdf5ValueDataset(str(root), camera_names=("head", "left"))


def test_batched_reads_preserve_order_images_and_labels_and_open_each_episode_once(tmp_path, monkeypatch):
    dataset = image_episodes(tmp_path)
    indices = [0, 1, 3, 2, 4, 3, 0]
    expected = []
    original_open = h5py.File
    for index in indices:
        episode, frame = (0, index) if index < 3 else (1, index - 3)
        ep = dataset.episodes[episode]
        with original_open(ep["hdf5_path"], "r") as f:
            expected.append((decode_hdf5_camera_frame(f, "head", frame),
                             decode_hdf5_camera_frame(f, "left", frame), ep["value_labels"][frame]))
    opened = []
    def tracking_open(path, *args, **kwargs):
        opened.append(str(path));return original_open(path, *args, **kwargs)
    monkeypatch.setattr(h5py, "File", tracking_open)
    batch = dataset.__getitems__(indices)
    assert len(opened) == 2
    for sample, (head, left, value) in zip(batch, expected, strict=True):
        np.testing.assert_array_equal(sample["observation.images.cam_high"], head)
        np.testing.assert_array_equal(sample["observation.images.cam_left_wrist"], left)
        assert sample["value_label"] == value
        assert "observation.images.cam_right_wrist" not in sample
    for index in (-1, len(dataset)):
        with pytest.raises(IndexError):dataset[index]


def test_block_sampler_covers_every_frame_and_reproducibly_changes_epoch():
    sampler = EpisodeBlockSampler([0, 3, 25, 29], block_size=4, seed=42)
    first = list(sampler);second = list(sampler)
    assert sorted(first) == sorted(second) == list(range(29))
    assert first != second
    assert first == list(EpisodeBlockSampler([0, 3, 25, 29], block_size=4, seed=42))
    assert len(sampler) == 29
    for start, length in sampler.blocks:
        assert any(begin <= start < start + length <= end for begin, end in ((0, 3), (3, 25), (25, 29)))


def test_transformed_dataset_retains_batched_fetch():
    class Raw:
        def __len__(self):return 5
        def __getitems__(self, indices):return [{"value": i} for i in indices]
        def __getitem__(self, index):raise AssertionError("scalar fetch should not be used")
    transformed = TransformedDataset(Raw(), [lambda d: {"value": d["value"] * 2}])
    assert transformed.__getitems__([3, 1]) == [{"value": 6}, {"value": 2}]


def test_prefetch_prepares_ahead_and_cancels_full_queue():
    ahead = threading.Event();closed = threading.Event()
    def source():
        try:
            yield 0
            ahead.set()
            yield 1
            yield 2
        finally:closed.set()
    iterator = PrefetchIterator(source(), depth=1)
    assert ahead.wait(timeout=2)
    assert next(iterator) == 0
    iterator.close()
    assert closed.wait(timeout=2)
    assert not iterator._thread.is_alive()


def test_prefetch_preserves_order_and_propagates_source_errors():
    assert list(PrefetchIterator(range(10), depth=2)) == list(range(10))
    def broken():
        yield 1
        raise ValueError("bad source batch")
    iterator = PrefetchIterator(broken(), depth=2)
    assert next(iterator) == 1
    with pytest.raises(ValueError, match="bad source batch"):next(iterator)


def test_tokenizer_cache_uses_cpu_arrays_and_accepts_legacy_pickled_objects():
    class Stub:
        def __init__(self):self.calls = 0
        def encode(self, text, **kwargs):self.calls += 1;return [2, 10, 20]
    tokenizer = GemmaValueTokenizer(max_len=8);tokenizer._tokenizer = Stub()
    first, mask = tokenizer.tokenize("Pick")
    assert isinstance(first, np.ndarray) and mask.dtype == np.bool_
    first[0] = 999
    second, _ = tokenizer.tokenize("Pick")
    assert second[0] == 2 and tokenizer._tokenizer.calls == 1
    del tokenizer._token_cache
    legacy, _ = tokenizer.tokenize("Pick")
    assert legacy[0] == 2 and tokenizer._tokenizer.calls == 2
