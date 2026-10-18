"""Regression checks for value-model dropout and episode validation splits."""
import json

import flax.nnx as nnx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.value_model import ValueHead
from openpi.models.value_model_config import ValueModelConfig
from openpi.training.agilex_hdf5_value_dataset import AgilexHdf5ValueDataset
from scripts.train_value import _make_trainable_filter


def test_dropout_is_reproducible_stochastic_and_disabled_for_validation():
    head = ValueHead(16, 32, 5, 0.1, nnx.Rngs(0))
    x = jnp.arange(64 * 16, dtype=jnp.float32).reshape(64, 16)
    graph, state = nnx.split(head)

    # train is static under JIT, as in train/eval model calls.
    train_forward = jax.jit(lambda s, k: nnx.merge(graph, s)(x, train=True, rng=k))
    eval_forward = jax.jit(lambda s, k: nnx.merge(graph, s)(x, train=False, rng=k))
    a = train_forward(state, jax.random.key(1))
    b = train_forward(state, jax.random.key(1))
    c = train_forward(state, jax.random.key(2))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    np.testing.assert_array_equal(eval_forward(state, jax.random.key(1)),
                                  eval_forward(state, jax.random.key(2)))
    loss, grads = nnx.value_and_grad(lambda h: jnp.mean(h(x, train=True, rng=jax.random.key(3)) ** 2))(head)
    assert np.isfinite(loss)
    assert all(np.isfinite(v.value).all() for v in grads.flat_state().values())
    with pytest.raises(ValueError, match="explicit RNG"):
        head(x, train=True)


def make_episodes(root, statuses):
    for i, status in enumerate(statuses):
        folder = root / f"episode_{i:03d}"
        folder.mkdir(parents=True)
        (folder / "meta.json").write_text(json.dumps({"if_success": status}))
        with h5py.File(folder / "episode.hdf5", "w") as f:
            f.create_dataset("action", shape=(i + 3, 1))
            f.create_dataset("value_label", data=np.linspace(-1, 0, i + 3))
            f.attrs["value_label_if_success"] = status
    return AgilexHdf5ValueDataset(str(root))


def paths(dataset):
    return {ep["hdf5_path"] for ep in dataset.episodes}


def test_episode_split_has_no_leakage_is_reproducible_and_stratified(tmp_path):
    full = make_episodes(tmp_path, ["success"] * 10 + ["take_over"] * 10 + ["fail"] * 10)
    train, val = full.split_by_episode(0.1, 42)
    train2, val2 = full.split_by_episode(0.1, 42)
    assert paths(train).isdisjoint(paths(val))
    assert paths(train) | paths(val) == paths(full)
    assert len(train) + len(val) == len(full)
    assert paths(train) == paths(train2) and paths(val) == paths(val2)
    assert len(train.episodes) == 27 and len(val.episodes) == 3
    assert {ep["if_success"] for ep in val.episodes} == {"success", "take_over", "fail"}
    assert paths(val) != paths(full.split_by_episode(0.1, 43)[1])
    for subset in (train, val):
        for i, ep in enumerate(subset.episodes):
            assert subset._locate(subset.episode_offsets[i]) == (i, 0)
            assert subset._locate(subset.episode_offsets[i + 1] - 1) == (i, ep["length"] - 1)
    assert len(full.episodes) == 30


def test_small_split_remains_nonempty_and_rejects_invalid_fraction(tmp_path):
    full = make_episodes(tmp_path, ["success", "fail"])
    train, val = full.split_by_episode(0.1)
    assert len(train.episodes) == len(val.episodes) == 1
    for fraction in (0, 1, -0.1):
        with pytest.raises(ValueError):
            full.split_by_episode(fraction)


def test_full_parameter_filter_includes_embedding_and_preserves_freeze_options():
    state = nnx.State({"img": {"kernel": nnx.Param(jnp.ones(1))},
                       "llm": {"embedder": {"input_embedding": nnx.Param(jnp.ones(1))}},
                       "value_head": {"kernel": nnx.Param(jnp.ones(1))}})
    assert len(state.filter(_make_trainable_filter("none")).flat_state()) == 3
    assert len(state.filter(_make_trainable_filter("siglip_only")).flat_state()) == 2
    assert len(state.filter(_make_trainable_filter("all_backbones")).flat_state()) == 1
    assert ValueModelConfig().dropout_rate == 0.1
    with pytest.raises(ValueError):
        ValueModelConfig(dropout_rate=1.0)


def test_cli_defaults_enable_requested_training_settings(monkeypatch):
    import sys
    import scripts.train_value as train
    import argparse
    original_parse = argparse.ArgumentParser.parse_args
    captured = {}

    class ParsedDefaults(Exception):
        pass

    def capture(parser, *args, **kwargs):
        parsed = original_parse(parser, *args, **kwargs)
        captured.update(vars(parsed))
        raise ParsedDefaults

    monkeypatch.setattr(sys, "argv", ["train_value.py", "--data_dir", "/data", "--checkpoint_dir", "/checkpoints"])
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(ParsedDefaults):
        train.main()
    assert captured["freeze_mode"] == "none"
    assert captured["dropout_rate"] == 0.1
    assert captured["val_fraction"] == 0.1
    assert captured["val_interval"] == 1000
    assert captured["peak_lr"] == 5e-6
    assert captured["decay_lr"] == 5e-7


def test_value_tokenizer_matches_shared_prompt_transform_interface():
    from scripts.train_value import GemmaValueTokenizer
    from openpi.transforms import TokenizePrompt

    class StubTokenizer:
        def encode(self, text, **kwargs):
            return [2, 10, 20]

    tokenizer = GemmaValueTokenizer(max_len=8)
    tokenizer._tokenizer = StubTokenizer()
    result = TokenizePrompt(tokenizer)({"prompt": "Pick the package"})
    assert result["tokenized_prompt"].shape == (8,)
    assert result["tokenized_prompt_mask"].sum() == 3


@pytest.mark.parametrize("barcode", ["ITEM-01y005", "ITEM- 25;;=", "ITEMt02-003", "ITEM-01u; 2", "ITEM-01%003"])
def test_invalid_barcode_episode_is_excluded_before_worker_iteration(tmp_path, barcode):
    make_episodes(tmp_path, ["success", "fail", "take_over"])
    path = tmp_path / "episode_001" / "episode.hdf5"
    with h5py.File(path, "a") as f:
        f.create_dataset("extra_info/bar_code/0", data=barcode)
    dataset = AgilexHdf5ValueDataset(str(tmp_path))
    assert len(dataset.episodes) == 2
    assert dataset.excluded_prompt_episodes[0]["path"] == str(path.resolve())
    assert all(ep["hdf5_path"] != path for ep in dataset.episodes)
    assert path.exists()


def test_existing_validation_membership_is_preserved_after_exclusion(tmp_path):
    full = make_episodes(tmp_path, ["success"] * 10 + ["fail"] * 10)
    train, val = full.split_by_episode(0.1, 42)
    invalid_train = train.episodes[0]["hdf5_path"]
    invalid_val = val.episodes[0]["hdf5_path"]
    for path in (invalid_train, invalid_val):
        with h5py.File(path, "a") as f:
            f.create_dataset("extra_info/bar_code/0", data="ITEM-01u; 2")
    cleaned = AgilexHdf5ValueDataset(str(tmp_path))
    new_train, new_val = cleaned.split_by_episode(0.1, 42, [str(p.resolve()) for p in paths(val)])
    assert paths(new_train) == paths(train) - {invalid_train}
    assert paths(new_val) == paths(val) - {invalid_val}
    assert paths(new_train).isdisjoint(paths(new_val))


def test_valid_barcodes_and_segment_prompt_transitions_remain_supported():
    from openpi.training.agilex_hdf5_value_dataset import PI05_PROMPT, prompt_for_frame, validate_episode_prompts
    validate_episode_prompts(PI05_PROMPT, {2: "ITEM-01-003"}, None, 10)
    assert "red basket-01" in prompt_for_frame(PI05_PROMPT, {2: "ITEM-01-003"}, 3, None)
    assert "blue basket-02" in prompt_for_frame(PI05_PROMPT, {2: "ITEM-02-003"}, 3, None)
    segments = [{"start_idx": 0, "end_idx": 9, "barcode_idx": 2, "barcode_value": "ITEM-01u; 2"}]
    with pytest.raises(ValueError, match="mapping"):
        validate_episode_prompts(PI05_PROMPT, {}, segments, 10)
