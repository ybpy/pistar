from types import SimpleNamespace

import numpy as np
import pytest

from openpi.policies import policy_config
import openpi.transforms as transforms


class _RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, state, adv_ind, *, adv_ind_dropout):
        self.calls.append((prompt, state, adv_ind, adv_ind_dropout))
        return np.array([1 if adv_ind is None else 2]), np.array([True])


@pytest.mark.parametrize(
    ("config_beta", "sample_kwargs", "pistar", "expected_guidance"),
    [
        (2.0, None, True, True),
        (1.0, None, True, False),
        (2.0, {"adv_guidance_beta": 1.0}, True, False),
        (1.0, {"adv_guidance_beta": 3.0}, True, True),
        (2.0, {"adv_guidance_beta": 0.0}, True, True),
        (2.0, {"adv_guidance_beta": None}, True, True),
        (2.0, None, False, False),
    ],
)
def test_policy_prepares_advantage_guidance(
    monkeypatch, tmp_path, config_beta, sample_kwargs, pistar, expected_guidance
):
    tokenizer = _RecordingTokenizer()
    training_transform = transforms.TokenizePrompt(
        tokenizer, discrete_state_input=True, adv_ind_input=pistar, adv_ind_dropout=True
    )
    other_transform = transforms.InjectDefaultPrompt("fallback task")
    data_config = SimpleNamespace(
        data_transforms=transforms.Group(),
        model_transforms=transforms.Group(inputs=[other_transform, training_transform]),
        use_quantile_norm=False,
    )
    model = object()
    config = SimpleNamespace(
        model=SimpleNamespace(adv_guidance_beta=config_beta, load=lambda params: model),
        data=SimpleNamespace(create=lambda *args: data_config),
        assets_dirs=tmp_path,
        policy_metadata={},
    )
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda path: tmp_path)
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        policy_config._policy, "Policy", lambda loaded_model, **kwargs: SimpleNamespace(model=loaded_model, **kwargs)
    )

    policy = policy_config.create_trained_policy(config, tmp_path, norm_stats={}, sample_kwargs=sample_kwargs)
    inference_transform = policy.transforms[-1]
    assert policy.model is model
    assert policy.sample_kwargs == sample_kwargs
    assert policy.transforms[-2] is other_transform
    assert training_transform.adv_ind_dropout is True
    assert training_transform.adv_guidance_input is False
    assert inference_transform.adv_guidance_input is expected_guidance
    if pistar:
        assert inference_transform.adv_ind_dropout is False
    else:
        assert inference_transform is training_transform

    state = np.array([0.1, -0.2])
    result = inference_transform({"prompt": "insert plug", "state": state, "adv_ind": "positive"})
    assert "adv_ind" not in result
    assert ("tokenized_prompt_uncond" in result) is expected_guidance
    assert ("tokenized_prompt_uncond_mask" in result) is expected_guidance
    assert tokenizer.calls[0][0] == "insert plug"
    assert tokenizer.calls[0][1] is state
    assert tokenizer.calls[0][2] == ("positive" if pistar else None)
    if pistar:
        assert tokenizer.calls[0][3] is False
    if expected_guidance:
        assert len(tokenizer.calls) == 2
        assert tokenizer.calls[1][0] == "insert plug"
        assert tokenizer.calls[1][1] is state
        assert tokenizer.calls[1][2:] == (None, False)
        np.testing.assert_array_equal(result["tokenized_prompt"], [2])
        np.testing.assert_array_equal(result["tokenized_prompt_uncond"], [1])
    else:
        assert len(tokenizer.calls) == 1
