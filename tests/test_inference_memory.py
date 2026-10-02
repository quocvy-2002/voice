"""Inference must not retain unused reference graphs or diffusion KV caches."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

from omnivoice import OmniVoice


def test_reference_encoding_disables_autograd():
    modes = []

    def encode(waveform):
        modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        return SimpleNamespace(audio_codes=torch.zeros((1, 8, 5), dtype=torch.long))

    model = SimpleNamespace(
        sampling_rate=16000,
        audio_tokenizer=SimpleNamespace(
            device=torch.device("cpu"),
            config=SimpleNamespace(hop_length=320),
            encode=encode,
        ),
    )
    with torch.enable_grad():
        prompt = OmniVoice.create_voice_clone_prompt(
            model,
            (np.full((1, 1600), 0.2, dtype=np.float32), 16000),
            "A reference.",
            preprocess_prompt=False,
        )
    assert prompt.ref_audio_tokens.shape == (8, 5)
    assert modes == [(False, True)]


@pytest.mark.parametrize("with_labels", [False, True])
def test_forward_omits_cache_without_changing_logits_or_training_loss(with_labels):
    torch.manual_seed(7)
    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
    )
    config._attn_implementation = "sdpa"
    backbone = Qwen3Model(config).eval()
    embeds = torch.randn(2, 17, 64)
    mask = torch.ones((2, 1, 17, 17), dtype=torch.bool)
    heads = torch.nn.Linear(64, 2 * 9, bias=False)
    outputs = []

    def llm(**kwargs):
        output = backbone(**kwargs)
        outputs.append(output)
        return output

    model = SimpleNamespace(
        _prepare_embed_inputs=lambda *_args: embeds,
        llm=llm,
        audio_heads=heads,
        config=SimpleNamespace(num_audio_codebook=2, audio_vocab_size=9),
        normalized_audio_codebook_weights=[0.5, 0.5],
    )
    kwargs = dict(
        input_ids=torch.zeros((2, 2, 17), dtype=torch.long),
        audio_mask=torch.ones((2, 17), dtype=torch.bool),
        attention_mask=mask,
        labels=torch.zeros((2, 2, 17), dtype=torch.long) if with_labels else None,
    )
    actual = OmniVoice.forward(model, **kwargs)
    assert outputs[0].past_key_values is None

    def cached_llm(**kw):
        kw["use_cache"] = True
        return backbone(**kw)

    model.llm = cached_llm
    baseline = OmniVoice.forward(model, **kwargs)
    assert torch.equal(actual.logits, baseline.logits)
    if with_labels:
        assert torch.equal(actual.loss, baseline.loss)
        actual.loss.backward()
        assert heads.weight.grad is not None
