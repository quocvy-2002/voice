from types import SimpleNamespace

import numpy as np
import pytest

from omnivoice.models import omnivoice as model_module


def test_long_reference_audio_with_manual_transcript_is_rejected_before_encoding(monkeypatch):
    sample_rate = 24_000
    audio = np.zeros((1, sample_rate * 21), dtype=np.float32)
    tokenizer = SimpleNamespace(
        config=SimpleNamespace(hop_length=1),
        device="cpu",
    )
    model = SimpleNamespace(sampling_rate=sample_rate, audio_tokenizer=tokenizer)

    monkeypatch.setattr(model_module, "load_audio", lambda *_args: audio)
    monkeypatch.setattr(
        model_module.torch,
        "from_numpy",
        lambda *_args: pytest.fail("long reference audio must be rejected before encoding"),
    )

    with pytest.raises(ValueError, match="3-10 second"):
        model_module.OmniVoice.create_voice_clone_prompt(
            model,
            "long-reference.wav",
            ref_text="Transcript for the full reference audio.",
            preprocess_prompt=False,
        )
