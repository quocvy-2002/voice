from types import SimpleNamespace

import pytest
import torch

from omnivoice.models.omnivoice import (
    GenerationTask,
    OmniVoice,
    OmniVoiceGenerationConfig,
)
from omnivoice.utils.task_control import RunControl, TaskCancelled


def _task(text="hello", target_len=1):
    return GenerationTask(
        batch_size=1,
        texts=[text],
        target_lens=[target_len],
        langs=[None],
        instructs=[None],
        ref_texts=[None],
        ref_audio_tokens=[None],
        ref_rms=[None],
        speed=None,
    )


def test_iterative_generation_stops_before_next_forward_pass():
    control = RunControl()
    forwards = []
    steps = []

    class FakeModel:
        device = "cpu"
        config = SimpleNamespace(num_audio_codebook=1, audio_mask_id=3)

        def _prepare_inference_inputs(self, *_args):
            return {
                "input_ids": torch.full((1, 1, 2), 3, dtype=torch.long),
                "audio_mask": torch.ones((1, 2), dtype=torch.bool),
            }

        def __call__(self, **_kwargs):
            forwards.append(True)
            return SimpleNamespace(logits=torch.zeros((2, 1, 2, 4)))

        def _predict_tokens_with_scoring(self, *_args):
            return torch.zeros((1, 1, 1), dtype=torch.long), torch.ones((1, 1, 1))

    def on_step(done, total):
        steps.append((done, total))
        control.request_stop()

    with pytest.raises(TaskCancelled):
        OmniVoice._generate_iterative(
            FakeModel(),
            _task(),
            OmniVoiceGenerationConfig(num_step=3, position_temperature=0),
            step_progress=on_step,
            check_cancelled=control.check_cancelled,
        )

    assert forwards == [True]
    assert steps == [(1, 3)]


def test_chunk_progress_is_monotonic_and_stop_skips_later_chunks(monkeypatch):
    monkeypatch.setattr(
        "omnivoice.models.omnivoice.chunk_text_punctuation",
        lambda **_kwargs: ["first", "second", "third"],
    )
    control = RunControl()
    fractions = []
    chunks_started = []

    class FakeModel:
        audio_tokenizer = SimpleNamespace(config=SimpleNamespace(frame_rate=1))

        def _estimate_target_tokens(self, *_args, **_kwargs):
            return 1

        def _generate_iterative(self, task, config, step_progress=None, check_cancelled=None):
            chunks_started.append(task.texts[0])
            for step in range(config.num_step):
                check_cancelled()
                step_progress(step + 1, config.num_step)
            return [torch.zeros((1, 1), dtype=torch.long)]

    def on_progress(fraction, _stage):
        fractions.append(fraction)
        if fraction >= 2 / 3:
            control.request_stop()

    with pytest.raises(TaskCancelled):
        OmniVoice._generate_chunked(
            FakeModel(),
            _task("a long sentence", 40),
            OmniVoiceGenerationConfig(num_step=2),
            progress_callback=on_progress,
            check_cancelled=control.check_cancelled,
        )

    assert chunks_started == ["first", "second"]
    assert fractions == sorted(fractions)
    assert all(value < 1.0 for value in fractions)
