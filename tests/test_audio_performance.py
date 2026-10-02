"""Guard long-form assembly cost and boundary behavior without model weights."""

import numpy as np
import pytest

from omnivoice.utils.audio import cross_fade_chunks


def test_long_form_join_concatenates_at_most_once(monkeypatch):
    concatenate = np.concatenate
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return concatenate(*args, **kwargs)

    monkeypatch.setattr(np, "concatenate", counted)
    chunks = [np.ones((1, 100), dtype=np.float32) for _ in range(100)]
    result = cross_fade_chunks(chunks, sample_rate=30)
    assert result.shape == (1, 100 * 100 + 99 * 3)
    assert len(calls) <= 1, (
        "Repeatedly copying the accumulated audio makes assembly quadratic"
    )


def test_join_preserves_fades_gaps_and_inputs():
    chunks = [np.ones((1, 6), dtype=np.float32) * value for value in (1, 2, 3)]
    originals = [chunk.copy() for chunk in chunks]
    result = cross_fade_chunks(chunks, sample_rate=30)
    expected = np.array(
        [[1, 1, 1, 1, 0.5, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 1.5, 3, 3, 3, 3]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result, expected)
    for chunk, original in zip(chunks, originals):
        np.testing.assert_array_equal(chunk, original)


@pytest.mark.parametrize("silence", [0, 0.3])
def test_single_chunk_is_returned_unchanged(silence):
    chunk = np.ones((2, 10), dtype=np.float32)
    assert cross_fade_chunks([chunk], sample_rate=30, silence_duration=silence) is chunk


def test_tiny_and_empty_chunks_keep_existing_tail_fades():
    chunks = [
        np.ones((1, 2), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        np.empty((1, 0), dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
    ]
    result = cross_fade_chunks(chunks, sample_rate=30)
    expected = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("channels", [(2, 1), (1, 2)])
def test_mismatched_channels_raise_instead_of_broadcasting(channels):
    chunks = [np.ones((count, 10), dtype=np.float32) for count in channels]
    with pytest.raises(ValueError):
        cross_fade_chunks(chunks, sample_rate=30)
