"""Ensure nested stage percentages do not count the same work twice."""

from types import SimpleNamespace

import pytest

from examples import profile_inference


def test_nested_timings_are_exclusive(monkeypatch):
    ticks = iter([0.0, 1.0, 3.0, 4.0])
    monkeypatch.setattr(
        profile_inference, "time", SimpleNamespace(perf_counter=lambda: next(ticks))
    )
    timer = profile_inference.StageTimer("cpu")
    with timer.stage("preprocessing"):
        with timer.stage("voice_encoding"):
            pass
    assert dict(timer.seconds) == {"voice_encoding": 2.0, "preprocessing": 2.0}
    assert sum(timer.seconds.values()) == 4.0
    assert timer.stack == []


def test_failed_stage_closes_timing_stack(monkeypatch):
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(
        profile_inference, "time", SimpleNamespace(perf_counter=lambda: next(ticks))
    )
    timer = profile_inference.StageTimer("cpu")
    with pytest.raises(ValueError), timer.stage("audio_generation"):
        raise ValueError("Generation failed")
    assert timer.seconds["audio_generation"] == 2.0
    assert timer.stack == []
