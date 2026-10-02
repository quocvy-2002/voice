from pathlib import Path
from types import SimpleNamespace

import pytest

from omnivoice.stt.transcriber import TranscriptionCancelled, WhisperTranscriber
from omnivoice.utils.task_control import RunControl


def _model_with_segments(*segments, duration=10.0):
    class Model:
        def transcribe(self, *_args, **_kwargs):
            return iter(segments), SimpleNamespace(language="vi", duration=duration)

    return Model()


def _segment(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


def test_reports_progress_from_segment_end_times(monkeypatch, tmp_path):
    transcriber = WhisperTranscriber(output_dir=tmp_path)
    monkeypatch.setattr(
        transcriber,
        "_get_model",
        lambda *_: _model_with_segments(
            _segment(0, 2, " Xin "), _segment(2, 5, " chào ")
        ),
    )
    reports = []

    result = transcriber.transcribe_media(
        str(tmp_path / "audio.wav"), progress_callback=lambda fraction, stage: reports.append((fraction, stage))
    )

    assert result.text == "Xin chào"
    assert [fraction for fraction, _ in reports if fraction is not None] == [0.2, 0.5]
    assert all(fraction is None or fraction < 1.0 for fraction, _ in reports)


def test_stop_retains_partial_text_without_downloads(monkeypatch, tmp_path):
    transcriber = WhisperTranscriber(output_dir=tmp_path)
    monkeypatch.setattr(
        transcriber,
        "_get_model",
        lambda *_: _model_with_segments(
            _segment(0, 2, "đoạn đầu"), _segment(2, 4, "đoạn sau")
        ),
    )
    control = RunControl()

    def on_progress(fraction, _stage):
        if fraction is not None and fraction >= 0.2:
            control.request_stop()

    with pytest.raises(TranscriptionCancelled) as stopped:
        transcriber.transcribe_media(
            str(tmp_path / "audio.wav"),
            progress_callback=on_progress,
            check_cancelled=control.check_cancelled,
        )

    assert stopped.value.text == "đoạn đầu"
    assert stopped.value.language == "vi"
    assert list(tmp_path.glob("*.txt")) == []
    assert list(tmp_path.glob("*.srt")) == []


def test_stop_after_conversion_removes_temporary_wav(monkeypatch, tmp_path):
    transcriber = WhisperTranscriber(output_dir=tmp_path)
    control = RunControl()
    converted = tmp_path / "converted.wav"

    def convert(_source):
        converted.write_bytes(b"wav")
        control.request_stop()
        return converted

    monkeypatch.setattr(transcriber, "_convert_to_wav", convert)
    monkeypatch.setattr(
        transcriber, "_get_model", lambda *_: pytest.fail("model must not load")
    )

    with pytest.raises(TranscriptionCancelled):
        transcriber.transcribe_media(
            str(tmp_path / "audio.mp4"), check_cancelled=control.check_cancelled
        )

    assert not converted.exists()


def test_unknown_duration_keeps_progress_indeterminate(monkeypatch, tmp_path):
    transcriber = WhisperTranscriber(output_dir=tmp_path)
    monkeypatch.setattr(
        transcriber,
        "_get_model",
        lambda *_: _model_with_segments(_segment(0, 1, "xong"), duration=0),
    )
    reports = []

    result = transcriber.transcribe_media(
        str(tmp_path / "audio.wav"), progress_callback=lambda fraction, stage: reports.append((fraction, stage))
    )

    assert result.text == "xong"
    assert all(fraction is None for fraction, _ in reports)
    assert Path(result.txt_path).exists()
