from types import SimpleNamespace

from omnivoice.cli import demo
from omnivoice.stt.transcriber import TranscriptionCancelled
from omnivoice.utils.task_control import RunSnapshot, RunControl


def test_progress_markup_has_visible_percent_and_accessible_bar():
    snapshot = RunSnapshot("run", "running", 0.425, "Đang chép lời…")

    markup = demo._progress_markup(snapshot)

    assert "42%" in markup
    assert 'role="progressbar"' in markup
    assert 'aria-valuenow="42"' in markup
    assert "Đang chép lời" in markup


def test_cancelled_transcription_preserves_partial_text(monkeypatch):
    def stopped(**_kwargs):
        raise TranscriptionCancelled("đoạn đã nhận", "vi", 10.0)

    monkeypatch.setattr(demo.whisper_transcriber, "transcribe_media", stopped)
    control = RunControl()

    status, text, txt, srt, metadata = demo.transcribe_uploaded_media(
        "audio.mp3", "small", "vi", "transcribe", "cpu", control=control
    )

    assert "một phần" in status.lower()
    assert text == "đoạn đã nhận"
    assert txt is None and srt is None
    assert "vi" in metadata


def test_every_workflow_has_progress_and_stop_controls():
    fake_model = SimpleNamespace(sampling_rate=24_000)
    app = demo.build_demo(fake_model, "local", generate_fn=lambda *_args, **_kwargs: (None, ""))
    config = app.get_config_file()
    components = config["components"]

    stop_buttons = [
        item for item in components
        if item["type"] == "button" and item["props"].get("value") == "Dừng"
    ]
    timers = [item for item in components if item["type"] == "timer"]

    assert len(stop_buttons) == 3
    assert len(timers) == 3
    assert all(timer["props"]["active"] is False for timer in timers)


def test_begin_clone_activates_its_progress_timer():
    fake_model = SimpleNamespace(sampling_rate=24_000)
    app = demo.build_demo(fake_model, "local", generate_fn=lambda *_args, **_kwargs: (None, ""))
    begin_clone = next(
        function.fn
        for function in app.fns.values()
        if getattr(function.fn, "__name__", "") == "_begin_clone"
    )

    result = begin_clone("test-session")

    assert result[-1]["active"] is True
