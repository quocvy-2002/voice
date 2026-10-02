from pathlib import Path

from omnivoice.cli import demo


def test_transcribe_uploaded_media_returns_download_paths(monkeypatch):
    class Result:
        text = "hello"
        language = "en"
        duration = 1.5
        txt_path = "C:/tmp/out.txt"
        srt_path = "C:/tmp/out.srt"

    monkeypatch.setattr(
        demo.whisper_transcriber,
        "transcribe_media",
        lambda **_kwargs: Result(),
    )

    status, text, txt_path, srt_path, metadata = demo.transcribe_uploaded_media(
        "C:/tmp/input.mp4", "small", "", "transcribe", "cpu"
    )

    assert status.startswith("Hoàn tất")
    assert text == "hello"
    assert (txt_path, srt_path) == ("C:/tmp/out.txt", "C:/tmp/out.srt")
    assert "en" in metadata


def test_transcribe_uploaded_media_handles_service_error(monkeypatch):
    monkeypatch.setattr(
        demo.whisper_transcriber,
        "transcribe_media",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("FFmpeg unavailable")),
    )

    status, text, txt_path, srt_path, metadata = demo.transcribe_uploaded_media(
        "C:/tmp/input.mp4", "small", "", "transcribe", "cpu"
    )

    assert "Không thể" in status
    assert (text, txt_path, srt_path, metadata) == ("", None, None, "")


def test_project_declares_faster_whisper_dependency():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"faster-whisper>=1.2,<2"' in text


def test_shared_upload_hint_supports_video_and_audio():
    source = Path(demo.__file__).read_text(encoding="utf-8")
    assert "Kéo thả tệp hoặc bấm để chọn tệp" in source
    assert "Kéo thả âm thanh hoặc bấm để chọn tệp" not in source
