from pathlib import Path

import pytest

from omnivoice.stt.transcriber import WhisperTranscriber, is_supported_media


def test_extensions_are_case_insensitive():
    assert is_supported_media("clip.MP4")
    assert is_supported_media("voice.wav")
    assert not is_supported_media("notes.txt")


def test_wav_skips_conversion_and_writes_txt_srt(monkeypatch, tmp_path):
    class Segment:
        start, end, text = 0.0, 1.5, " hello "

    class Info:
        language, duration = "en", 1.5

    class Model:
        def transcribe(self, *_args, **_kwargs):
            return iter([Segment()]), Info()

    service = WhisperTranscriber(output_dir=tmp_path)
    monkeypatch.setattr(service, "_get_model", lambda *_: Model())
    monkeypatch.setattr(
        service,
        "_convert_to_wav",
        lambda *_: pytest.fail("WAV must skip conversion"),
    )

    result = service.transcribe_media(str(tmp_path / "voice.wav"))

    assert result.text == "hello"
    assert Path(result.txt_path).read_text(encoding="utf-8") == "hello"
    assert "00:00:00,000 --> 00:00:01,500" in Path(result.srt_path).read_text(encoding="utf-8")


def test_missing_ffmpeg_rejects_non_wav_before_model_load(monkeypatch, tmp_path):
    service = WhisperTranscriber(output_dir=tmp_path)
    monkeypatch.setattr(service, "_ffmpeg_available", lambda: False)
    monkeypatch.setattr(service, "_get_model", lambda *_: pytest.fail("must not load model"))

    with pytest.raises(RuntimeError, match="FFmpeg"):
        service.transcribe_media(str(tmp_path / "clip.mp4"))


def test_empty_result_writes_empty_output_files(monkeypatch, tmp_path):
    class Info:
        language, duration = "vi", 0.0

    class Model:
        def transcribe(self, *_args, **_kwargs):
            return iter([]), Info()

    service = WhisperTranscriber(output_dir=tmp_path)
    monkeypatch.setattr(service, "_get_model", lambda *_: Model())

    result = service.transcribe_media(str(tmp_path / "silence.wav"))

    assert result.text == ""
    assert Path(result.txt_path).read_text(encoding="utf-8") == ""
    assert Path(result.srt_path).read_text(encoding="utf-8") == ""
