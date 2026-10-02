# Gradio Video-to-Text Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a video/audio transcription tab to OmniVoice's existing Gradio demo, returning visible text plus downloadable TXT and SRT files.

**Architecture:** Keep all STT logic in `omnivoice/stt/transcriber.py` so the Gradio UI has a single callback boundary. The module lazily caches Faster Whisper models and owns conversion, output formatting, and temp cleanup; `demo.py` only binds Gradio inputs and outputs.

**Tech Stack:** Python 3.10+, Gradio, Faster Whisper, pydub/FFmpeg, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-gradio-video-transcription-design.md`

## Global Constraints

- Preserve the user's uncommitted changes in `omnivoice/cli/demo.py` and `pyproject.toml`; stage only STT-specific hunks/files.
- Support `.mp3`, `.wav`, `.m4a`, `.ogg`, `.webm`, `.flac`, `.mp4`, `.mkv`, `.mov`, and `.avi`, case-insensitively.
- Use Faster Whisper lazily, CPU `int8` by default, and CUDA `float16` only on explicit selection.
- Use pydub/FFmpeg conversion only for non-WAV media; delete temporary converted files.
- Retain TXT/SRT outputs under `.tmp/transcripts/` during the local session.
- Do not alter existing TTS model loading, TTS controls, or the existing optional ASR settings.

## Review Focus

- An uppercase video extension must be accepted and a non-media extension rejected; covered in Task 1.
- WAV input must not trigger conversion; covered in Task 1.
- An empty Whisper result must still produce valid empty TXT/SRT files; covered in Task 1.
- Missing FFmpeg for MP4 must return a useful error without loading Whisper; covered in Task 1.
- The STT tab must not cause Faster Whisper to load while a user only uses TTS; covered in Task 2 via lazy-import/callback tests.

---

## Planned File Structure

```text
omnivoice/
  stt/
    __init__.py
    transcriber.py          # STT service, conversion, files, model cache
  cli/
    demo.py                 # existing Gradio app, receives only STT tab/callback additions
tests/
  test_stt_transcriber.py   # service tests with fake Faster Whisper models
  test_demo_stt.py          # Gradio callback/tab construction tests
pyproject.toml              # add Faster Whisper dependency without changing existing user edits
README.md                   # add STT/FFmpeg usage notes
```

### Task 1: Isolated Faster Whisper Transcription Service

**Files:**
- Create: `omnivoice/stt/__init__.py`
- Create: `omnivoice/stt/transcriber.py`
- Create: `tests/test_stt_transcriber.py`

**Interfaces:**
- Produces `TranscriptSegment(start: float, end: float, text: str)`.
- Produces `TranscriptionResult(text: str, language: str | None, duration: float | None, segments: list[TranscriptSegment], txt_path: str, srt_path: str)`.
- Produces `WhisperTranscriber.transcribe_media(media_path: str, model_size: str = "small", language: str | None = None, task: str = "transcribe", device: str = "cpu") -> TranscriptionResult`.
- Produces module singleton `whisper_transcriber` consumed by Task 2.

- [ ] **Step 1: Write failing service tests**

```python
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
    monkeypatch.setattr(service, "_convert_to_wav", lambda *_: pytest.fail("WAV must skip conversion"))
    result = service.transcribe_media(str(tmp_path / "voice.wav"))

    assert result.text == "hello"
    assert Path(result.txt_path).read_text(encoding="utf-8") == "hello"
    assert "00:00:00,000 --> 00:00:01,500" in Path(result.srt_path).read_text(encoding="utf-8")
```

Add a test that mocks `_ffmpeg_available` as `False`, calls `transcribe_media("clip.mp4")`, and asserts it raises `RuntimeError` containing `FFmpeg`; add a test whose fake model returns no segments and asserts both output files exist with empty content.

- [ ] **Step 2: Run service tests to confirm failure**

Run: `uv run pytest tests/test_stt_transcriber.py -v`

Expected: FAIL because `omnivoice.stt` does not exist.

- [ ] **Step 3: Implement the service**

```python
@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    duration: float | None
    segments: list[TranscriptSegment]
    txt_path: str
    srt_path: str
```

Use `Path(media_path).suffix.lower()` against the exact allowed extension set. In `transcribe_media`, reject unsupported paths before model loading; use the original path for WAV and call `_convert_to_wav` for any other allowed type. `_convert_to_wav` must first check `shutil.which("ffmpeg")`, then use `AudioSegment.from_file(media_path).set_channels(1).set_frame_rate(16000).export(temp_wav, format="wav")`. Wrap cleanup in `finally` and delete only the temporary WAV created by this call.

Create the model cache key `(model_size, device, "int8" if device == "cpu" else "float16")`; call `WhisperModel.transcribe(..., vad_filter=True, language=language or None, task=task)`. Strip blank segment text, join kept entries with one space, then write UTF-8 TXT/SRT files under `self.output_dir`, named `{safe_stem}_{uuid4().hex}.txt` and `.srt`.

- [ ] **Step 4: Run service tests to verify pass**

Run: `uv run pytest tests/test_stt_transcriber.py -v`

Expected: PASS without a Whisper model download.

- [ ] **Step 5: Commit only new service files**

```bash
git add omnivoice/stt tests/test_stt_transcriber.py
git commit -m "feat: add video transcription service"
```

### Task 2: Gradio STT Tab and Safe Callback Boundary

**Files:**
- Modify: `omnivoice/cli/demo.py`
- Create: `tests/test_demo_stt.py`

**Interfaces:**
- Consumes `whisper_transcriber.transcribe_media()` from Task 1.
- Adds a helper `transcribe_uploaded_media(media_path, model_size, language, task, device) -> tuple[str, str, str | None, str | None, str | None]`.
- The tuple order is `(status, transcript, txt_path, srt_path, metadata)` and matches the five Gradio output components in the STT tab.

- [ ] **Step 1: Write failing callback tests**

```python
from omnivoice.cli import demo


def test_transcribe_uploaded_media_returns_download_paths(monkeypatch):
    class Result:
        text = "hello"
        language = "en"
        duration = 1.5
        txt_path = "C:/tmp/out.txt"
        srt_path = "C:/tmp/out.srt"

    monkeypatch.setattr(demo.whisper_transcriber, "transcribe_media", lambda **_: Result())
    status, text, txt_path, srt_path, metadata = demo.transcribe_uploaded_media(
        "C:/tmp/input.mp4", "small", "", "transcribe", "cpu"
    )

    assert status.startswith("Hoàn tất")
    assert text == "hello"
    assert (txt_path, srt_path) == ("C:/tmp/out.txt", "C:/tmp/out.srt")
    assert "en" in metadata


def test_transcribe_uploaded_media_handles_service_error(monkeypatch):
    monkeypatch.setattr(demo.whisper_transcriber, "transcribe_media", lambda **_: (_ for _ in ()).throw(RuntimeError("FFmpeg unavailable")))
    status, text, txt_path, srt_path, metadata = demo.transcribe_uploaded_media(
        "C:/tmp/input.mp4", "small", "", "transcribe", "cpu"
    )
    assert "Không thể" in status
    assert (text, txt_path, srt_path, metadata) == ("", None, None, "")
```

- [ ] **Step 2: Run callback tests to confirm failure**

Run: `uv run pytest tests/test_demo_stt.py -v`

Expected: FAIL because the callback and STT import are absent.

- [ ] **Step 3: Implement the callback and tab**

Add a module-level import `from omnivoice.stt.transcriber import whisper_transcriber` without constructing a Whisper model. Add this exact callback shape:

```python
def transcribe_uploaded_media(media_path, model_size, language, task, device):
    if not media_path:
        return "Hãy chọn một tệp video hoặc âm thanh.", "", None, None, ""
    try:
        result = whisper_transcriber.transcribe_media(
            media_path=media_path,
            model_size=model_size,
            language=language or None,
            task=task,
            device=device,
        )
    except (RuntimeError, ValueError) as exc:
        return f"Không thể chuyển đổi: {exc}", "", None, None, ""
    metadata = f"Ngôn ngữ: {result.language or 'không xác định'} · Thời lượng: {result.duration or 0:.1f}s"
    return "Hoàn tất chuyển đổi.", result.text, result.txt_path, result.srt_path, metadata
```

Inside the existing `gr.Tabs` block, add the third tab after voice design. Use `gr.File(label="Video hoặc âm thanh", type="filepath")`, Dropdown choices `tiny`, `base`, `small`, `medium`, `large-v3` with default `small`, language values `""`, `"vi"`, `"en"`, task choices `transcribe`/`translate`, and device choices `cpu`/`cuda`. Wire its button with `stt_button.click(transcribe_uploaded_media, inputs=[...], outputs=[status, transcript, txt_file, srt_file, metadata])`.

Do not move, reformat, or stage the existing user-owned Vietnamese UI changes outside the necessary import, helper, and tab additions.

- [ ] **Step 4: Run callback tests and existing tests**

Run: `uv run pytest tests/test_demo_stt.py tests/test_lora.py -v`

Expected: PASS. No model should load during test collection.

- [ ] **Step 5: Commit only the STT hunk and its test**

Use interactive staging to avoid taking ownership of the user's existing demo edits:

```bash
git add -p omnivoice/cli/demo.py
git add tests/test_demo_stt.py
git commit -m "feat: add transcription tab to demo"
```

### Task 3: Dependency, Documentation, and Full Verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `tests/test_demo_stt.py`

**Interfaces:**
- Adds `faster-whisper>=1.2,<2` to the runtime dependency list.
- Documents running `uv sync`, ensuring `ffmpeg -version` succeeds, and using the STT tab after `omnivoice-demo` starts.

- [ ] **Step 1: Write a failing dependency declaration test**

```python
from pathlib import Path


def test_project_declares_faster_whisper_dependency():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"faster-whisper>=1.2,<2"' in text
```

- [ ] **Step 2: Run the dependency test to confirm failure**

Run: `uv run pytest tests/test_demo_stt.py::test_project_declares_faster_whisper_dependency -v`

Expected: FAIL because the dependency is not declared.

- [ ] **Step 3: Add dependency and usage guidance**

Add exactly `"faster-whisper>=1.2,<2",` adjacent to the existing audio/runtime dependencies in `pyproject.toml`. Add a README section titled `Video to Text` that tells the user to run `uv sync`, verify FFmpeg through `ffmpeg -version`, launch `omnivoice-demo`, open the STT tab, upload a supported file, and download TXT/SRT. State that CPU is the default and CUDA selection needs a CUDA-capable Faster Whisper installation.

- [ ] **Step 4: Run project verification**

Run: `uv run pytest -q`

Expected: PASS, including `tests/test_lora.py`, `tests/test_stt_transcriber.py`, and `tests/test_demo_stt.py`.

Run: `uv run python -m compileall omnivoice`

Expected: compilation succeeds.

Run: `uv run python -c "from omnivoice.cli.demo import transcribe_uploaded_media; print(callable(transcribe_uploaded_media))"`

Expected: `True` without loading a Whisper model.

- [ ] **Step 5: Commit only STT dependency/docs changes**

```bash
git add -p pyproject.toml README.md
git add tests/test_demo_stt.py
git commit -m "docs: document video transcription setup"
```

## Self-Review

- **Spec coverage:** Task 1 implements all upload validation, conversion, cache, outputs, and cleanup. Task 2 creates the required Gradio tab without changing the TTS behavior. Task 3 declares the dependency, explains FFmpeg, and verifies the whole project.
- **Placeholder scan:** No unfinished work markers or deferred implementation steps remain.
- **Type consistency:** `WhisperTranscriber.transcribe_media()` returns `TranscriptionResult`; the Task 2 callback consumes its `text`, `language`, `duration`, `txt_path`, and `srt_path` fields in the exact defined order.
- **Review-focus coverage:** Task 1 covers case-insensitive extension handling, WAV bypass, empty output, and FFmpeg absence. Task 2 covers lazy model behavior at the Gradio boundary.
