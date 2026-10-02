from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydub import AudioSegment

from omnivoice.utils.task_control import TaskCancelled

ALLOWED_EXTENSIONS = frozenset({
    ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac",
    ".mp4", ".mkv", ".mov", ".avi",
})
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / ".tmp" / "transcripts"


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


class TranscriptionCancelled(TaskCancelled):
    def __init__(self, text: str, language: str | None, duration: float | None) -> None:
        super().__init__("Đã dừng chuyển đổi.")
        self.text = text
        self.language = language
        self.duration = duration


def is_supported_media(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def format_srt_timestamp(seconds: float) -> str:
    total_ms = round(max(seconds, 0) * 1000)
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    whole_seconds, milliseconds = divmod(total_ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def build_srt(segments: list[TranscriptSegment]) -> str:
    blocks = [
        f"{index}\n{format_srt_timestamp(segment.start)} --> "
        f"{format_srt_timestamp(segment.end)}\n{segment.text}"
        for index, segment in enumerate(segments, start=1)
    ]
    return "\n\n".join(blocks)


class WhisperTranscriber:
    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = output_dir
        self._model_cache: dict[tuple[str, str, str], object] = {}

    def _get_model(self, model_size: str, device: str, compute_type: str):
        key = (model_size, device, compute_type)
        if key not in self._model_cache:
            from faster_whisper import WhisperModel

            self._model_cache[key] = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
        return self._model_cache[key]

    def _ffmpeg_available(self) -> bool:
        return shutil.which("ffmpeg") is not None

    def _convert_to_wav(self, media_path: Path) -> Path:
        if not self._ffmpeg_available():
            raise RuntimeError("FFmpeg chưa được cài đặt hoặc chưa có trong PATH.")
        destination = self.output_dir / f"temp_{uuid.uuid4().hex}.wav"
        try:
            AudioSegment.from_file(media_path).set_channels(1).set_frame_rate(16_000).export(
                destination,
                format="wav",
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return destination

    def transcribe_media(
        self,
        media_path: str,
        model_size: str = "small",
        language: str | None = None,
        task: str = "transcribe",
        device: str = "cpu",
        progress_callback: Callable[[float | None, str], None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> TranscriptionResult:
        source = Path(media_path)
        if not is_supported_media(source.name):
            raise ValueError("Định dạng tệp không được hỗ trợ.")
        if task not in {"transcribe", "translate"}:
            raise ValueError("Tác vụ không hợp lệ.")
        if device not in {"cpu", "cuda"}:
            raise ValueError("Thiết bị không hợp lệ.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary_wav: Path | None = None
        segments: list[TranscriptSegment] = []
        detected_language: str | None = None
        duration: float | None = None

        def check() -> None:
            if check_cancelled is not None:
                check_cancelled()

        def report(fraction: float | None, stage: str) -> None:
            if progress_callback is not None:
                progress_callback(fraction, stage)

        try:
            try:
                check()
                audio_path = source
                if source.suffix.lower() != ".wav":
                    report(None, "Đang chuẩn bị âm thanh…")
                    temporary_wav = self._convert_to_wav(source)
                    audio_path = temporary_wav
                    check()

                report(None, "Đang tải mô hình nhận dạng…")
                compute_type = "int8" if device == "cpu" else "float16"
                model = self._get_model(model_size, device, compute_type)
                check()
                raw_segments, info = model.transcribe(
                    str(audio_path),
                    language=language or None,
                    task=task,
                    vad_filter=True,
                )
                detected_language = getattr(info, "language", None)
                duration = getattr(info, "duration", None)
                check()
                report(None, "Đang chép lời…")
                iterator = iter(raw_segments)
                while True:
                    check()
                    try:
                        item = next(iterator)
                    except StopIteration:
                        break
                    check()
                    clean_text = item.text.strip()
                    if clean_text:
                        segments.append(
                            TranscriptSegment(float(item.start), float(item.end), clean_text)
                        )
                    fraction = None
                    if duration is not None and duration > 0:
                        fraction = min(0.99, max(0.0, float(item.end) / duration))
                    report(fraction, "Đang chép lời…")
                check()

                text = " ".join(segment.text for segment in segments)
                safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source.stem).strip("_") or "transcript"
                token = uuid.uuid4().hex
                txt_path = self.output_dir / f"{safe_stem}_{token}.txt"
                srt_path = self.output_dir / f"{safe_stem}_{token}.srt"
                try:
                    txt_path.write_text(text, encoding="utf-8")
                    srt_path.write_text(build_srt(segments), encoding="utf-8")
                    check()
                except Exception:
                    txt_path.unlink(missing_ok=True)
                    srt_path.unlink(missing_ok=True)
                    raise
                return TranscriptionResult(
                    text=text,
                    language=detected_language,
                    duration=duration,
                    segments=segments,
                    txt_path=str(txt_path),
                    srt_path=str(srt_path),
                )
            except TaskCancelled as exc:
                raise TranscriptionCancelled(
                    text=" ".join(segment.text for segment in segments),
                    language=detected_language,
                    duration=duration,
                ) from exc
        finally:
            if temporary_wav is not None:
                temporary_wav.unlink(missing_ok=True)


whisper_transcriber = WhisperTranscriber()
