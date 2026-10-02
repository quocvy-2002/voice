# Gradio Video-to-Text Integration Design

## Objective

Add video/audio transcription to the existing OmniVoice Gradio demo at `D:\youtube\voice_new`. Users upload a media file through a new interface tab, receive a transcript in the browser, and can download matching TXT and SRT files.

The existing voice-cloning and voice-design tabs, including the user's uncommitted Vietnamese UI changes, remain intact.

## User Flow

1. The user launches the existing `omnivoice-demo` Gradio interface.
2. They select the new **Chuyển video thành text** tab.
3. They upload one supported audio or video file.
4. They choose a Whisper model size, language (automatic detection by default), and task (`transcribe` or `translate`).
5. They start transcription and see status plus the completed text.
6. They download generated TXT and SRT files from the same tab.

## Architecture

```
Gradio tab in omnivoice/cli/demo.py
  └─ transcription callback
        └─ omnivoice/stt/transcriber.py
              ├─ Faster Whisper model cache
              ├─ media conversion with pydub/FFmpeg when required
              └─ TXT/SRT output writer under .tmp/transcripts/
```

### New STT Module

- `omnivoice/stt/transcriber.py` owns all speech-to-text behavior; the Gradio demo only calls one high-level function.
- `WhisperTranscriber` caches Faster Whisper models using `(model_size, device, compute_type)` keys.
- `transcribe_media()` accepts an uploaded Gradio path, converts non-WAV input to a temporary mono 16 kHz WAV with pydub/FFmpeg, transcribes with VAD filtering, and returns text, detected language, duration, timestamped segments, plus absolute TXT/SRT paths.
- Temporary converted WAV files are deleted after completion or failure. Download outputs remain under `.tmp/transcripts/` for the current local session.

### Demo Integration

- Add a third `gr.TabItem` in `build_demo()` called **Chuyển video thành text**.
- Use `gr.File(type="filepath")` so both video and audio extensions can be uploaded without changing the existing TTS input controls.
- Use a `gr.Textbox` for the transcript, `gr.File` components for the resulting TXT and SRT files, and a status `gr.Textbox`.
- The callback catches expected conversion/transcription errors and returns a Vietnamese status message rather than crashing the Gradio queue.
- The TTS model continues to load exactly as today. Faster Whisper loads lazily only when the STT tab is submitted, so TTS-only users do not pay its model startup cost.

## Input and Output Rules

- Supported input extensions: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.webm`, `.flac`, `.mp4`, `.mkv`, `.mov`, `.avi`.
- Input extension matching is case-insensitive.
- WAV input skips conversion. Other inputs need an FFmpeg executable available to pydub.
- Output names are derived from the uploaded stem plus a unique suffix to prevent overwrite:
  - `<stem>_<token>.txt`
  - `<stem>_<token>.srt`
- TXT contains joined transcript text. SRT contains numbered segments and `HH:MM:SS,mmm` timestamps.

## Dependencies and Devices

- Add `faster-whisper>=1.2,<2` to the main `pyproject.toml` dependencies.
- Reuse existing `pydub` for conversion and document FFmpeg as a system dependency.
- Default to CPU `int8`. Allow `cuda` only via an explicit STT device selector; use `float16` when CUDA is selected.
- Do not modify the existing OmniVoice TTS model loading or its optional ASR configuration.

## Testing

- Unit tests validate extension handling, SRT timestamp formatting, and generated output files without loading a Whisper model.
- Service tests replace the model loader with a fake model to verify segment cleanup, joined text, metadata, and caching behavior.
- Demo tests inject a fake STT callback/service and verify that `build_demo()` includes the STT callback without loading the OmniVoice model.
- Existing `tests/test_lora.py` remains unchanged and must pass.

## Out of Scope

- No separate FastAPI app, React/Vite frontend, authentication, cloud storage, background task queue, live percentage progress, persistent job history, or changes to the current TTS tabs.
