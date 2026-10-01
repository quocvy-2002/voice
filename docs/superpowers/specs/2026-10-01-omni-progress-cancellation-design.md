# OmniVoice Task Progress and Cancellation

## Goal

Give users of all three long-running workflows—voice cloning, voice design, and media transcription—a visible estimate of progress and a way to stop the active task. A stop request must reach the backend work that is already running, rather than only removing a queued Gradio event.

## User Experience

- Each workflow has its own progress display and Stop button, placed next to its run action and result area.
- Before a task starts, progress is idle and Stop is unavailable.
- While it runs, show a short stage/status message, a percentage, and a progress bar. The percentage is an estimate of completed work, not a promise of remaining wall-clock time.
- Stop requests cancellation for that workflow's current run. The UI reports that stopping is in progress until the backend reaches a safe checkpoint, then shows a stopped state.
- A completed run shows 100% and the normal result. A failed run shows a useful error and does not masquerade as a successful completion.
- A workflow can be started again after completion, failure, or cancellation. A late progress/result update from an older run must not overwrite the newer run's state.

## Workflow Behavior

### Media transcription

- Progress is estimated from the end timestamp of the latest recognized segment divided by the media duration, clamped to 0–99% until completion. Conversion, model loading, and other preparation stages show an indeterminate/preparing message until measurable transcription progress is available.
- Check for cancellation between recognized segments. Because Faster-Whisper exposes segments lazily, this provides cooperative interruption without waiting for the whole file to finish.
- On cancellation, keep and display the text recognized so far with a clear “stopped / partial transcript” status. Do not create or offer final TXT/SRT downloads for an incomplete transcript.
- On normal completion, retain the existing complete transcript, metadata, and TXT/SRT outputs.
- Temporary converted audio is removed on success, error, and cancellation.

### Voice cloning and voice design

- Progress is reported from generation steps, with chunk-level advancement for long text. The status distinguishes preparation, active generation, and audio post-processing where those stages are observable.
- Check for cancellation at model generation step boundaries and between text chunks. A stop request takes effect at the next safe boundary; it is not required to interrupt a single in-flight GPU/CPU kernel.
- If canceled, discard incomplete generated audio and show a stopped status. Do not present partial audio as a completed result.
- On normal completion, preserve the existing generated audio and status behavior.

## Backend Contract

- Add a small run-scoped progress/cancellation contract shared by the Gradio callbacks and model/transcriber work. It carries a unique run identity, a thread-safe cancellation signal, and a progress/status reporting mechanism.
- Each user session and each workflow owns its own active run state. A Stop action must not cancel another user's run or another workflow in the same session.
- Cancel actions must be dispatchable while the corresponding synchronous task is executing. Gradio queue cancellation may still be used for queued work, but cannot be the only cancellation mechanism.
- Cancellation is cooperative. Long-running loops check the signal at defined safe boundaries and propagate a distinguishable cancelled outcome so callbacks can apply the workflow-specific result policy.
- Progress callbacks/parameters are optional and default to the current behavior for existing model and transcription callers.
- Reset or replace a run's state on each new invocation. Ignore reports whose run identity is no longer active.

## UI State Contract

Each workflow exposes these states: `idle`, `preparing`, `running`, `cancelling`, `cancelled`, `completed`, and `failed`.

- Progress is monotonic within one run and capped below 100% while active; only successful completion sets it to 100%.
- Stage text supplies context when a numeric estimate is not yet available.
- Stop is enabled only while the run is preparing or running; after it is pressed, it is disabled until the run resolves.
- Stopped, failed, and completed states are visually and textually distinct; status must not rely on color alone.

## Scope

### Included

- Backend progress reporting and cooperative cancellation for the local default OmniVoice generation and Faster-Whisper media transcription paths.
- Per-workflow Gradio controls and status displays for voice cloning, voice design, and media transcription.
- Partial transcript display after cancellation, with complete-download behavior retained only for successful transcription.
- Run isolation, stale-update protection, and cleanup on all terminal paths.

### Excluded

- Hard interruption of an active native/GPU kernel or forcibly terminating the server process.
- Persisting task state across server restarts, browser refreshes, or separate sessions.
- Exact predictions of remaining time or an assertion that progress percentage equals elapsed compute time.
- Cancelling third-party/custom generation callables that do not implement the optional progress/cancellation contract; these should continue to work with limited or unavailable fine-grained progress and cancellation.
- Changes to transcription quality, model selection defaults, audio formats, or unrelated interface redesign.

## Acceptance Criteria

1. Each of the three workflows shows progress and has a Stop control while work is active.
2. Stop interrupts ongoing transcription at a segment boundary and displays the recognized partial text with an explicit partial/stopped status.
3. Cancelled transcription does not expose incomplete TXT/SRT downloads; completed transcription still exposes the normal downloads.
4. Stop interrupts local voice cloning and voice design generation at a model step or chunk boundary and never exposes incomplete audio as a successful result.
5. Progress advances during TTS generation and reflects recognized media duration during transcription; active runs do not report 100% before completion.
6. Preparing/loading and stopping stages are communicated even when no reliable numeric progress is available.
7. Concurrent runs are isolated by session and workflow, and stale updates from older runs cannot replace newer state.
8. Existing callers that do not provide progress/cancellation continue to work with their current call pattern.
9. Success, cancellation, and failure all release temporary transcription files and leave the workflow ready to run again.

## Risks and Limitations

- Segment-based transcription progress is an estimate based on audio timestamps. VAD gaps and segment batching can make it pause or advance unevenly.
- Cooperative cancellation latency depends on how long the current model step or audio segment operation takes.
- Some external generation functions may not expose step boundaries. Their task can still report broad stages, but cannot promise the same responsiveness as the built-in model path.
- Gradio session-state behavior under concurrent browser requests must be handled with thread-safe run state rather than relying on mutation of a shared component value alone.

## Assumptions

- The existing Gradio demo remains the UI and local execution surface.
- “Both voice workflows” means voice cloning and voice design; “media workflow” means uploaded audio/video transcription.
- The user's chosen cancellation policy is to retain partial transcript text and discard incomplete generated audio.
- Percentages communicate completed work within the task, not a calibrated ETA.
