# OmniVoice Task Progress and Cancellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show meaningful progress and stop active work in voice cloning, voice design, and media transcription, retaining partial transcript text after cancellation.

**Architecture:** A per-session, per-workflow run registry owns thread-safe cancellation and progress state. A Gradio `State` holds a private session token and uses its deletion callback to release that session; the existing callbacks pass optional progress/check functions into Faster-Whisper and OmniVoice. Short polling events render the latest run state while inference runs in Gradio's worker pool. Cancellation is cooperative at segment and diffusion-step boundaries.

**Tech Stack:** Python 3.10+, Gradio 6.9, Faster-Whisper 1.2, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-omni-progress-cancellation-design.md`

## Global Constraints

- Keep the current Gradio application and local model architecture.
- Progress is an estimate of completed work; active tasks remain below 100%.
- Stopped transcription displays partial text without final TXT/SRT files; stopped TTS discards incomplete audio.
- Existing model and transcriber callers must work without passing the new optional callbacks.
- Preserve the unrelated pre-existing edits in `README.md`, `omnivoice/cli/demo.py`, `pyproject.toml`, `uv.lock`, `omnivoice/stt/`, and existing tests. Stage only reviewed feature changes.
- A current user transcription may still be running on port 8001. Do not restart the local server until that run has finished or the user stops it.

## File Map

- Create `omnivoice/utils/task_control.py`: cancellation exception, run snapshot/control, session/workflow registry.
- Modify `omnivoice/stt/transcriber.py`: optional callbacks, segment progress, partial cancellation result, cleanup.
- Modify `omnivoice/models/omnivoice.py`: optional generation callbacks and step/chunk progress.
- Modify `omnivoice/cli/demo.py`: three sets of Stop/progress controls, event wiring, status and result policies.
- Create `tests/test_task_control.py`, `tests/test_stt_progress.py`, `tests/test_omnivoice_progress.py`, and `tests/test_demo_task_progress.py`: focused behavior checks with fakes; no model download.
- Update `README.md` only if the existing usage section needs a short description of Stop and estimated progress; preserve its unrelated working-tree changes.

## Review Focus

1. Two browser sessions start tasks: stopping one must leave the other's cancellation signal unset. Task 1 owns this test.
2. A stop arrives during slow media conversion or model loading: the next checkpoint must stop before transcription begins or files are published. Task 2 owns this test.
3. Faster-Whisper produces no segments or a final segment with a zero/unknown duration: progress must remain valid and successful completion still reaches 100%. Task 2 owns this test.
4. A long TTS input is split into chunks: progress must be monotonic across chunk boundaries and Stop must prevent later chunks and audio decoding. Task 3 owns this test.
5. An older callback reports after a newer run begins: the registry and UI adapter must ignore the older run's status/result. Tasks 1 and 4 own this test.

---

### Task 1: Run-scoped control and session isolation

**Files:** Create `omnivoice/utils/task_control.py`; create `tests/test_task_control.py`.

**Interfaces:** Produce `TaskCancelled`, `RunControl.report(fraction: float | None, stage: str)`, `RunControl.check_cancelled()`, `RunControl.request_stop()`, and `RunControl.snapshot()`. Registry methods are `begin(session_token, workflow) -> RunControl | None`, `get(session_token, workflow, run_id) -> RunControl | None`, `request_stop(session_token, workflow, run_id) -> bool`, `finish(session_token, workflow, run_id, outcome) -> bool`, `snapshot(session_token, workflow) -> RunSnapshot | None`, and `drop_session(session_token) -> None`. A run has an opaque UUID `run_id`; workflows are `clone`, `design`, and `transcribe`.

- [ ] **Step 1: Add focused tests for isolated cancellation, monotonic progress, terminal states, duplicate starts, and stale run IDs.**

```python
registry = RunRegistry()
first = registry.begin("browser-a", "transcribe")
other = registry.begin("browser-b", "transcribe")
first.report(0.7, "Đang chép lời")
first.report(0.3, "Đang chép lời")
assert first.snapshot().fraction == 0.7
assert registry.request_stop("browser-a", "transcribe", first.run_id)
with pytest.raises(TaskCancelled):
    first.check_cancelled()
other.check_cancelled()
assert registry.begin("browser-a", "transcribe") is None  # still active
registry.finish("browser-a", "transcribe", first.run_id, "cancelled")
new = registry.begin("browser-a", "transcribe")
assert new is not None and new.run_id != first.run_id
assert not registry.finish("browser-a", "transcribe", first.run_id, "completed")
```

- [ ] **Step 2: Run only the new control tests; confirm failure because the module does not exist.** Run `.venv\Scripts\python.exe -m pytest tests/test_task_control.py -q`.
- [ ] **Step 3: Implement `RunControl` with `threading.Event` and a lock, plus `RunRegistry` with a lock.** Use a frozen `RunSnapshot(run_id, state, fraction, stage)`; `report` clamps fractions to `[0, .99]`, never decreases them, and cannot turn `cancelling` back into `running`; only `finish(..., "completed")` sets `1.0`. `request_stop` sets the Event and `cancelling` under the same lock. Reject duplicate active starts, and only mutate state when the supplied `run_id` matches the current run.

```python
def check_cancelled(self) -> None:
    if self.cancel_event.is_set():
        raise TaskCancelled()

def finish(self, session_token: str, workflow: str, run_id: str,
           outcome: Literal["cancelled", "completed", "failed"]) -> bool:
    with self._lock:
        current = self._runs.get((session_token, workflow))
        if current is None or current.run_id != run_id:
            return False
        current.finish(outcome)
        return True
```

- [ ] **Step 4: Run the control tests; review the reported assertions.**
- [ ] **Step 5: Commit only the new control module and its focused tests.**

### Task 2: Streaming transcription progress and partial cancellation

**Files:** Modify `omnivoice/stt/transcriber.py`; create `tests/test_stt_progress.py`.

**Interfaces:** Add optional `progress_callback: Callable[[float | None, str], None] | None = None` and `check_cancelled: Callable[[], None] | None = None` parameters to `WhisperTranscriber.transcribe_media`. Produce `TranscriptionCancelled(TaskCancelled)` with `text`, `language`, and `duration`. Existing `TranscriptionResult` and successful file paths stay unchanged.

- [ ] **Step 1: Add fake-model tests for per-segment progress, cancellation with partial text, cancellation during preparation, empty transcript, and WAV cleanup.** A fake segment iterator calls `stop_control.request_stop()` after yielding its first segment; assert `TranscriptionCancelled.text` contains that segment and the output directory has no TXT/SRT. Set `info.duration=10.0` and segment ends `2.0`, `5.0`; assert reported fractions include `.2`, `.5` and stay below `1.0`.
- [ ] **Step 2: Run `tests/test_stt_progress.py`; confirm the new callback arguments or exception are missing.**
- [ ] **Step 3: Replace the eager segment list comprehension with an explicit loop.** Call `check_cancelled()` before conversion, after conversion, after model load, before/after fetching a segment, and just before publishing downloads. After each accepted segment, call `progress_callback(min(.99, max(0.0, end / info.duration)), "Đang chép lời")` when duration is positive; otherwise report `None` with a stage. Maintain `segments` and build partial text if `TaskCancelled` occurs. Raise `TranscriptionCancelled` carrying partial text and metadata; preserve `finally` removal of temporary WAV. Write final TXT/SRT only after the iteration completes without cancellation and remove any partial final files on write error.

```python
def check() -> None:
    if check_cancelled is not None:
        check_cancelled()

try:
    for item in raw_segments:
        check()
        clean_text = item.text.strip()
        if clean_text:
            segments.append(TranscriptSegment(float(item.start), float(item.end), clean_text))
        if progress_callback is not None:
            duration = getattr(info, "duration", None)
            fraction = min(.99, max(0.0, float(item.end) / duration)) if duration and duration > 0 else None
            progress_callback(fraction, "Đang chép lời")
    check()
except TaskCancelled as exc:
    raise TranscriptionCancelled(
        text=" ".join(segment.text for segment in segments),
        language=getattr(info, "language", None),
        duration=getattr(info, "duration", None),
    ) from exc
```

- [ ] **Step 4: Run the new transcription tests and existing `tests/test_stt_transcriber.py`; inspect all results.**
- [ ] **Step 5: Commit only transcription changes and new tests.**

### Task 3: Step and chunk progress in OmniVoice

**Files:** Modify `omnivoice/models/omnivoice.py`; create `tests/test_omnivoice_progress.py`.

**Interfaces:** Add optional `progress_callback: Callable[[float | None, str], None] | None = None` and `check_cancelled: Callable[[], None] | None = None` parameters before `**kwargs` in `OmniVoice.generate`. Pass them into `_generate_iterative` and `_generate_chunked`; keep every old call valid. `check_cancelled` raises `TaskCancelled` from Task 1.

- [ ] **Step 1: Add fake inference tests that record progress across multiple steps and chunks, stop at a selected step, and verify the decoder is never called after cancellation.** Use a small fake task/model path without loading checkpoints; cover a one-chunk input and a chunked input.
- [ ] **Step 2: Run `tests/test_omnivoice_progress.py`; confirm failure on the absent optional callback contract.**
- [ ] **Step 3: Add cancellation checkpoints before and after preprocessing, before each diffusion step, between chunks, and before decode/post-processing.** Call the step callback after each completed iteration; `_generate_chunked` scales `(chunk_index * num_step + step_index) / (chunk_count * num_step)` into a fraction. `generate` maps short/long phase fractions into a monotonic global fraction when a batch contains both kinds. Keep active fractions below `1.0` and report the decoding stage after generation.

```python
def check() -> None:
    if check_cancelled is not None:
        check_cancelled()

def report_step(step: int) -> None:
    if progress_callback is not None:
        progress_callback(step / gen_config.num_step, "Đang tạo giọng")

# Call check() immediately before the existing forward pass in each iteration.
# Call report_step(step + 1) immediately after the existing token update.
```

- [ ] **Step 4: Run the new model progress tests and the existing model tests selected by `rg --files tests | rg 'omnivoice|generate'`; inspect failures before moving on.**
- [ ] **Step 5: Commit only the model callback changes and their tests.**

### Task 4: Connect Stop and progress to all three Gradio workflows

**Files:** Modify `omnivoice/cli/demo.py`; create `tests/test_demo_task_progress.py`.

**Interfaces:** Consume Tasks 1–3. The `build_demo` closure owns one `RunRegistry`. Use `gr.State(value=lambda: uuid.uuid4().hex, delete_callback=registry.drop_session)` for the private session token, separate `gr.State` values for active run IDs, and `gr.Timer(value=0.5)` per workflow. Run functions return the existing audio or transcript outputs plus terminal status; timer functions return progress markup and button enabled states.

- [ ] **Step 1: Add callback-level tests with a fake model/transcriber for cancellation result policy, three independent workflow keys, custom `generate_fn` compatibility, and stale IDs.** Confirm stopped TTS returns no audio, stopped STT returns partial text with both downloads `None`, and a callback for an old `run_id` returns `gr.skip()` outputs.
- [ ] **Step 2: Run `tests/test_demo_task_progress.py`; confirm the new UI/callback contract is absent.**
- [ ] **Step 3: Implement shared Gradio adapters inside `build_demo`.** A `queue=False` start callback calls `registry.begin`, clears old output, disables Run, and enables Stop. Its `.then(...)` event runs inference with `concurrency_limit=1` and a shared TTS `concurrency_id`; STT may use its own worker. A `queue=False` Stop callback calls `request_stop` and immediately shows “Đang dừng…”. A `queue=False` timer tick reads `snapshot()` and renders a text-labelled `<progress max="100" value="...">` with escaped stage text; when fraction is unknown, render an indeterminate progress element and stage. The timer also keeps button states aligned with the snapshot. `gr.State.delete_callback` calls `registry.drop_session`, which requests stop for any active runs before removing their registry entries.

```python
def _stop(workflow: str, run_id: str, session_token: str):
    registry.request_stop(session_token, workflow, run_id)
    return gr.update(interactive=False)

def _poll(workflow: str, session_token: str):
    snapshot = registry.snapshot(session_token, workflow)
    return render_progress(snapshot), run_button_update(snapshot), stop_button_update(snapshot)
```

- [ ] **Step 4: Pass `control.report` and `control.check_cancelled` to built-in `_gen_core`/`model.generate` and transcription.** In clone preparation, report a preparation stage and check cancellation before/after `create_voice_clone_prompt`. For external `generate_fn`, call the original signature, report only broad stages, then check cancellation before returning audio. Catch `TranscriptionCancelled` separately from errors to retain partial text; catch `TaskCancelled` in TTS to discard audio. Guard completion and returned outputs with the matching run ID.
- [ ] **Step 5: Add Stop buttons and progress displays in the existing result panels, preserving current desktop layout and Vietnamese labels.** Initialize them to idle, keep Stop disabled until a run starts, and make terminal statuses distinct in words. Do not let a polled old run overwrite a new run's component state.
- [ ] **Step 6: Run new callback tests and existing `tests/test_demo_stt.py`; inspect results.**
- [ ] **Step 7: Commit only Gradio wiring and focused tests.** If touching `demo.py` includes pre-existing edits, stage the feature hunks interactively or commit a reviewed full file only after confirming the diff clearly attributes the existing work; never silently include unrelated files.

### Task 5: Local behavior review and usage note

**Files:** Modify `README.md` only for user-facing Stop/progress usage; otherwise no source file.

**Interfaces:** Verify the three tasks from the same browser UI; no new API.

- [ ] **Step 1: Check that the earlier transcription on the local server is no longer active before restarting to load changes.** If still active, inspect its state without terminating it and postpone restart until it completes.
- [ ] **Step 2: Run targeted tests for Tasks 1–4 once, then launch the modified app on the existing local port.** Observe actual command output and browser console/network errors.
- [ ] **Step 3: In the browser, start and stop a media transcription after at least one segment; confirm percentage advances, partial text remains, downloads stay empty, and another run can complete to 100% with TXT/SRT.** Use a short local sample or fake inference fixture if a large live model run would be impractical.
- [ ] **Step 4: Start and stop voice cloning and voice design; confirm step progress, responsive Stop, no incomplete audio, and successful retry.** Check at desktop width and narrow fallback for visible focus, labels, and layout.
- [ ] **Step 5: Add a brief Vietnamese README note explaining that percentage is estimated and Stop takes effect at the next safe checkpoint.** Review the exact `README.md` hunk to preserve existing local edits, then commit only that hunk if added.
- [ ] **Step 6: Review the complete feature diff against the spec and report any remaining latency limits or custom generator restrictions.**

## Completion Check

- Every acceptance criterion in the spec maps to a task above: UI (Task 4), partial STT and cleanup (Task 2), TTS cancellation/progress (Task 3), session isolation and stale update protection (Tasks 1 and 4), and local usability (Task 5).
- Verify both `tests/test_demo_stt.py` and `tests/test_stt_transcriber.py` still pass without changed call sites.
- A final review should inspect cleanup after errors, late callback races, queued cancellation, and CPU/GPU checkpoint latency before any completion claim.
