# OmniVoice Desktop UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Redesign the current Vietnamese OmniVoice Gradio page into a compact desktop voice-production workspace that keeps the three existing workflows easy to choose, complete, and review.

**Architecture:** Keep the current Gradio application, components, callback signatures, and model-facing values. Refactor the page composition and theme/CSS in `build_demo()` in focused slices: shared chrome, voice-generation workspaces, transcription workspace, then browser review at desktop and narrow widths.

**Tech Stack:** Python 3.10+, Gradio 6, existing inline CSS and `gr.Blocks` components, existing `omnivoice-demo` local server.

**Spec:** `docs/superpowers/specs/2026-09-30-desktop-ui-redesign-design.md`

## Global Constraints

- The primary use case is running OmniVoice locally in a desktop browser.
- The approved visual direction is a compact desktop voice-production workspace using the current teal identity.
- Use a centered maximum content width around 1440px.
- On wide screens, place the task form on the left and the output/status area on the right.
- At narrower widths, stack the form and output while preserving their order and readable control widths.
- Preserve all three workflows, current input controls and outputs, current Vietnamese copy unless hierarchy/guidance requires a small change, and all model-facing option values.
- Implement inside `omnivoice/cli/demo.py` using Gradio components, existing callbacks, CSS classes, and theme settings.
- Do not add a frontend framework or change model loading, inference, transcription, audio processing, file formats, callback contracts, or queue behavior.
- Preserve the user's pre-existing uncommitted changes to `omnivoice/cli/demo.py`, speech-to-text support, tests, and documentation.

## Review Focus

1. The 600+ language dropdown contains long Vietnamese labels; confirm it remains searchable and does not widen or overflow its panel.
2. Long user text and long transcripts can expand component height; confirm the input and transcript areas remain readable and usable at desktop width.
3. Before generation, audio output is empty; confirm the right panel still communicates what will appear there without showing a fake result.
4. When automatic reference transcription is disabled, the reference-text helper changes; confirm the reordered voice-cloning controls keep this requirement visible.
5. At 1024px and below, the desktop columns can become cramped; confirm the layout stacks before controls clip or force horizontal scrolling.

## File Map

- `omnivoice/cli/demo.py` — sole implementation target. It owns the Gradio theme, CSS, header, three task tabs, inputs, output areas, and callback wiring. Keep model/transcriber functions and callback contracts unchanged.
- `docs/superpowers/specs/2026-09-30-desktop-ui-redesign-design.md` — approved design contract; read before making UI changes.

## Implementation Tasks

### Task 1: Build the shared desktop workspace frame

**Files:**
- Modify: `omnivoice/cli/demo.py` (`build_demo()`: theme/CSS, current `.ov-hero`, `.ov-tabs`, footer)

**Interfaces:**
- Consumes: current `build_demo(model, checkpoint, generate_fn=None, asr_enabled=True)` signature.
- Produces: the same `gr.Blocks` result and the same three `gr.TabItem` workflows, with a compact header, visible task navigation, and shared desktop/responsive styling.

- [x] Replace the large hero content with a compact Vietnamese header containing OmniVoice, a one-line purpose statement, and a truthful ready-state label. Since `build_demo()` runs only after the model loads, label the state as ready rather than inventing dynamic loading or device information.

  Use this structure in the existing `gr.HTML` block:

  ```html
  <header class="ov-header">
    <div class="ov-brand">
      <span class="ov-brand-mark">OV</span>
      <div><strong>OmniVoice</strong><small>Phòng thu giọng nói</small></div>
    </div>
    <p>Nhân bản giọng nói, thiết kế chất giọng và chép lời media.</p>
    <span class="ov-ready" role="status">Sẵn sàng</span>
  </header>
  ```

- [x] Set `fill_width=True` on the outer `gr.Blocks`; Gradio otherwise constrains the app to its default maximum width, so the CSS max-width alone cannot use a wide desktop viewport.

- [x] Update the existing theme and CSS with a restrained teal/neutral palette, a centered content width around 1440px, consistent spacing, visible focus styles, readable labels, and moderate panel borders/radii.

  Keep the styling in the existing `css` string and use named classes for the new frame:

  ```css
  .gradio-container { width: calc(100vw - 24px) !important; max-width: 1440px !important; margin: 0 auto !important; }
  .ov-header { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 24px; }
  .ov-panel { min-width: 0; border: 1px solid var(--border-color-primary); border-radius: 12px; }
  :is(button, input, textarea, [role="tab"]):focus-visible { outline: 3px solid #1b806d; outline-offset: 2px; }
  .ov-workspace-row > .column { min-width: 0; }
  @media (max-width: 1024px) {
      .ov-workspace-row { flex-wrap: wrap !important; }
      .ov-workspace-row > .column { flex: 1 1 100% !important; min-width: 100% !important; }
  }
  @media (max-width: 1024px) { .ov-header { grid-template-columns: 1fr auto; } }
  @media (max-width: 820px) { .ov-header { grid-template-columns: 1fr; } }
  ```

- [x] Style the existing three Gradio tabs as the primary task navigation; keep their Vietnamese names and click behavior.
- [x] Add `elem_classes="ov-workspace-row"` to the three existing `gr.Row` instances, then use the CSS above to preserve two columns on desktop and stack before the panels become cramped.
- [x] Use the open local app at `http://127.0.0.1:8001/` to inspect 1440px and 1024px views. Confirm the compact header, selected tab, visible focus, no horizontal overflow, and no missing tab content.

### Task 2: Refine voice-cloning input and output hierarchy

**Files:**
- Modify: `omnivoice/cli/demo.py` (voice-clone tab controls and result column)

**Interfaces:**
- Consumes: the shared workspace frame from Task 1 and the existing `_clone_fn`/`vc_btn.click` contract.
- Produces: the same clone inputs, `vc_audio`, `vc_status`, and callback outputs, arranged as an ordered work area.

- [x] Reorder the form visually to: target text, reference audio, reference-text guidance/input, language, optional voice instruction, and generation settings.
- [x] Keep the 3–10 second guidance adjacent to the reference-audio component; retain the `asr_enabled`-dependent helper so users know when reference text is required.
- [x] Give the output column a clear heading, a short empty-state instruction before generation, and grouped generated audio/status controls. Initialize status with truthful guidance only; keep callback success/error messages unchanged.

  Initialize only the presentation text; keep the callback's returned status contract intact:

  ```python
  vc_status = gr.Textbox(
      label="Trạng thái",
      lines=2,
      value="Chưa có âm thanh được tạo.",
  )
  ```

- [x] Inspect the clone tab at 1440px and 1024px. Confirm upload remains identifiable, text areas stay within the form column, and audio/status remain easy to find.
- [x] Confirm the tab still passes the same values, in the same order, to `_clone_fn` and that `_clone_fn` still returns the two values expected by `vc_btn.click`.

### Task 3: Refine voice-design controls and shared result treatment

**Files:**
- Modify: `omnivoice/cli/demo.py` (voice-design tab controls and result column)

**Interfaces:**
- Consumes: the shared workspace frame from Task 1 and existing `_design_fn`, `_build_instruct`, and `vd_btn.click` contracts.
- Produces: unchanged language/attribute controls and generated output, with a compact attribute section and the same output hierarchy as voice cloning.

- [x] Place target text and language first; group the existing voice attribute dropdowns in a scannable section without changing choice labels or model values.

  Keep each component in `vd_groups`, in the same category order, and place two dropdowns on each desktop row inside a labeled visual group:

  ```python
  vd_groups = []
  _attribute_items = list(_CATEGORIES.items())
  with gr.Group(elem_classes="ov-attribute-group"):
      gr.Markdown("### Đặc điểm chất giọng")
      for _start in range(0, len(_attribute_items), 2):
          with gr.Row(elem_classes="ov-attribute-row"):
              for _cat, _choices in _attribute_items[_start : _start + 2]:
                  vd_groups.append(
                      gr.Dropdown(
                          label=_cat,
                          choices=[("Tự động", "")] + _choices,
                          value="",
                          info=_ATTR_INFO.get(_cat),
                      )
                  )
  ```

- [x] Keep generation tuning in the existing collapsed accordion and make its advanced status visually clear without duplicating settings.
- [x] Apply the same output heading, truthful initial status, audio placement, and spacing used by the clone tab.
- [x] Inspect the design tab at 1440px and 1024px. Confirm all attribute groups fit without overlap, the language list remains searchable, and the right-side audio result is legible.
- [x] Confirm the attribute values still flow unchanged through `_build_instruct` into `_design_fn` and the outputs remain `[vd_audio, vd_status]`.

### Task 4: Refine the media-transcription workspace

**Files:**
- Modify: `omnivoice/cli/demo.py` (transcription tab controls and output column)

**Interfaces:**
- Consumes: the shared workspace frame from Task 1 and the existing `transcribe_uploaded_media` callback.
- Produces: unchanged media/model/language/task/device inputs, transcript, metadata, and TXT/SRT download outputs in a clear desktop layout.

- [x] Arrange controls in task order: media upload, Whisper model, language, transcription mode, device, and submit button.
- [x] Make the transcript the largest output element; group status, metadata, transcript, and TXT/SRT downloads in that order.
- [x] Initialize the status with concise next-step guidance; preserve callback return values and exception handling.

  Initialize the status textbox rather than adding static text that would remain after transcription:

  ```python
  stt_status = gr.Textbox(
      label="Trạng thái",
      value="Chọn tệp rồi bấm Chuyển thành văn bản để bắt đầu.",
      interactive=False,
  )
  stt_text = gr.Textbox(label="Transcript", lines=16, interactive=False)
  ```

- [x] Inspect the transcription tab at 1440px and 1024px using a long transcript sample or representative content. Confirm transcript remains readable, downloads are visible, and no output clips or forces horizontal scrolling.
- [x] Confirm the submit callback retains the same five inputs and five outputs.

### Task 5: Complete visual and interaction review

**Files:**
- Modify if needed: `omnivoice/cli/demo.py` (CSS and presentation only)

**Interfaces:**
- Consumes: the completed three-tab page from Tasks 1–4.
- Produces: a polished, usable desktop UI with a narrow-window fallback and unchanged application behavior.

- [x] Review all three tabs at 1440px and 1024px, then narrow the browser below the layout breakpoint. Confirm two columns at desktop widths, stacked order when narrow, readable controls, and zero horizontal overflow.
- [x] Tab through navigation, text inputs, dropdowns, accordions, upload controls, buttons, and audio controls. Confirm focus is visible and the workflow can be navigated without a mouse.
- [x] Check empty and populated output states; confirm no fake generation/transcription result is shown before submission and existing success/error feedback is legible.
- [x] Review the final diff against the approved spec. Confirm only the presentation/layout changed in `demo.py`; preserve transcription and generation callback signatures, model values, queue wiring, and unrelated working-tree changes.
- [x] Restart the local demo only after editing is complete and inspect the final live page at `http://127.0.0.1:8001/`.

## Execution Notes

- All UI tasks touch the same existing Python file and depend on shared CSS/classes, so implement them sequentially in this order.
- The current checkout already has user changes to the target file and related STT files. Do not reset, overwrite, stage, or commit those existing changes as part of this plan.
- The plan intentionally relies on browser-based visual review specified in the approved design. It does not add automated tests or change backend behavior.
- The app uses `fill_width=True`, then explicitly sets `.gradio-container` to `calc(100vw - 24px)` with a 1440px cap; visual review showed that the maximum width alone did not use the desktop viewport.
