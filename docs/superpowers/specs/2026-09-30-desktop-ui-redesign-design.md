# OmniVoice Desktop UI Redesign

## Goal

Redesign the existing Vietnamese OmniVoice Gradio interface as a clear, polished desktop voice-production workspace. The main success criterion is that users can choose one of the three existing tasks, understand the required inputs, run it, and find the result without losing visual context or scanning through a large promotional header.

## Design Direction

Use a compact studio layout that fits wide desktop screens first and remains usable in a narrow browser window. Keep the existing restrained teal identity, improve contrast and hierarchy, and use spacing and borders to distinguish work areas without adding decorative effects that compete with audio and text content.

### Shared workspace frame

- Replace the oversized hero with a compact header containing OmniVoice identity, a short purpose statement, and a small local-model readiness indicator where its state can be represented truthfully.
- Keep the three tasks easy to switch between: voice cloning, voice design, and media transcription. Use clear Vietnamese names and a visible selected state.
- Use the available desktop width deliberately, with a centered maximum content width around 1440px and consistent spacing.
- On wide screens, place the task form on the left and the output/status area on the right. Keep the output region visually stable so it remains easy to locate between requests.
- At narrower widths, stack the form and output while preserving their order and readable control widths.

### Voice cloning

- Organize controls in the order users need them: target text, reference audio, reference transcript, language, then optional voice and generation settings.
- Make the reference-audio area visually distinct and explain the recommended 3–10 second sample near the upload control.
- Keep advanced generation settings collapsed by default.
- Present generated audio and status together in the right output area, with an explicit empty state before generation and clear success or error feedback afterward.

### Voice design

- Put target text and language first.
- Group voice attributes into a compact, scannable section and keep generation tuning under the existing advanced accordion.
- Preserve the current supported attribute choices and their model values.
- Reuse the same output area and feedback patterns as voice cloning so both generation workflows feel consistent.

### Media transcription

- Put file upload first, followed by model, language, task, and device options.
- Give transcript output the largest area in the result column; keep metadata and TXT/SRT downloads grouped with it.
- Show a clear initial state before a file is submitted, and make completion or transcription errors easy to identify.

## Interaction and Accessibility

- Preserve keyboard access and visible focus for tabs, buttons, dropdowns, accordions, file inputs, and audio controls.
- Keep visible labels and helper text for all inputs; do not communicate required steps through color alone.
- Use strong text and control contrast on both the application background and panels.
- Preserve feedback states already returned by the backend and make long-running action states visually apparent using Gradio's supported queue behavior.
- Avoid hover-only instructions and ensure uploaded files remain identifiable after selection.

## Technical Approach

- Implement the redesign inside the existing Gradio interface in `omnivoice/cli/demo.py`.
- Use Gradio components, existing callbacks, CSS classes, and theme settings. Do not add a separate frontend framework or change model-serving architecture.
- Reuse the current input/output controls and callback wiring wherever possible. The visual changes must preserve existing voice generation, transcription, downloads, and queue behavior.
- Keep the current Vietnamese copy and supported option values, improving wording only where needed for task hierarchy and guidance.

## Scope

### Included

- Shared header, navigation hierarchy, desktop layout, panel styling, and responsive fallback.
- Visual treatment and ordering of inputs, advanced options, output, empty state, success state, and error state across all three existing tasks.
- Browser-based visual review at desktop widths and a narrower fallback width.

### Excluded

- Changes to model loading, inference, transcription, audio processing, file formats, or callback contracts.
- New product capabilities, user accounts, persistent projects/history, and public hosting.
- Replacing Gradio with a separate frontend/backend application.

## Acceptance Criteria

1. The three workflows remain available with all current input controls and outputs.
2. At a 1440px-wide viewport, the interface uses a balanced two-column workspace without an oversized hero consuming the first screen.
3. The active task is visually obvious, and each task's controls appear in a logical top-to-bottom order.
4. Output, transcript, status, and downloadable files are easy to locate and remain legible.
5. At narrower widths, the form and output stack without horizontal overflow or clipped controls.
6. Keyboard focus remains visible and text/control contrast is sufficient for the light teal-and-neutral palette.
7. Existing callbacks and model-facing values are unchanged.

## Assumptions

- The primary use case is running OmniVoice locally in a desktop browser.
- The approved visual direction is a compact desktop voice-production workspace using the current teal identity.
- The current working tree contains pre-existing changes to `omnivoice/cli/demo.py` and speech-to-text support. Those functional changes are user work and must be preserved.
