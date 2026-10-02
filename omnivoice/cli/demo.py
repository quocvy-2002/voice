#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import ctypes
import html
import logging
import os
import uuid
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch
from babel import Locale

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.stt.transcriber import TranscriptionCancelled, whisper_transcriber
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_IDS
from omnivoice.utils.task_control import RunControl, RunRegistry, RunSnapshot, TaskCancelled


# ---------------------------------------------------------------------------
# Vietnamese display names for all supported language codes.
# ---------------------------------------------------------------------------
_VI_LANGUAGE_NAMES = Locale.parse("vi").languages
_COMMON_LANGUAGE_IDS = ("vi", "en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ru")


def _language_label(code: str) -> str:
    name = _VI_LANGUAGE_NAMES.get(code)
    if not name or any("\u4e00" <= char <= "\u9fff" for char in name):
        return f"Ngôn ngữ mã {code.upper()}"
    return name[0].upper() + name[1:]


_LANGUAGE_CHOICES = [(f"{_language_label(code)} · {code}", code) for code in LANG_IDS]
_ALL_LANGUAGES = [("Tự động nhận diện", "")] + [
    choice for code in _COMMON_LANGUAGE_IDS for choice in _LANGUAGE_CHOICES if choice[1] == code
] + sorted(
    (choice for choice in _LANGUAGE_CHOICES if choice[1] not in _COMMON_LANGUAGE_IDS),
    key=lambda choice: choice[0],
)


def _stt_cuda_ready() -> bool:
    """Check the native libraries required by Faster-Whisper before offering CUDA."""
    libraries = (
        ("cublas64_12.dll", "cudnn64_9.dll")
        if os.name == "nt"
        else ("libcublas.so.12", "libcudnn.so.9")
    )
    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    try:
        for library in libraries:
            loader(library)
    except OSError:
        return False
    return True


def _progress_markup(snapshot: RunSnapshot | None) -> str:
    if snapshot is None:
        state, fraction, stage = "idle", None, "Chưa bắt đầu."
    else:
        state, fraction, stage = snapshot.state, snapshot.fraction, snapshot.stage
    percent = None if fraction is None else round(fraction * 100)
    state_label = {
        "idle": "Chưa chạy",
        "preparing": "Đang chuẩn bị",
        "running": "Đang xử lý",
        "cancelling": "Đang dừng",
        "publishing": "Đang hoàn tất",
        "cancelled": "Đã dừng",
        "completed": "Hoàn tất",
        "failed": "Có lỗi",
    }[state]
    value = (
        '<progress max="100" role="progressbar" aria-label="Tiến độ xử lý"></progress>'
        if percent is None
        else f'<progress max="100" value="{percent}" role="progressbar" '
        f'aria-label="Tiến độ xử lý" aria-valuenow="{percent}"></progress>'
    )
    amount = "Đang xác định tiến độ" if percent is None else f"{percent}%"
    return (
        '<div class="ov-task-progress" aria-live="polite">'
        f'<div class="ov-task-progress-head"><strong>{html.escape(state_label)}</strong>'
        f'<span>{amount}</span></div>{value}'
        f'<p>{html.escape(stage)}</p></div>'
    )


def transcribe_uploaded_media(media_path, model_size, language, task, device, control=None):
    """Run local STT without loading Faster Whisper until the user submits."""
    if not media_path:
        return "Hãy chọn một tệp video hoặc âm thanh.", "", None, None, ""
    if device == "cuda" and not _stt_cuda_ready():
        return (
            "CUDA chưa sẵn sàng: thiếu thư viện cuBLAS 12 hoặc cuDNN 9. "
            "Hãy chọn CPU để chuyển đổi, hoặc cài thư viện CUDA rồi khởi động lại ứng dụng.",
            "", None, None, "",
        )
    try:
        options = dict(
            media_path=media_path,
            model_size=model_size,
            language=language or None,
            task=task,
            device=device,
        )
        if control is not None:
            options["progress_callback"] = control.report
            options["check_cancelled"] = control.check_cancelled
        result = whisper_transcriber.transcribe_media(**options)
    except TranscriptionCancelled as exc:
        metadata = (
            f"Ngôn ngữ: {exc.language or 'không xác định'} · "
            f"Thời lượng: {exc.duration or 0:.1f}s"
        )
        return "Đã dừng. Đây là bản chép lời một phần.", exc.text, None, None, metadata
    except (RuntimeError, ValueError) as exc:
        if device == "cuda" and ("cublas" in str(exc).lower() or "cudnn" in str(exc).lower()):
            return (
                "CUDA không tải được cuBLAS/cuDNN. Hãy chọn CPU hoặc kiểm tra bộ thư viện "
                "CUDA 12 và cuDNN 9, rồi khởi động lại ứng dụng.",
                "", None, None, "",
            )
        return f"Không thể chuyển đổi: {exc}", "", None, None, ""
    metadata = (
        f"Ngôn ngữ: {result.language or 'không xác định'} · "
        f"Thời lượng: {result.duration or 0:.1f}s"
    )
    return "Hoàn tất chuyển đổi.", result.text, result.txt_path, result.srt_path, metadata


# ---------------------------------------------------------------------------
# Vietnamese labels with model-compatible instruction values.
# ---------------------------------------------------------------------------
_CATEGORIES = {
    "Giới tính": [("Nam", "male"), ("Nữ", "female")],
    "Độ tuổi": [
        ("Trẻ em", "child"),
        ("Thiếu niên", "teenager"),
        ("Thanh niên", "young adult"),
        ("Trung niên", "middle-aged"),
        ("Cao tuổi", "elderly"),
    ],
    "Cao độ": [
        ("Rất trầm", "very low pitch"),
        ("Trầm", "low pitch"),
        ("Vừa", "moderate pitch"),
        ("Cao", "high pitch"),
        ("Rất cao", "very high pitch"),
    ],
    "Phong cách": [("Thì thầm", "whisper")],
    "Giọng vùng miền khi đọc tiếng Anh": [
        ("Mỹ", "american accent"),
        ("Úc", "australian accent"),
        ("Anh", "british accent"),
        ("Trung Quốc", "chinese accent"),
        ("Canada", "canadian accent"),
        ("Ấn Độ", "indian accent"),
        ("Hàn Quốc", "korean accent"),
        ("Bồ Đào Nha", "portuguese accent"),
        ("Nga", "russian accent"),
        ("Nhật Bản", "japanese accent"),
    ],
    "Phương ngữ khi đọc tiếng Trung": [
        ("Hà Nam", "河南话"),
        ("Thiểm Tây", "陕西话"),
        ("Tứ Xuyên", "四川话"),
        ("Quý Châu", "贵州话"),
        ("Vân Nam", "云南话"),
        ("Quế Lâm", "桂林话"),
        ("Tế Nam", "济南话"),
        ("Thạch Gia Trang", "石家庄话"),
        ("Cam Túc", "甘肃话"),
        ("Ninh Hạ", "宁夏话"),
        ("Thanh Đảo", "青岛话"),
        ("Đông Bắc", "东北话"),
    ],
}

_ATTR_INFO = {
    "Giọng vùng miền khi đọc tiếng Anh": "Chỉ có tác dụng với văn bản tiếng Anh.",
    "Phương ngữ khi đọc tiếng Trung": "Chỉ có tác dụng với văn bản tiếng Trung.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
    asr_enabled: bool = True,
) -> gr.Blocks:
    sampling_rate = model.sampling_rate

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
        control: RunControl | None = None,
    ):
        if not text or not text.strip():
            return None, "Hãy nhập văn bản cần đọc."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language or None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            if control is not None:
                control.report(None, "Đang chuẩn bị tạo giọng…")
                control.check_cancelled()
            if mode == "clone":
                if not ref_audio:
                    return None, "Hãy tải lên âm thanh giọng mẫu."
                if not asr_enabled and not ref_text:
                    return None, "Hãy nhập lời thoại của bản ghi âm mẫu."
                kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
                if control is not None:
                    control.check_cancelled()
            if control is not None:
                kw["progress_callback"] = control.report
                kw["check_cancelled"] = control.check_cancelled
            audio = model.generate(**kw)
            if control is not None:
                control.check_cancelled()
        except TaskCancelled:
            return None, "Đã dừng tạo giọng."
        except ValueError as exc:
            logging.info("Invalid voice generation input: %s", exc)
            return None, f"Đầu vào không hợp lệ: {exc}"
        except Exception:
            logging.exception("Audio generation failed")
            return None, "Không thể tạo âm thanh. Hãy kiểm tra đầu vào và thử lại."

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Đã tạo âm thanh."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    gradio-app { background: #f3f5f3 !important; }
    body { background: #f3f5f3 !important; }
    .gradio-container {
        --background-fill-primary: #f3f5f3;
        --background-fill-secondary: #ffffff;
        --body-background-fill: #f3f5f3;
        --body-text-color: #1d302e;
        --body-text-color-subdued: #60726c;
        --block-background-fill: #ffffff;
        --block-label-background-fill: #e4f1eb;
        --block-label-text-color: #1c574b;
        --input-background-fill: #ffffff;
        --input-background-fill-focus: #ffffff;
        --input-border-color: #cddbd3;
        --border-color-primary: #dbe5df;
        --button-primary-background-fill: #0f6959;
        --button-primary-text-color: #ffffff;
        width: calc(100vw - 24px) !important;
        max-width: 1440px !important;
        margin: 0 auto !important;
        padding: 24px 28px 40px !important;
        color: #1d302e !important;
    }
    .ov-header {
        display: grid;
        grid-template-columns: auto 1fr auto;
        align-items: center;
        gap: 24px;
        margin-bottom: 22px;
        padding: 12px 2px 20px;
        border-bottom: 1px solid #dbe5df;
    }
    .ov-brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .ov-brand-mark {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: #155f52;
        color: #fff;
        font-size: 15px;
        font-weight: 800;
        letter-spacing: -.04em;
    }
    .ov-brand-copy { display: grid; gap: 2px; }
    .ov-brand-copy strong { color: #173b35; font-size: 17px; letter-spacing: -.02em; }
    .ov-brand-copy small { color: #60726c; font-size: 12px; }
    .ov-header > p { margin: 0; color: #536963; font-size: 14px; line-height: 1.5; }
    .ov-ready {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border: 1px solid #c9e3d8;
        border-radius: 999px;
        background: #e9f5ef;
        color: #1c5b49;
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
    }
    .ov-ready::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #2f8a67;
    }
    .ov-tabs { margin-top: 0; }
    .ov-tabs [role="tab"] {
        min-height: 44px;
        padding: 0 18px;
        border-radius: 10px;
        color: #52655f;
        font-size: 14px;
        font-weight: 650;
    }
    .ov-tabs [role="tab"][aria-selected="true"] {
        background: #fff;
        color: #145c50;
        box-shadow: inset 0 0 0 1px #cbded5;
    }
    .ov-panel {
        min-width: 0;
        padding: 24px;
        border: 1px solid #dbe5df;
        border-radius: 14px;
        background: #fff;
    }
    .ov-panel-title h2 { margin: 0 0 6px; color: #183731; font-size: 19px; }
    .ov-panel-title p { margin: 0 0 18px; color: #60726c; font-size: 14px; line-height: 1.5; }
    .ov-panel .form { gap: 16px; background: transparent !important; }
    .ov-panel label { color: #263d37; font-size: 13px; font-weight: 650; }
    .ov-panel input, .ov-panel textarea { border-color: #cddbd3 !important; }
    .ov-panel input:focus, .ov-panel textarea:focus { border-color: #287d6c !important; }
    .ov-panel .info { color: #60726c; font-size: 12px; }
    .ov-workspace-row > .column { min-width: 0; }
    .ov-result-panel { position: sticky; top: 16px; align-self: flex-start; }
    .ov-attribute-title h3 { margin: 4px 0 10px; color: #28463f; font-size: 15px; }
    .ov-attribute-row { align-items: flex-start; gap: 16px; }
    .ov-attribute-row > * { flex: 1 1 0; min-width: 0; }
    .ov-option-row { align-items: flex-start; gap: 14px; }
    .ov-option-row > * { flex: 1 1 0; min-width: 0; }
    .ov-submit { margin-top: 8px; min-height: 48px; font-weight: 700 !important; }
    .ov-stop { margin-top: 8px; min-height: 48px; font-weight: 700 !important; }
    .ov-task-progress { padding: 14px 16px; border: 1px solid #dbe5df; border-radius: 10px; background: #f5faf7; }
    .ov-task-progress-head { display: flex; justify-content: space-between; gap: 12px; color: #244b41; font-size: 13px; }
    .ov-task-progress progress { display: block; width: 100%; height: 12px; margin: 10px 0 8px; accent-color: #0f6959; }
    .ov-task-progress p { margin: 0; color: #526b60; font-size: 12px; }
    .ov-hint { margin: -4px 0 0; color: #536963; font-size: 13px; line-height: 1.5; }
    .ov-footer { margin-top: 20px; color: #53645e; font-size: 12px; }
    :is(button, input, textarea, [role="tab"], [role="combobox"]):focus-visible {
        outline: 3px solid #1b806d !important;
        outline-offset: 2px !important;
    }
    .ov-panel button.upload-container [data-testid="upload-text"] { font-size: 0; }
    .ov-panel button.upload-container [data-testid="upload-text"]::after {
        content: "Kéo thả tệp hoặc bấm để chọn tệp";
        display: block;
        margin-top: 8px;
        font-size: 14px;
        font-weight: 600;
        color: #425e53;
    }
    .ov-panel button.upload-container .or { display: none; }
    footer[aria-label="Gradio footer navigation"] { display: none !important; }
    .compact-audio audio { height: 64px !important; }
    .compact-audio .waveform { min-height: 72px !important; }
    @media (max-width: 1024px) {
        .ov-workspace-row { flex-wrap: wrap !important; }
        .ov-workspace-row > .column { flex: 1 1 100% !important; min-width: 100% !important; }
        .ov-option-row > * { min-width: 240px; }
        .ov-result-panel { position: static; }
        .ov-header { grid-template-columns: 1fr auto; gap: 14px 20px; }
        .ov-header > p { grid-column: 1 / -1; grid-row: 2; }
    }
    @media (max-width: 820px) {
        .gradio-container { padding: 16px 14px 32px !important; }
        .ov-header { grid-template-columns: 1fr; gap: 12px; margin-bottom: 16px; padding-bottom: 16px; }
        .ov-header > p { grid-column: auto; grid-row: auto; }
        .ov-panel { padding: 18px; }
        .ov-tabs [role="tab"] { padding: 0 12px; font-size: 13px; }
    }
    @media (max-width: 620px) {
        .ov-tabs [role="tablist"] { flex-wrap: wrap; }
        .ov-option-row { flex-wrap: wrap !important; }
        .ov-option-row > * { flex: 1 1 100% !important; min-width: 100% !important; }
        .ov-attribute-row { flex-wrap: wrap !important; }
        .ov-attribute-row > * { flex: 1 1 100% !important; min-width: 100% !important; }
    }
    """

    # Reusable: language dropdown component
    def _lang_dropdown():
        return gr.Dropdown(
            label="Ngôn ngữ",
            choices=_ALL_LANGUAGES,
            value="",
            allow_custom_value=False,
            interactive=True,
            info="Để tự động nếu không muốn chọn ngôn ngữ. Có thể tìm theo mã ngôn ngữ.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Tùy chỉnh âm thanh", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="Tốc độ đọc",
                info="1,0 là bình thường; số lớn hơn đọc nhanh hơn. Bỏ qua khi đặt thời lượng.",
            )
            du = gr.Number(
                value=None,
                label="Thời lượng cố định (giây)",
                info="Để trống nếu muốn dùng tốc độ đọc ở trên.",
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="Số bước tạo giọng",
                info="Mặc định 32. Ít bước hơn sẽ nhanh hơn; nhiều bước hơn có thể cải thiện chất lượng.",
            )
            dn = gr.Checkbox(
                label="Giảm nhiễu",
                value=True,
                info="Bật mặc định để âm thanh sạch hơn.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Mức bám theo chỉ dẫn",
                info="Mặc định 2,0.",
            )
            pp = gr.Checkbox(
                label="Làm sạch âm thanh mẫu",
                value=True,
                info="Cắt khoảng lặng và chuẩn hóa lời thoại trong bản ghi âm mẫu.",
            )
            po = gr.Checkbox(
                label="Cắt khoảng lặng ở kết quả",
                value=True,
                info="Loại bỏ những khoảng lặng dài trong âm thanh đã tạo.",
            )
        return ns, gs, dn, sp, du, pp, po

    registry = RunRegistry()

    def _begin(workflow: str, session_token: str):
        control = registry.begin(session_token, workflow)
        if control is None:
            snapshot = registry.snapshot(session_token, workflow)
            return snapshot.run_id, gr.update(interactive=False), gr.update(interactive=True)
        return control.run_id, gr.update(interactive=False), gr.update(interactive=True)

    def _stop(workflow: str, session_token: str):
        snapshot = registry.snapshot(session_token, workflow)
        if snapshot is None or not registry.request_stop(
            session_token, workflow, snapshot.run_id
        ):
            return gr.update(interactive=False), gr.skip()
        if registry.snapshot(session_token, workflow).state == "cancelled":
            return gr.update(interactive=False), "Đã dừng. Bạn có thể chạy lại."
        return gr.update(interactive=False), "Đang dừng, vui lòng chờ điểm xử lý an toàn…"

    def _stop_with_timer(workflow: str, session_token: str):
        return (*_stop(workflow, session_token), gr.update(active=False))

    def _poll(workflow: str, session_token: str):
        snapshot = registry.snapshot(session_token, workflow)
        active = snapshot is not None and snapshot.state in {"preparing", "running", "cancelling", "publishing"}
        can_stop = snapshot is not None and snapshot.state in {"preparing", "running"}
        return (
            _progress_markup(snapshot),
            gr.update(interactive=not active),
            gr.update(interactive=can_stop),
        )

    def _run_voice(workflow, values, run_id, session_token, mode, ref_text=None):
        control = registry.get(session_token, workflow, run_id)
        if control is None or not control.claim_execution():
            return gr.skip(), gr.skip(), run_id, ""
        try:
            control.check_cancelled()
            if generate_fn is None:
                audio, status = _gen_core(
                    *values, mode=mode, ref_text=ref_text, control=control
                )
            else:
                control.report(None, "Đang tạo giọng…")
                options = {"mode": mode}
                if mode == "clone":
                    options["ref_text"] = ref_text
                audio, status = _gen(*values, **options)
                control.check_cancelled()
            if control.cancel_event.is_set():
                audio, status = None, "Đã dừng tạo giọng."
                outcome = "cancelled"
            else:
                outcome = "completed" if audio is not None else "failed"
        except TaskCancelled:
            audio, status, outcome = None, "Đã dừng tạo giọng.", "cancelled"
        except Exception:
            logging.exception("Audio generation failed")
            audio, status, outcome = None, "Không thể tạo âm thanh. Hãy thử lại.", "failed"
        outcome = control.prepare_delivery(outcome)
        if registry.get(session_token, workflow, run_id) is not control:
            return gr.skip(), gr.skip(), run_id, ""
        if outcome == "cancelled":
            return None, "Đã dừng tạo giọng.", run_id, outcome
        return audio, status, run_id, outcome

    def _run_transcription(media, model_size, language, task, device, run_id, session_token):
        control = registry.get(session_token, "transcribe", run_id)
        if control is None or not control.claim_execution():
            return (gr.skip(),) * 5 + (run_id, "")
        try:
            control.check_cancelled()
            outputs = transcribe_uploaded_media(
                media, model_size, language, task, device, control=control
            )
            if control.cancel_event.is_set():
                outputs = ("Đã dừng. Đây là bản chép lời một phần.", outputs[1], None, None, outputs[4])
                outcome = "cancelled"
            else:
                outcome = "completed" if outputs[2] is not None else "failed"
        except TaskCancelled:
            outputs, outcome = ("Đã dừng chuyển đổi.", "", None, None, ""), "cancelled"
        except Exception:
            logging.exception("Transcription failed")
            outputs, outcome = ("Không thể chuyển đổi. Hãy thử lại.", "", None, None, ""), "failed"
        outcome = control.prepare_delivery(outcome)
        if registry.get(session_token, "transcribe", run_id) is not control:
            return (gr.skip(),) * 5 + (run_id, "")
        if outcome == "cancelled":
            outputs = ("Đã dừng. Đây là bản chép lời một phần.", outputs[1], None, None, outputs[4])
        return (*outputs, run_id, outcome)

    def _finish_delivery(workflow: str, run_id: str, outcome: str, session_token: str):
        if outcome in {"completed", "cancelled", "failed"}:
            registry.finish(session_token, workflow, run_id, outcome)
        snapshot = registry.snapshot(session_token, workflow)
        active = snapshot is not None and snapshot.state in {
            "preparing", "running", "cancelling", "publishing"
        }
        return gr.update(active=active)

    with gr.Blocks(
        theme=theme,
        css=css,
        title="OmniVoice · Phòng thu giọng nói",
        fill_width=True,
    ) as demo:
        session_token = gr.State(value=lambda: uuid.uuid4().hex, delete_callback=registry.drop_session)
        # Browser values are captured with each queued request, so a cancelled
        # request cannot accidentally claim a newer run in the same session.
        vc_run_id = gr.Textbox(value="", visible=False)
        vd_run_id = gr.Textbox(value="", visible=False)
        stt_run_id = gr.Textbox(value="", visible=False)
        vc_finished_id, vc_outcome = gr.State(""), gr.State("")
        vd_finished_id, vd_outcome = gr.State(""), gr.State("")
        stt_finished_id, stt_outcome = gr.State(""), gr.State("")
        gr.HTML(
            """
<header class="ov-header">
  <div class="ov-brand">
    <span class="ov-brand-mark" aria-hidden="true">OV</span>
    <div class="ov-brand-copy">
      <strong>OmniVoice</strong>
      <small>Phòng thu giọng nói</small>
    </div>
  </div>
  <p>Nhân bản giọng nói, thiết kế chất giọng và chép lời media.</p>
  <span class="ov-ready" role="status">Sẵn sàng</span>
</header>
"""
        )

        with gr.Tabs(elem_classes="ov-tabs"):
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Sao chép giọng"):
                with gr.Row(elem_classes="ov-workspace-row"):
                    with gr.Column(scale=6, elem_classes="ov-panel"):
                        gr.Markdown(
                            "## Tạo giọng từ bản ghi âm\n"
                            "Tải lên một đoạn âm thanh mẫu để tạo lời nói mới bằng chất giọng đó.",
                            elem_classes="ov-panel-title",
                        )
                        vc_text = gr.Textbox(
                            label="Văn bản cần đọc",
                            lines=4,
                            placeholder="Nhập nội dung bạn muốn chuyển thành giọng nói...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Âm thanh giọng mẫu",
                            type="filepath",
                            sources=["upload"],
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "Nên dùng bản ghi rõ tiếng, dài khoảng 3–10 giây.",
                            elem_classes="ov-hint",
                        )
                        vc_ref_text = gr.Textbox(
                            label="Lời thoại trong âm thanh mẫu",
                            lines=2,
                            placeholder="Nhập chính xác những gì được nói trong bản ghi âm...",
                            info=(
                                "Có thể để trống để hệ thống tự nhận dạng lời thoại."
                                if asr_enabled
                                else "Cần nhập lời thoại vì nhận dạng tự động đang tắt."
                            ),
                        )
                        vc_lang = _lang_dropdown()
                        with gr.Accordion("Chỉ dẫn thêm về giọng (nâng cao)", open=False):
                            vc_instruct = gr.Textbox(
                                label="Mô tả bổ sung",
                                lines=2,
                                info="Mô hình hiểu chỉ dẫn tự do bằng tiếng Anh hoặc tiếng Trung.",
                            )
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        with gr.Row(elem_classes="ov-option-row"):
                            vc_btn = gr.Button(
                                "Tạo giọng nói", variant="primary", elem_classes="ov-submit"
                            )
                            vc_stop = gr.Button(
                                "Dừng", interactive=False, elem_classes="ov-stop"
                            )
                    with gr.Column(scale=4, elem_classes=["ov-panel", "ov-result-panel"]):
                        gr.Markdown(
                            "## Kết quả âm thanh\nÂm thanh sẽ xuất hiện tại đây sau khi tạo.",
                            elem_classes="ov-panel-title",
                        )
                        vc_audio = gr.Audio(
                            label="Âm thanh đã tạo",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(
                            label="Trạng thái",
                            lines=2,
                            value="Chưa có âm thanh được tạo.",
                        )
                        vc_progress = gr.HTML(value=_progress_markup(None))
                vc_timer = gr.Timer(value=0.7, active=False)

                def _clone_fn(
                    text, lang, ref_aud, ref_text, instruct, ns, gs, dn, sp, du, pp, po,
                    run_id, token,
                ):
                    return _run_voice(
                        "clone",
                        (text, lang, ref_aud, instruct, ns, gs, dn, sp, du, pp, po),
                        run_id, token, "clone", ref_text=ref_text or None,
                    )

                def _begin_clone(token):
                    return (
                        *_begin("clone", token),
                        None,
                        "Đang chuẩn bị tạo giọng…",
                        gr.update(active=True),
                    )

                vc_start = vc_btn.click(
                    _begin_clone,
                    inputs=[session_token],
                    outputs=[vc_run_id, vc_btn, vc_stop, vc_audio, vc_status, vc_timer],
                    queue=False,
                    show_progress="hidden",
                )
                vc_worker = vc_start.then(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                        vc_run_id,
                        session_token,
                    ],
                    outputs=[vc_audio, vc_status, vc_finished_id, vc_outcome],
                    concurrency_id="omnivoice_tts",
                    concurrency_limit=1,
                )
                vc_worker.then(
                    lambda run_id, outcome, token: _finish_delivery("clone", run_id, outcome, token),
                    inputs=[vc_finished_id, vc_outcome, session_token],
                    outputs=[vc_timer], queue=False, show_progress="hidden",
                )
                vc_stop.click(
                    lambda token: _stop_with_timer("clone", token),
                    inputs=[session_token],
                    outputs=[vc_stop, vc_status, vc_timer],
                    queue=False,
                    show_progress="hidden",
                    cancels=[vc_worker],
                )
                vc_timer.tick(
                    lambda token: _poll("clone", token),
                    inputs=[session_token],
                    outputs=[vc_progress, vc_btn, vc_stop],
                    queue=False,
                    show_progress="hidden",
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Thiết kế giọng"):
                with gr.Row(elem_classes="ov-workspace-row"):
                    with gr.Column(scale=6, elem_classes="ov-panel"):
                        gr.Markdown(
                            "## Tự chọn chất giọng\n"
                            "Kết hợp các đặc điểm dưới đây để tạo giọng mà không cần bản ghi âm mẫu.",
                            elem_classes="ov-panel-title",
                        )
                        vd_text = gr.Textbox(
                            label="Văn bản cần đọc",
                            lines=4,
                            placeholder="Nhập nội dung bạn muốn chuyển thành giọng nói...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = ""
                        vd_groups = []
                        gr.Markdown("### Đặc điểm chất giọng", elem_classes="ov-attribute-title")
                        _attribute_items = list(_CATEGORIES.items())
                        for _start in range(0, len(_attribute_items), 2):
                            with gr.Row(elem_classes="ov-attribute-row"):
                                for _cat, _choices in _attribute_items[_start : _start + 2]:
                                    vd_groups.append(
                                        gr.Dropdown(
                                            label=_cat,
                                            choices=[("Tự động", _AUTO)] + _choices,
                                            value=_AUTO,
                                            info=_ATTR_INFO.get(_cat),
                                        )
                                    )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        with gr.Row(elem_classes="ov-option-row"):
                            vd_btn = gr.Button(
                                "Tạo giọng nói", variant="primary", elem_classes="ov-submit"
                            )
                            vd_stop = gr.Button(
                                "Dừng", interactive=False, elem_classes="ov-stop"
                            )
                    with gr.Column(scale=4, elem_classes=["ov-panel", "ov-result-panel"]):
                        gr.Markdown(
                            "## Kết quả âm thanh\nÂm thanh sẽ xuất hiện tại đây sau khi tạo.",
                            elem_classes="ov-panel-title",
                        )
                        vd_audio = gr.Audio(
                            label="Âm thanh đã tạo",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(
                            label="Trạng thái",
                            lines=2,
                            value="Chưa có âm thanh được tạo.",
                        )
                        vd_progress = gr.HTML(value=_progress_markup(None))
                vd_timer = gr.Timer(value=0.7, active=False)

                def _build_instruct(groups):
                    """Join the model values behind the Vietnamese dropdown labels."""
                    return ", ".join(g for g in groups if g) or None

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *tail):
                    groups, run_id, token = tail[:-2], tail[-2], tail[-1]
                    return _run_voice(
                        "design",
                        (text, lang, None, _build_instruct(groups), ns, gs, dn, sp, du, pp, po),
                        run_id, token, "design",
                    )

                def _begin_design(token):
                    return (
                        *_begin("design", token),
                        None,
                        "Đang chuẩn bị tạo giọng…",
                        gr.update(active=True),
                    )

                vd_start = vd_btn.click(
                    _begin_design,
                    inputs=[session_token],
                    outputs=[vd_run_id, vd_btn, vd_stop, vd_audio, vd_status, vd_timer],
                    queue=False,
                    show_progress="hidden",
                )
                vd_worker = vd_start.then(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups
                    + [vd_run_id, session_token],
                    outputs=[vd_audio, vd_status, vd_finished_id, vd_outcome],
                    concurrency_id="omnivoice_tts",
                    concurrency_limit=1,
                )
                vd_worker.then(
                    lambda run_id, outcome, token: _finish_delivery("design", run_id, outcome, token),
                    inputs=[vd_finished_id, vd_outcome, session_token],
                    outputs=[vd_timer], queue=False, show_progress="hidden",
                )
                vd_stop.click(
                    lambda token: _stop_with_timer("design", token),
                    inputs=[session_token],
                    outputs=[vd_stop, vd_status, vd_timer],
                    queue=False,
                    show_progress="hidden",
                    cancels=[vd_worker],
                )
                vd_timer.tick(
                    lambda token: _poll("design", token),
                    inputs=[session_token],
                    outputs=[vd_progress, vd_btn, vd_stop],
                    queue=False,
                    show_progress="hidden",
                )

            # ==============================================================
            # Video to text
            # ==============================================================
            with gr.TabItem("Chuyển video thành text"):
                with gr.Row(elem_classes="ov-workspace-row"):
                    with gr.Column(scale=6, elem_classes="ov-panel"):
                        gr.Markdown(
                            "## Chuyển video hoặc âm thanh thành văn bản\n"
                            "Tải tệp lên, chọn ngôn ngữ nếu cần, rồi nhận transcript kèm TXT và SRT.",
                            elem_classes="ov-panel-title",
                        )
                        stt_media = gr.File(
                            label="Video hoặc âm thanh",
                            type="filepath",
                            file_types=[
                                ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac",
                                ".mp4", ".mkv", ".mov", ".avi",
                            ],
                        )
                        with gr.Row(elem_classes="ov-option-row"):
                            stt_model = gr.Dropdown(
                                label="Mô hình Whisper",
                                choices=["tiny", "base", "small", "medium", "large-v3"],
                                value="small",
                            )
                            stt_language = gr.Dropdown(
                                label="Ngôn ngữ",
                                choices=[("Tự động nhận diện", ""), ("Tiếng Việt", "vi"), ("English", "en")],
                                value="",
                            )
                        with gr.Row(elem_classes="ov-option-row"):
                            stt_task = gr.Radio(
                                label="Chế độ",
                                choices=[("Chép lại nguyên ngữ", "transcribe"), ("Dịch sang tiếng Anh", "translate")],
                                value="transcribe",
                            )
                            stt_cuda_ready = _stt_cuda_ready()
                            stt_device = gr.Radio(
                                label="Thiết bị",
                                choices=[("CPU", "cpu")] + ([("CUDA", "cuda")] if stt_cuda_ready else []),
                                value="cpu",
                                info=(
                                    "Máy này chưa có cuBLAS 12/cuDNN 9 cho Faster-Whisper; "
                                    "hiện dùng CPU."
                                    if not stt_cuda_ready else None
                                ),
                            )
                        with gr.Row(elem_classes="ov-option-row"):
                            stt_button = gr.Button(
                                "Chuyển thành văn bản", variant="primary", elem_classes="ov-submit"
                            )
                            stt_stop = gr.Button(
                                "Dừng", interactive=False, elem_classes="ov-stop"
                            )
                    with gr.Column(scale=4, elem_classes=["ov-panel", "ov-result-panel"]):
                        gr.Markdown(
                            "## Bản chép lời\nKết quả và tệp tải xuống sẽ xuất hiện tại đây.",
                            elem_classes="ov-panel-title",
                        )
                        stt_status = gr.Textbox(
                            label="Trạng thái",
                            value="Chọn tệp rồi bấm Chuyển thành văn bản để bắt đầu.",
                            interactive=False,
                        )
                        stt_progress = gr.HTML(value=_progress_markup(None))
                        stt_metadata = gr.Textbox(label="Thông tin", interactive=False)
                        stt_text = gr.Textbox(label="Transcript", lines=16, interactive=False)
                        stt_txt_file = gr.File(label="Tải TXT", interactive=False)
                        stt_srt_file = gr.File(label="Tải SRT", interactive=False)
                stt_timer = gr.Timer(value=0.7, active=False)

                def _begin_transcription(token):
                    return (
                        *_begin("transcribe", token),
                        "Đang chuẩn bị chuyển đổi…", "", None, None, "",
                        gr.update(active=True),
                    )

                stt_start = stt_button.click(
                    _begin_transcription,
                    inputs=[session_token],
                    outputs=[
                        stt_run_id, stt_button, stt_stop, stt_status, stt_text,
                        stt_txt_file, stt_srt_file, stt_metadata, stt_timer,
                    ],
                    queue=False,
                    show_progress="hidden",
                )
                stt_worker = stt_start.then(
                    _run_transcription,
                    inputs=[
                        stt_media, stt_model, stt_language, stt_task, stt_device,
                        stt_run_id, session_token,
                    ],
                    outputs=[
                        stt_status, stt_text, stt_txt_file, stt_srt_file, stt_metadata,
                        stt_finished_id, stt_outcome,
                    ],
                    concurrency_limit=1,
                )
                stt_worker.then(
                    lambda run_id, outcome, token: _finish_delivery("transcribe", run_id, outcome, token),
                    inputs=[stt_finished_id, stt_outcome, session_token],
                    outputs=[stt_timer], queue=False, show_progress="hidden",
                )
                stt_stop.click(
                    lambda token: _stop_with_timer("transcribe", token),
                    inputs=[session_token],
                    outputs=[stt_stop, stt_status, stt_timer],
                    queue=False,
                    show_progress="hidden",
                    cancels=[stt_worker],
                )
                stt_timer.tick(
                    lambda token: _poll("transcribe", token),
                    inputs=[session_token],
                    outputs=[stt_progress, stt_button, stt_stop],
                    queue=False,
                    show_progress="hidden",
                )

        gr.Markdown(
            "Phát triển từ [mã nguồn OmniVoice](https://github.com/k2-fsa/OmniVoice).",
            elem_classes="ov-footer",
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint, asr_enabled=not args.no_asr)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
