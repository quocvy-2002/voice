"""Profile the existing inference path without changing the production API.

Run from the repository root with ``python -m examples.profile_inference --help``.
Stage times are exclusive, CUDA-synchronized wall times; model loading is separate.
"""

import argparse
import atexit
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import threading
import time
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt
from omnivoice.models import omnivoice as implementation
from omnivoice.utils.common import fix_random_seed, get_best_device


class StageTimer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.stack = []

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name):
        self.synchronize()
        entry = [time.perf_counter(), 0.0, name]
        self.stack.append(entry)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter() - entry[0]
            self.stack.pop()
            self.seconds[name] += elapsed - entry[1]
            self.calls[name] += 1
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            with self.stage(name):
                return function(*args, **kwargs)

        return measured


class GpuSampler:
    """Sample NVIDIA utilization during processing, excluding model loading."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.samples = []
        self.process = None
        self.error = None

    def start(self):
        if self.device.type != "cuda":
            return
        try:
            props = torch.cuda.get_device_properties(self.device)
            uuid = str(props.uuid)
            if not uuid.startswith("GPU-"):
                uuid = "GPU-" + uuid
            self.process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "-i",
                    uuid,
                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                    "--loop-ms=500",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, AttributeError) as exc:
            self.error = str(exc)
            return

        def read_samples():
            for line in self.process.stdout:
                try:
                    self.samples.append([float(v.strip()) for v in line.split(",")])
                except ValueError:
                    self.error = line.strip()

        self.reader = threading.Thread(target=read_samples, daemon=True)
        self.reader.start()
        atexit.register(self.stop)

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.reader.join(timeout=5)
            self.process.stdout.close()
            self.process = None
            atexit.unregister(self.stop)
        return {
            "interval_seconds": 0.5,
            "samples": len(self.samples),
            "mean_gpu_util_percent": float(np.mean([s[0] for s in self.samples]))
            if self.samples
            else None,
            "max_device_memory_mib": max((s[1] for s in self.samples), default=None),
            "raw_gpu_util_memory_mib_power_w": self.samples,
            "error": self.error,
        }


def model_placement(module):
    counts = defaultdict(int)
    for parameter in module.parameters():
        counts[f"{parameter.device}/{parameter.dtype}"] += parameter.numel()
    return dict(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default=None)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text-file", type=Path)
    parser.add_argument(
        "--prompt",
        type=Path,
        help="Saved VoiceClonePrompt, bypassing reference preprocessing/encoding.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON with existing OmniVoiceGenerationConfig fields.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--language")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm up on a short auto-voice request; excluded from timings.",
    )
    args = parser.parse_args()
    if args.prompt and (args.ref_audio or args.ref_text_file):
        parser.error("--prompt cannot be combined with --ref-audio/--ref-text-file")
    device = args.device or get_best_device()
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        parser.error(
            f"CUDA unavailable in PyTorch {torch.__version__} (build CUDA={torch.version.cuda})"
        )
    text = args.text_file.read_text(encoding="utf-8-sig").strip()
    if not text:
        parser.error("--text-file must contain nonempty text")
    ref_text = (
        args.ref_text_file.read_text(encoding="utf-8-sig").strip()
        if args.ref_text_file
        else None
    )
    config_values = (
        json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
    )
    unknown = (
        config_values.keys() - OmniVoiceGenerationConfig.__dataclass_fields__.keys()
    )
    if unknown:
        parser.error(f"Unknown generation parameters: {sorted(unknown)}")
    config = OmniVoiceGenerationConfig(**config_values)
    fix_random_seed(args.seed)
    started = time.perf_counter()
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16
    )
    timer = StageTimer(model.device)
    timer.synchronize()
    load_seconds = time.perf_counter() - started
    if args.warmup:
        model.generate("A short warmup.", duration=1.0)
        timer.synchronize()
    fix_random_seed(args.seed)
    placement = {
        "all_registered_parameters": model_placement(model),
        "tts_backbone": model_placement(model.llm),
        "codec": model_placement(model.audio_tokenizer),
    }
    tensor_observations = {}
    backbone_observations = {}
    forward_calls = []

    def observe(module, args, kwargs):
        forward_calls.append(1)
        if not tensor_observations:
            tensor_observations.update(
                {
                    k: {
                        "device": str(v.device),
                        "dtype": str(v.dtype),
                        "shape": list(v.shape),
                    }
                    for k, v in kwargs.items()
                    if isinstance(v, torch.Tensor)
                }
            )

    def observe_backbone(module, args, kwargs):
        if not backbone_observations:
            backbone_observations.update(
                {
                    k: {
                        "device": str(v.device),
                        "dtype": str(v.dtype),
                        "shape": list(v.shape),
                    }
                    for k, v in kwargs.items()
                    if isinstance(v, torch.Tensor)
                }
            )

    hook = model.register_forward_pre_hook(observe, with_kwargs=True)
    backbone_hook = model.llm.register_forward_pre_hook(
        observe_backbone, with_kwargs=True
    )
    sampler = GpuSampler(model.device)
    if timer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(timer.device)
    sampler.start()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def measured_silence(*args, **kwargs):
        name = (
            "reference_preprocessing"
            if any(entry[2] == "reference_preprocessing" for entry in timer.stack)
            else "audio_postprocessing"
        )
        with timer.stage(name):
            return remove_silence(*args, **kwargs)

    remove_silence = implementation.remove_silence
    with (
        patch.object(
            implementation,
            "load_audio",
            timer.wrap("reference_preprocessing", implementation.load_audio),
        ),
        patch.object(
            implementation,
            "trim_long_audio",
            timer.wrap("reference_preprocessing", implementation.trim_long_audio),
        ),
        patch.object(implementation, "remove_silence", measured_silence),
        patch.object(
            implementation,
            "chunk_text_punctuation",
            timer.wrap("text_splitting", implementation.chunk_text_punctuation),
        ),
        patch.object(
            implementation,
            "cross_fade_chunks",
            timer.wrap("audio_concatenation", implementation.cross_fade_chunks),
        ),
        patch.object(
            model,
            "create_voice_clone_prompt",
            timer.wrap("reference_preprocessing", model.create_voice_clone_prompt),
        ),
        patch.object(
            model.audio_tokenizer,
            "encode",
            timer.wrap("voice_encoding", model.audio_tokenizer.encode),
        ),
        patch.object(
            model, "transcribe", timer.wrap("reference_asr", model.transcribe)
        ),
        patch.object(
            model,
            "load_asr_model",
            timer.wrap("asr_model_loading", model.load_asr_model),
        ),
        patch.object(
            model,
            "_preprocess_all",
            timer.wrap("text_preprocessing", model._preprocess_all),
        ),
        patch.object(
            model,
            "_generate_iterative",
            timer.wrap("audio_generation", model._generate_iterative),
        ),
        patch.object(
            model.audio_tokenizer,
            "decode",
            timer.wrap("audio_decoding", model.audio_tokenizer.decode),
        ),
        patch.object(
            model,
            "_post_process_audio",
            timer.wrap("audio_postprocessing", model._post_process_audio),
        ),
    ):
        prompt = None
        if args.prompt:
            with timer.stage("prompt_loading"):
                prompt = VoiceClonePrompt.load(str(args.prompt))
        audio = model.generate(
            text,
            ref_audio=str(args.ref_audio) if args.ref_audio else None,
            ref_text=ref_text,
            voice_clone_prompt=prompt,
            duration=args.duration,
            language=args.language,
            generation_config=config,
        )[0]
        with timer.stage("file_writing"):
            sf.write(args.output, audio, model.sampling_rate)
    timer.synchronize()
    processing_seconds = time.perf_counter() - started
    gpu = sampler.stop()
    hook.remove()
    backbone_hook.remove()
    duration = len(audio) / model.sampling_rate
    report = {
        "model": args.model,
        "device": str(model.device),
        "placement": placement,
        "input_tensors": tensor_observations,
        "attention_implementation": model.llm.config._attn_implementation,
        "backbone_input_tensors": backbone_observations,
        "forward_calls": len(forward_calls),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": {
            k: importlib.metadata.version(k)
            for k in ["torchaudio", "transformers", "accelerate", "numpy", "soundfile"]
        },
        "seed": args.seed,
        "warmup": args.warmup,
        "config": config.__dict__,
        "requested_duration": args.duration,
        "language": args.language,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "ref_audio_sha256": hashlib.sha256(args.ref_audio.read_bytes()).hexdigest()
        if args.ref_audio
        else None,
        "ref_text_sha256": hashlib.sha256(ref_text.encode()).hexdigest()
        if ref_text is not None
        else None,
        "prompt_sha256": hashlib.sha256(args.prompt.read_bytes()).hexdigest()
        if args.prompt
        else None,
        "reference_input_seconds": sf.info(args.ref_audio).duration
        if args.ref_audio
        else None,
        "model_load_seconds": load_seconds,
        "processing_seconds": processing_seconds,
        "audio_seconds": duration,
        "rtf": processing_seconds / duration if duration else None,
        "stages": {
            k: {
                "seconds": v,
                "percent": 100 * v / processing_seconds,
                "calls": timer.calls[k],
            }
            for k, v in timer.seconds.items()
        },
        "unattributed_seconds": processing_seconds - sum(timer.seconds.values()),
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(timer.device) / 2**20
        if timer.device.type == "cuda"
        else None,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(timer.device) / 2**20
        if timer.device.type == "cuda"
        else None,
        "gpu": gpu,
        "output": str(args.output),
    }
    if timer.device.type == "cuda":
        props = torch.cuda.get_device_properties(timer.device)
        report["gpu_hardware"] = {
            "name": props.name,
            "total_memory_mib": props.total_memory / 2**20,
            "compute_capability": [props.major, props.minor],
        }
    if model._asr_pipe is not None:
        report["placement"]["asr"] = model_placement(model._asr_pipe.model)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "device",
                    "processing_seconds",
                    "audio_seconds",
                    "rtf",
                    "cuda_peak_allocated_mib",
                    "stages",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
