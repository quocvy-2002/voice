"""Compare long-form assembly with a recorded Git revision, using synthetic audio.

Example: python -m examples.benchmark_audio_join --baseline COMMIT --report result.json
This measures assembly only, not TTS throughput or voice quality.
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from omnivoice.utils.audio import cross_fade_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        help="Git revision containing the original assembly function.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:omnivoice/utils/audio.py"], encoding="utf-8"
    )
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "cross_fade_chunks"
    )
    namespace = {"np": np}
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), "<baseline_assembly>", "exec"
        ),
        namespace,
    )
    baseline = namespace["cross_fade_chunks"]
    rng = np.random.default_rng(42)
    cases = 0
    for count in (1, 2, 5, 20):
        for channels in (1, 2):
            for silence in (0, 0.01, 0.3):
                for dtypes in (
                    (np.float32,),
                    (np.float64,),
                    (np.float16,),
                    (np.float16, np.float32, np.float64),
                ):
                    chunks = [
                        rng.uniform(
                            -0.5, 0.5, size=(channels, int(rng.integers(0, 100)))
                        ).astype(dtypes[i % len(dtypes)])
                        for i in range(count)
                    ]
                    expected = baseline(chunks, 1000, silence)
                    actual = cross_fade_chunks(chunks, 1000, silence)
                    np.testing.assert_array_equal(actual, expected)
                    assert actual.dtype == expected.dtype
                    cases += 1
    rng = np.random.default_rng(42)
    chunks = [
        rng.uniform(-0.5, 0.5, size=(1, 15 * 24000)).astype(np.float32)
        for _ in range(100)
    ]
    times = {"baseline": [], "optimized": []}
    for _ in range(3):
        results = []
        for name, function in (
            ("baseline", baseline),
            ("optimized", cross_fade_chunks),
        ):
            started = time.perf_counter()
            result = function(chunks, 24000)
            times[name].append(time.perf_counter() - started)
            results.append(result)
        np.testing.assert_array_equal(*results)
    report = {
        "baseline_revision": args.baseline,
        "cases_bit_exact": cases,
        "output_seconds": result.shape[-1] / 24000,
        "times": times,
        "speedup_median": float(
            np.median(times["baseline"]) / np.median(times["optimized"])
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
