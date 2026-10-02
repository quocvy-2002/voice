import numpy as np
import soundfile as sf

from omnivoice.utils import audio


def test_load_waveform_uses_ffmpeg_fallback_when_soundfile_cannot_decode(monkeypatch):
    interleaved_samples = np.array([-32768, 0, 16384, 32767], dtype=np.int16)

    class DecodedAudio:
        channels = 2
        frame_rate = 48_000

        @staticmethod
        def get_array_of_samples():
            return interleaved_samples

    def reject_soundfile(*_args, **_kwargs):
        raise sf.LibsndfileError(1)

    monkeypatch.setattr(
        audio.sf,
        "read",
        reject_soundfile,
    )
    monkeypatch.setattr(
        audio.AudioSegment,
        "from_file",
        lambda _path: DecodedAudio(),
    )

    waveform, sample_rate = audio.load_waveform("voice-reference.m4a")

    assert sample_rate == 48_000
    np.testing.assert_allclose(
        waveform,
        np.array([[-1.0, 0.5], [0.0, 32767 / 32768]], dtype=np.float32),
    )
