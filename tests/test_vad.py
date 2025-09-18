# TODO: Write the tests for vad.py
import torchcodec
import pytest
from speech_metrics.vad.metrics import VADMetrics


@pytest.mark.parametrize("test_case", list(range(1, 10)))
def test_vad_metrics(test_case, shared_datadir):
    audio_path = shared_datadir / f"audio_{test_case}.wav"
    audio_data = torchcodec.decoders.AudioDecoder(audio_path).get_all_samples()
    model = VADMetrics()
    metrics = model.compute(audio_data)

    silence_ratio = metrics["SILENCE_RATIO"]
    non_audio_silence = metrics["NON_AUDIO_SILENCE"]

    # The silence ratio must be a value between 0.0 and 1.0.
    assert 0.0 <= silence_ratio <= 1.0, (
        f"Expected SILENCE_RATIO to be between 0 and 1, but got {silence_ratio}"
    )

    # The duration of silence at the start and end cannot be negative or
    # exceed the total duration of the audio.
    assert 0.0 <= non_audio_silence <= audio_data.duration_seconds, (
        f"Expected NON_AUDIO_SILENCE ({non_audio_silence}s) to be within the total duration ({audio_data.duration_seconds}s)"
    )
