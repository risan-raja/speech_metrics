import torchcodec
import pytest
from speech_metrics.mos.metrics import SpeechMOSMetrics


@pytest.mark.parametrize("test_case", list(range(1, 10)))
def test_mos_metrics(test_case, shared_datadir):
    audio_path = shared_datadir / f"audio_{test_case}.wav"
    audio_data = torchcodec.decoders.AudioDecoder(audio_path).get_all_samples()
    model = SpeechMOSMetrics()
    metrics = model.compute(audio_data)
    assert any([v > 0 for v in metrics.values()])
    assert len(metrics) == 5
