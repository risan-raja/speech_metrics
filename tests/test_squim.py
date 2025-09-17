from speech_metrics.squim.metrics import SquimMetrics
import torchcodec
import pytest


@pytest.mark.parametrize("test_case", list(range(1, 10)))
def test_squim_metrics(test_case, shared_datadir):
    model_wts = shared_datadir / "squim_objective_dns2020.pth"
    model = SquimMetrics(model_path=str(model_wts))
    audio_path = shared_datadir / f"audio_{test_case}.wav"
    audio_data = torchcodec.decoders.AudioDecoder(audio_path).get_all_samples()
    metrics = model.compute(audio_data)
    assert all(k in metrics for k in ["STOI", "PESQ", "SI-SDR"])
    assert all(isinstance(v, float) for v in metrics.values())
