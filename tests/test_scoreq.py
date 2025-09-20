import torchcodec
import pytest

# from metrics import ScoreQMetrics
from speech_metrics.scoreq.metrics import ScoreQMetrics


@pytest.mark.parametrize("test_case", list(range(1, 10)))
def test_scoreq_metrics(test_case, shared_datadir):
    """
    Tests that the ScoreQ metric returns a valid MOS score between 1 and 5.
    """
    # Define paths to the audio file and the model weights
    # Assumes the test data is in a shared directory provided by pytest-datadir
    # and the model weights are in the 'scoreq' module directory.
    audio_path = shared_datadir / f"audio_{test_case}.wav"
    model_path = shared_datadir / "adapt_nr_telephone.pt"

    # 1. Load the audio file using torchcodec
    audio_data = torchcodec.decoders.AudioDecoder(str(audio_path)).get_all_samples()

    # 2. Initialize the ScoreQ model
    model = ScoreQMetrics(model_path=model_path)

    # 3. Compute the metrics
    metrics = model.compute(audio_data)
    score = metrics["SCOREQ_MOS"]

    print(score)

    # 4. Assert that the score is within the valid MOS range
    assert 1.0 <= score <= 5.0, (
        f"Expected SCOREQ_MOS to be between 1 and 5, but got {score}"
    )
