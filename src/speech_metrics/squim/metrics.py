import torch
import torchaudio
import torchaudio.functional as F
import torchcodec


class SquimMetrics:
    def __init__(self, model_path=""):
        self.model = torchaudio.models.squim_objective_base()
        self.state_dict = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(self.state_dict)
        self.device = "cpu"
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.model = self.model.to("mps")
            self.device = "mps"
        else:
            self.model = self.model.to("cpu")
        self.model = self.model.eval()

    def compute(self, audio_samples: torchcodec.AudioSamples):
        audio_tensor = audio_samples.data
        sample_rate = audio_samples.sample_rate
        if sample_rate != 16000:
            audio_tensor = F.resample(audio_tensor, sample_rate, 16000)
        audio_tensor = audio_tensor.to(self.device)
        stoi, pesq, si_sdr = self.model(audio_tensor)
        return {
            "STOI": stoi.item(),
            "PESQ": pesq.item(),
            "SI-SDR": si_sdr.item(),
        }
