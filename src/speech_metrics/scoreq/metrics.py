import torch
import torchcodec
import torch.nn as nn
import torchaudio.functional as F
from transformers import Wav2Vec2Model
from pathlib import Path

# --- Helper PyTorch Modules (Internal to this file) ---

class TripletModel(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        super(TripletModel, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = nn.Sequential(
            nn.ReLU(), nn.Linear(self.ssl_features, emb_dim)
        )

    def forward(self, wav, phead=False):
        wav = wav.squeeze(1)
        res = self.ssl_model(wav).last_hidden_state
        x = torch.mean(res, 1)
        if phead:
            x = self.embedding_layer(x)
        x = torch.nn.functional.normalize(x, dim=1)
        return x

class MosPredictor(nn.Module):
    def __init__(self, pt_model, emb_dim=768):
        super(MosPredictor, self).__init__()
        self.pt_model = pt_model
        self.mos_layer = nn.Linear(emb_dim, 1)

    def forward(self, wav):
        x = self.pt_model(wav, phead=False)
        if len(x.shape) == 3:
            x.squeeze_(2)
        out = self.mos_layer(x)
        return out

# --- Main Metric Class ---

class ScoreQMetrics:
    """
    Computes the ScoreQ MOS score for a given audio sample.
    """
    def __init__(self, model_path: Path, wav2vec_model_name: str = "facebook/wav2vec2-base-960h"):
        self.device = self._get_device()
        
        # Load the wav2vec base model from Hugging Face
        ssl_model = Wav2Vec2Model.from_pretrained(wav2vec_model_name).to(self.device)
        
        # Build the full model architecture
        pt_model = TripletModel(ssl_model, ssl_out_dim=768, emb_dim=256)
        self.scoreq_model = MosPredictor(pt_model, emb_dim=768)

        # Load the state dict for the fine-tuned mos_layer
        state_dict = torch.load(model_path, map_location=self.device)
        self.scoreq_model.load_state_dict(state_dict, strict=False)
        self.scoreq_model.to(self.device)
        self.scoreq_model.eval()
        print("ScoreQ model loaded successfully.")

    def compute(self, audio_samples: torchcodec.AudioSamples) -> dict:
        """
        Computes the ScoreQ MOS score. The audio will be resampled to 16kHz.

        Args:
            audio_samples (torchcodec.AudioSamples): The audio to be scored.

        Returns:
            dict: A dictionary containing the MOS score with the key "SCOREQ_MOS".
        """
        audio_tensor = audio_samples.data.float()
        sample_rate = audio_samples.sample_rate

        # Resample with torchaudio if needed
        if sample_rate != 16000:
            audio_tensor = F.resample(audio_tensor, sample_rate, 16000)

        # Ensure audio is mono and has a batch dimension [1, num_samples]
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        
        audio_tensor = audio_tensor.to(self.device)
        
        with torch.no_grad():
            mos_tensor = self.scoreq_model(audio_tensor)
        
        mos_score = mos_tensor.item()
        
        return {"SCOREQ_MOS": mos_score}

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
