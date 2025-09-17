import torch
from fairseq import checkpoint_utils
import torch.nn as nn

# TODO:  make this model loading independant of fairseq.


class TripletModel(nn.Module):
    """
    Helper class defining the underlying neural network architecture for the SCOREQ model.
    """

    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        """
        Initializes the TripletModel.

        Args:
            ssl_model: The pre-trained self-supervised learning model (e.g., wav2vec).
            ssl_out_dim: Output dimension of the SSL model.
            emb_dim: Dimension of the final embedding (default: 256).
        """

        super(TripletModel, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = nn.Sequential(
            nn.ReLU(), nn.Linear(self.ssl_features, emb_dim)
        )

    def forward(self, wav, phead=False):
        """
        Defines the forward pass of the model.

        Args:
            wav: Input audio waveform.
            phead: Attach embedding layer for reference mode prei

        Returns:
            The normalized embedding of the input audio.
        """

        wav = wav.squeeze(1)
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res["x"]
        x = torch.mean(x, 1)

        # Choose if you want to keep projection head, remove for NR mode. Const model shows better performance in ODM without phead.
        if phead:
            x = self.embedding_layer(x)
        x = torch.nn.functional.normalize(x, dim=1)
        return x


# ******** MOS PREDICTOR **********
class MosPredictor(nn.Module):
    """
    Helper class that adds a layer for predicting Mean Opinion Scores (MOS) in the no-reference mode.
    """

    def __init__(self, pt_model, emb_dim=768):
        """
        Initializes the MosPredictor.

        Args:
            pt_model: The pre-trained triplet model.
            emb_dim: Dimension of the embedding (default: 768).
        """
        super(MosPredictor, self).__init__()
        self.pt_model = pt_model
        self.mos_layer = nn.Linear(emb_dim, 1)

    def forward(self, wav):
        """
        Defines the forward pass of the MOS predictor.

        Args:
            wav: Input audio waveform.

        Returns:
            The predicted MOS and the embedding.
        """
        x = self.pt_model(wav, phead=False)
        if len(x.shape) == 3:
            x.squeeze_(2)
        out = self.mos_layer(x)
        return out


class ScoreQModel:
    def __init__(self, wav2vec_model_path: str, triplet_model_path: str):
        self.W2V_OUT_DIM = 768
        self.EMB_DIM = 256
        # Load w2v BASE
        w2v_model, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [wav2vec_model_path]
        )
        ssl_model = w2v_model[0]
        ssl_model.remove_pretraining_modules()
        pt_model = TripletModel(ssl_model, self.W2V_OUT_DIM, self.EMB_DIM)
        self.model = MosPredictor(pt_model, self.EMB_DIM)
        if torch.cuda.is_available():
            self.model = self.model.load_state_dict(
                torch.load(triplet_model_path, map_location="cuda")
            )
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.model = self.model.load_state_dict(
                torch.load(triplet_model_path, map_location="mps")
            )
            self.device = "mps"
        else:
            self.model = self.model.load_state_dict(
                torch.load(triplet_model_path, map_location="cpu")
            )
        self.model = self.model.eval()  # type: ignore
