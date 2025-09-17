from speechmos import dnsmos, plcmos
import torchaudio.functional as F
import torchcodec


class SpeechMOSMetrics:
    def __init__(self):
        self.dnsmos_model = dnsmos
        self.plcmos_model = plcmos

    def compute(self, audio_samples: torchcodec.AudioSamples):
        """
        Computes MOS scores for the given audio samples.
        The audio should be single-channel and will be resampled to 16kHz if necessary.

        Args:
            audio_samples (torchcodec.AudioSamples): The audio samples to compute MOS scores for.

        Returns:
            dict: A dictionary containing the MOS scores with keys:
                "DNSMOS_OVRL_MOS", "DNSMOS_SIG_MOS", "DNSMOS_BAK_MOS", "DNSMOS_P808_MOS", "PLCMOS_MOS".
        """
        audio_tensor = (
            F.resample(audio_samples.data, audio_samples.sample_rate, 16000)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        sample_rate = 16000
        dnsmos_score = self.dnsmos_model.run(audio_tensor, sample_rate)
        plcmos_score = self.plcmos_model.run(audio_tensor, sample_rate)
        if isinstance(plcmos_score, dict) and isinstance(dnsmos_score, dict):
            scores = {
                "DNSMOS_OVRL_MOS": dnsmos_score.get("ovrl_mos", 0.0),
                "DNSMOS_SIG_MOS": dnsmos_score.get("ovrl_mos", 0.0),
                "DNSMOS_BAK_MOS": dnsmos_score.get("bak_mos", 0.0),
                "DNSMOS_P808_MOS": dnsmos_score.get("p808_mos", 0.0),
                "PLCMOS_MOS": plcmos_score.get("plcmos", 0.0),
            }
        else:
            scores = {
                "DNSMOS_OVRL_MOS": 0.0,
                "DNSMOS_SIG_MOS": 0.0,
                "DNSMOS_BAK_MOS": 0.0,
                "DNSMOS_P808_MOS": 0.0,
                "PLCMOS_MOS": 0.0,
            }
        return scores
