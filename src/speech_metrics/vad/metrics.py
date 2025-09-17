import torch
import torchcodec
import torchaudio.functional as F


class VADMetrics:
    def __init__(self):
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad"
        )  # type: ignore
        (self.get_speech_timestamps, _, _, _, _) = utils

    def compute(self, audio_samples: torchcodec.AudioSamples):
        audio_tensor = audio_samples.data
        sample_rate = audio_samples.sample_rate
        if sample_rate != 16000:
            audio_tensor = F.resample(audio_tensor, sample_rate, 16000)
        audio_tensor.squeeze(0)
        duration = audio_samples.duration_seconds
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )
        silence_ratio = self.calculate_total_silence_ratio(speech_timestamps, duration)
        non_audio_silence = self.non_audio_silence(speech_timestamps, duration)
        return {
            "SILENCE_RATIO": silence_ratio,
            "NON_AUDIO_SILENCE": non_audio_silence,
        }

    @staticmethod
    def calculate_total_silence_ratio(segments, total_duration):
        speech_time = sum(seg["end"] - seg["start"] for seg in segments)
        silence_time = total_duration - speech_time
        return silence_time / total_duration

    @staticmethod
    def non_audio_silence(speech_timestamps, total_duration):
        timestamps = []
        for tm in speech_timestamps:
            for val in tm.values():
                timestamps.append(val)
        return timestamps[0] + (total_duration - timestamps[-1])
