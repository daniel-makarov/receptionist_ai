import torch
import numpy as np
import re
import soundfile as sf
import helper_functions as helpers
import data_manipulation as data_utils
import os
import librosa
from linguistic_processing import convert_text_to_sequence
from audio_analysis import generate_spectrogram
from voice_models import TextToSpeechModel


class VoiceSynthesizerBase(object):
    def __init__(self, configuration, compute_device='cuda:0'):
        self._verify_cuda_availability(compute_device)

        settings = helpers.load_hyperparameters(configuration)

        self.voice_model = TextToSpeechModel(
            symbol_count=len(getattr(settings, 'characters', [])),
            spectrogram_size=settings.audio.filter_length // 2 + 1,
            speaker_total=settings.audio.number_of_speakers,
            **settings.model_parameters,
        ).to(compute_device)

        self.voice_model.eval()
        self.settings = settings
        self.compute_device = compute_device

    def _verify_cuda_availability(self, device):
        if 'cuda' in device:
            assert torch.cuda.is_available(), "CUDA is not available on this device."

    def update_model_weights(self, weights_file):
        model_data = torch.load(weights_file, map_location=torch.device(self.compute_device))
        missing_keys, unexpected_keys = self.voice_model.load_state_dict(model_data['model_state'], strict=False)
        print(
            f"Model weights from '{weights_file}' loaded with missing keys: {missing_keys} and unexpected keys: {unexpected_keys}")


class SpeechSynthesis(VoiceSynthesizerBase):
    language_codes = {
        "english": "EN",
        "chinese": "ZH",
    }

    @staticmethod
    def prepare_text_for_synthesis(text, settings, use_symbols):
        processed_text = convert_text_to_sequence(text, settings.characters,
                                                  [] if use_symbols else settings.audio.text_cleaning_rules)
        if settings.audio.include_blank:
            processed_text = data_utils.insert_blanks(processed_text, blank_token=0)
        return torch.LongTensor(processed_text)

    @staticmethod
    def concatenate_audio_segments(segments, sample_rate, playback_speed=1.0):
        combined_audio = []
        for segment in segments:
            combined_audio.extend(segment.reshape(-1).tolist())
            combined_audio.extend([0] * int((sample_rate * 0.05) / playback_speed))
        return np.array(combined_audio, dtype=np.float32)

    def synthesize_text_to_speech(self, text_input, save_path, speaker_identity, language='English', speed=1.0):
        language_prefix = self.language_codes.get(language.lower(), None)
        assert language_prefix is not None, f"Unsupported language: {language}"

        segmented_texts = self._segment_text_based_on_language(text_input, language_prefix)

        synthesized_audio_segments = []
        for text_segment in segmented_texts:
            processed_segment = self.process_text_segment(text_segment, language_prefix)
            audio_segment = self.generate_audio_from_text(processed_segment, speaker_identity, speed)
            synthesized_audio_segments.append(audio_segment)

        final_audio = self.concatenate_audio_segments(synthesized_audio_segments,
                                                      sample_rate=self.settings.audio.sampling_rate,
                                                      playback_speed=speed)

        if save_path:
            sf.write(save_path, final_audio, self.settings.audio.sampling_rate)
        return final_audio

    def _segment_text_based_on_language(self, text, language_code):
        segmented_text = helpers.segment_text_based_on_language(text, language_code)
        print("Segmented text into sentences:")
        print('\n'.join(segmented_text))
        return segmented_text

    def process_text_segment(self, text_segment, language_code):
        modified_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text_segment)
        return f'[{language_code}]{modified_text}[{language_code}]'

    def generate_audio_from_text(self, text, speaker, playback_speed):
        text_tensor = self.prepare_text_for_synthesis(text, self.settings, False)
        speaker_id = self.settings.audio.speaker_ids[speaker]

        with torch.no_grad():
            text_tensor = text_tensor.unsqueeze(0).to(self.compute_device)
            text_length = torch.LongTensor([text_tensor.size(0)]).to(self.compute_device)
            speaker_tensor = torch.LongTensor([speaker_id]).to(self.compute_device)
            audio_output = self.voice_model.infer(text_tensor, text_length, speaker_id=speaker_tensor,
                                                  speed_control=1.0 / playback_speed)[0][0, 0].cpu().numpy().astype(
                np.float32)
        return audio_output
