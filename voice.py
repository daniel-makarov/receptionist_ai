import os
import torch
import argparse
import gradio as gr
from zipfile import ZipFile
import langid
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter

parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

en_ckpt_base = 'checkpoints/base_speakers/EN'
zh_ckpt_base = 'checkpoints/base_speakers/ZH'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
zh_base_speaker_tts = BaseSpeakerTTS(f'{zh_ckpt_base}/config.json', device=device)
zh_base_speaker_tts.load_ckpt(f'{zh_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# load speaker embeddings
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)
zh_source_se = torch.load(f'{zh_ckpt_base}/zh_default_se.pth').to(device)

# This online demo mainly supports English and Chinese
supported_languages = ['zh', 'en']


def predict(prompt, style, audio_file_pth, agree):
    # initialize a empty info
    text_hint = ''
    # agree with the terms
    if agree == False:
        text_hint += '[ERROR] Please accept the Terms & Condition!\n'
        gr.Warning("Please accept the Terms & Condition!")
        return (
            text_hint,
            None,
            None,
        )

    # first detect the input language
    language_predicted = langid.classify(prompt)[0].strip()
    print(f"Detected language:{language_predicted}")

    if language_predicted not in supported_languages:
        text_hint += f"[ERROR] The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}\n"
        gr.Warning(
            f"The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}"
        )

        return (
            text_hint,
            None,
            None,
        )

    speaker_wav = audio_file_pth

    if len(prompt) < 2:
        text_hint += f"[ERROR] Please give a longer prompt text \n"
        gr.Warning("Please give a longer prompt text")
        return (
            text_hint,
            None,
            None,
        )
    if len(prompt) > 200:
        text_hint += f"[ERROR] Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo and try for your usage \n"
        gr.Warning(
            "Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo for your usage"
        )
        return (
            text_hint,
            None,
            None,
        )

    # note diffusion_conditioning not used on hifigan (default mode), it will be empty but need to pass it to model.inference
    try:
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)
    except Exception as e:
        text_hint += f"[ERROR] Get target tone color error {str(e)} \n"
        gr.Warning(
            "[ERROR] Get target tone color error {str(e)} \n"
        )
        return (
            text_hint,
            None,
            None,
        )

    src_path = f'{output_dir}/tmp.wav'

    save_path = f'{output_dir}/output.wav'
    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message)

    return (
        text_hint,
        save_path,
        speaker_wav,
    )
