import subprocess
import uuid
import whisper
from deep_translator import GoogleTranslator
import nltk

from pytube import YouTube
import os

from TTS.api import TTS
import torch
from IPython.display import Audio, display

language_mapping = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Polish': 'pl',
    'Turkish': 'tr',
    'Russian': 'ru',
    'Dutch': 'nl',
    'Czech': 'cs',
    'Arabic': 'ar',
    'Chinese (Simplified)': 'zh-cn' 
}
class Downloader():
    # TODO: Support multi-source video download
    def download(url, file_name, out_path) -> str:
        if not os.path.exists(out_path):
            raise "output path not exist"
        try:
            yt = YouTube(url)
            yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().downloa.1d(output_path=out_path, filename=file_name)
            return os.path.join(out_path, file_name)
        except Exception as ex:
            raise "Fail to download file: " + str(ex)

class Translator():
    CALL_LIMIT = 5000
    def __init__(self, src, dst) -> None:
        self.google = GoogleTranslator(src, dst)
        
    def translate(self, content) -> str:
        if len(content) < self.CALL_LIMIT:
            return self.google.translate(content)
        translated = ""
        tmp = nltk.tokenize.sent_tokenize(content)
        slice = ""
        for sentence in tmp:
            if len(slice) + len(sentence) < 5000:
                slice = slice + sentence
                continue
            cur_translated = self.google.translate(slice)
            translated = translated + cur_translated
            slice = ""
        if len(slice) > 0:
            translated = translated + self.google.translate(slice)
        return translated

class Worker():
    downloader = None
    translator = None
    path = ""
    w_model = None
    def __init__(self, option: {}):
        self.path = option['output_path'] if "output_path" in option.keys() else '.'
        self.w_model =  whisper.load_model(option['w_model'] if "w_model" else "large")
        self.downloader = Downloader()
        self.translator = Translator('zh-CN', 'en')
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

    def run(self, url, target_lan='en'):
        filename = "{}.mp4".format(str(uuid.uuid4()).replace("-", ""))
        video_file = self.downloader.download(url, file_name=filename, out_path=self.path)
        wav_file = video_file.replace(".mp4", ".wav")
        ffmpeg_command = "ffmpeg -i '{video_f}' -acodec pcm_s24le -ar 48000 -q:a 0 -map a -y '{wav_f}.wav'".format(video_f=video_file, wav_f=wav_file)
        shell_cmd = subprocess.run(ffmpeg_command, shell=True)
        if shell_cmd.returncode != 0:
            raise "Error when run ffmpeg"
        text_context = self.w_model.transcribe(wav_file)
        text_traslated = self.translator.translate(text_context)
        wav_file_convert = os.path.join(self.path, "converted_" + filename.replace(".mp4", ".wav"))
        self.tts.tts_to_file(text_traslated, speaker_wav=wav_file,
                             file_path=wav_file_convert,
                             language='zh-cn'
                             )

        
if __name__ == '__main__':
    pass

