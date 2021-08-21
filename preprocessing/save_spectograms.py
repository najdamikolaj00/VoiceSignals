'''
Function for saving spectograms as .png files.

'''
import os
from audio_preprocessing import import_audio, AudioPreprocessing
from converter import converter

def save_spectrograms():
    os.chdir("..")
    rootdir = 'Data'
    for subdir, dirs, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith('.wav'):
                AUDIO_PATH = os.path.join(subdir, filename)
                SPEC_SAVE_PATH = os.path.join(os.getcwd(), "spectrograms", filename)
                if not os.path.exists(SPEC_SAVE_PATH):
                    audio_data, sample_rate = import_audio(AUDIO_PATH)
                    audio = AudioPreprocessing(filename, audio_data, sample_rate)
                    audio.crop_audio(threshold = 0.001)
                    audio.convert_to_png(SPEC_SAVE_PATH)
                else:
                    continue