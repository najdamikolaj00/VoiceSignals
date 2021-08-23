'''
Function for saving spectograms as .png files.

'''
import os
from audio_preprocessing import import_audio, AudioPreprocessing

def save_spectrograms():
    os.chdir("..")
    rootdir = 'Datano2'
    for subdir, dirs, files in os.walk(rootdir):
        for filename in files:
            print(filename)
            if filename.endswith('.wav'):
                AUDIO_PATH = os.path.join(subdir, filename)
                SPEC_SAVE_PATH = os.path.join(os.getcwd(), "spectrograms", filename)
                print(SPEC_SAVE_PATH)
                if not os.path.exists(SPEC_SAVE_PATH):
                    audio_data, sample_rate = import_audio(AUDIO_PATH)
                    audio = AudioPreprocessing(filename, audio_data, sample_rate)
                    audio.crop_audio(threshold = 0.001)
                    audio.convert_to_png(SPEC_SAVE_PATH)
                else:
                    continue
