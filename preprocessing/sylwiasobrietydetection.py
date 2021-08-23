'''
I need this program for plotting some values
'''
from audio_preprocessing import import_audio, AudioPreprocessing

AUDIO_PATH = r'C:\Users\mikol\Projekty\VoiceSignals\VoiceSignals\Datano2\unsober\sylwiaunsober1.wav'
audio_data, sample_rate = import_audio(AUDIO_PATH)
audio = AudioPreprocessing('sylwiaunsober1.wav', audio_data, sample_rate)
audio.crop_audio(threshold = 0.01)
audio.plot_timeseries()

