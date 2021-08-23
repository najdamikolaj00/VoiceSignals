'''
Small set of features which concisely describe the overall shape of a spectral envelope

Code/explanation thanks to https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html
'''
import librosa
import librosa.display
import matplotlib.pyplot as plt

# audio_data, sample_rate = librosa.load(r'C:\Users\mikol\Projekty\VoiceSignals\VoiceSignals\Datano2\sober\sylwiasober2.wav')
audio_data, sample_rate = librosa.load(r'C:\Users\mikol\Projekty\VoiceSignals\VoiceSignals\Datano2\unsober\sylwiaunsober1.wav')
mfccs = librosa.feature.mfcc(audio_data, sr = sample_rate)
#print(mfccs.shape)#if necessary
plt.figure(figsize=(12, 4))
librosa.display.specshow(mfccs, sr = sample_rate, x_axis='time')
plt.show()

