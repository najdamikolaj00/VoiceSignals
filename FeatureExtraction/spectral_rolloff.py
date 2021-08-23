'''
Spectral rolloff indicates the roll off frequency for each frame in signal. - https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

Code/explanation thanks to https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

'''
import sklearn
import librosa
import matplotlib.pyplot as plt
import librosa.display

audio_data, sample_rate = librosa.load(r'C:\Users\mikol\Projekty\VoiceSignals\VoiceSignals\Datano2\sober\sylwiasober2.wav')
# audio_data, sample_rate = librosa.load(r'C:\Users\mikol\Projekty\VoiceSignals\VoiceSignals\Datano2\unsober\sylwiaunsober1.wav')

spectral_rolloff = librosa.feature.spectral_rolloff(audio_data + 0.01, sr = sample_rate)[0]

'''
Computing the time variable for visualization
'''

frames = range(len(spectral_rolloff))
t = librosa.frames_to_time(frames)

'''
Normalising the spectral centroid for visualisation
'''
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

'''
Plotting the Spectral roll off along the waveform
'''

plt.figure(figsize=(12, 4))
librosa.display.waveplot(audio_data, sr = sample_rate, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.show()