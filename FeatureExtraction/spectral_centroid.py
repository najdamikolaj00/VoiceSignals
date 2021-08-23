'''
Spectral centroid indicates at which frequency the energy of a spectrum is centered upon. - https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html
It indicates where the center of mass of the spectrum is located. - https://en.wikipedia.org/wiki/Spectral_centroid
Środek ciężkości widma wskazuje, dla której częstotliwości energia widma się koncentruje. Opis kształtu widma mocy, wskazuje
jakie częstotliwości przeważają w widmie (wysokie/niskie).

Code/explanation thanks to https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

'''
import sklearn
import librosa
import matplotlib.pyplot as plt
import librosa.display


audio_data, sample_rate = librosa.load(r'C:\Users\mikol\Projekty\VoiceSignals\VoiceSignals\Datano2\sober\sylwiasober2.wav')
# audio_data, sample_rate = librosa.load(r'C:\Users\mikol\Projekty\VoiceSignals\VoiceSignals\Datano2\unsober\sylwiaunsober1.wav')
spectral_centroids = librosa.feature.spectral_centroid(audio_data, sr = sample_rate)[0]#returns an array with columns equal
#to number of frames in sample
#shape_of_spectral_centroids = spectral_centroids.shape #use only if you want to know the shape

'''
Computing the time variable for visualization
'''

frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)


'''
Normalising the spectral centroid for visualisation
'''
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

'''
Plotting the Spectral Centroid along the waveform
'''
plt.figure(figsize=(12, 4))
librosa.display.waveplot(audio_data, sr = sample_rate, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')
plt.show()