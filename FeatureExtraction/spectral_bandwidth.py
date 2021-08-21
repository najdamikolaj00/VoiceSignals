'''
The spectral bandwidth is defined as the width of the band of light at one-half the peak maximum (or full width at half maximum [FWHM]) 
and is represented by the two vertical red lines and λSB on the wavelength axis. - https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

Code/explanation thanks to https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

'''
from scipy.ndimage.measurements import label
import sklearn
import librosa
import matplotlib.pyplot as plt
import librosa.display


audio_data, sample_rate = librosa.load(r'Data\sober\mikolajsober1.wav')

spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(audio_data + 0.01 , sr = sample_rate)[0]#returns an array with columns equal
#to number of frames in sample
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(audio_data + 0.01, sr = sample_rate, p = 3)[0]#p parameter is the power
# to raise deviation from spectral centroid - moc zwiększająca odchylenie od środka ciężkości widma
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(audio_data + 0.01, sr = sample_rate, p = 4)[0]
#shape_of_spectral_centroids = spectral_centroids.shape #use only if you want to know the shape

'''
Computing the time variable for visualization
'''

frames = range(len(spectral_bandwidth_2))
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
plt.plot(t, normalize(spectral_bandwidth_2), color='r', label = 'p = 2')
plt.plot(t, normalize(spectral_bandwidth_3), color='g', label = 'p = 3')
plt.plot(t, normalize(spectral_bandwidth_4), color='y', label = 'p = 4')
plt.legend(loc = 'lower right')
plt.show()