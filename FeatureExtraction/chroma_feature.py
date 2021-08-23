'''
It provides a strong way to describe a similarity measure between voice pieces

Code/explanation thanks to https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html
'''
import librosa
import librosa.display
import matplotlib.pyplot as plt

# audio_data, sample_rate = librosa.load(r'C:\Users\mikol\Projekty\VoiceSignals\VoiceSignals\Datano2\sober\sylwiasober2.wav')
audio_data, sample_rate = librosa.load(r'C:\Users\mikol\Projekty\VoiceSignals\VoiceSignals\Datano2\unsober\sylwiaunsober1.wav')
chromagram = librosa.feature.chroma_stft(audio_data, sr = sample_rate)
plt.figure(figsize=(12, 4))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.show()

