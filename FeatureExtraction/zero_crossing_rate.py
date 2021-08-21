'''
How many time signal will cross zero value.

Code/explanation thanks to https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html
'''
import librosa

audio_data, sample_rate = librosa.load(r'Data\sober\mikolajsober1.wav')
zero_crossings = librosa.zero_crossings(audio_data, pad = False)#if pad is True audio_data[0] is considered a valid zero-crossing value
#in this case we do not take the audio_data[0] as valid zero-crossing value
print(sum(zero_crossings))#5605


