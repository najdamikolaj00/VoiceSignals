'''
Here I will try complex feature extraction, building and fit the model 
'''
'''
Importing required libraries
'''
import librosa
import numpy as np
import os
import csv 
import warnings
warnings.filterwarnings('ignore')
from save_spectograms import save_spectrograms
from converter import converter

# '''
# Converting audio data to files with .wav extension
# '''
# converter()
# '''
# Converting audio data to files with .PNG extension
# '''
# save_spectrograms()

'''
Creating header file for .csv
'''
header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 20):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('dataset.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
sobriety = 'sober unsober'.split()
for s in sobriety:
    for filename in os.listdir(f'C:/Users/mikol/Projekty/VoiceSignals/VoiceSignals/Data/' + s):
        audioname = f'C:/Users/mikol/Projekty/VoiceSignals/VoiceSignals/Data/' + s + '/' + filename
        y, sr = librosa.load(audioname, mono = True, duration = 30)
        # rms = librosa.feature.rmse(y = y)
        chroma_stft = librosa.feature.chroma_stft(y = y, sr = sr)
        spec_cent = librosa.feature.spectral_centroid(y = y, sr = sr)
        spec_bw = librosa.feature.spectral_bandwidth(y = y, sr = sr)
        rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y = y, sr = sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {s}'
        file = open('dataset.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


