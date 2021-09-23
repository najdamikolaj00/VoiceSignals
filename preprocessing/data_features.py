'''
Class for feature extraction
Steps to cover:

4. Feature extraction
6. Data preprocessing
'''

import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Data_operator():

    def __init__(self):
       pass
    def features(self, audio_file, sample_rate):
        '''
        Function for features extraction using librosa library

        returns ''six'' features from audio file
        '''

        chroma_stft = librosa.feature.chroma_stft(y = audio_file, sr = sample_rate)
        spec_cent = librosa.feature.spectral_centroid(y = audio_file, sr = sample_rate)
        spec_bw = librosa.feature.spectral_bandwidth(y = audio_file, sr = sample_rate)
        rolloff = librosa.feature.spectral_rolloff(y = audio_file, sr = sample_rate)
        zcr = librosa.feature.zero_crossing_rate(audio_file)
        mfcc = librosa.feature.mfcc(y = audio_file, sr = sample_rate)

        return chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc

    def data_preprocessing(self):
        data = pd.read_csv('dataset.csv')
        data.head() # Dropping unneccesary columns
        data = data.drop(['filename'], axis=1) #Encoding the Labels
        sobriety_list = data.iloc[:, -1]
        encoder = LabelEncoder()
        y = encoder.fit_transform(sobriety_list)#Scaling the Feature columns
        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
        # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5)
        # return X_train, X_test, y_train, y_test
