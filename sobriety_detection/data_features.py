'''
Class for feature extraction
Steps to cover:
4. Feature extraction
'''

import librosa

class Data_features(object):

    def __init__(self, audio_data, sample_rate):
        '''
        Constructor takes dictionary of all audio_data, sample_rate and sobriety.
        '''
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def features(self):
        '''
        Function for features extraction using librosa library
        returns ''six'' features from audio file
        '''
        chroma_stft = librosa.feature.chroma_stft(y = self.audio_data, sr = self.sample_rate)
        spec_cent = librosa.feature.spectral_centroid(y = self.audio_data, sr = self.sample_rate)
        spec_bw = librosa.feature.spectral_bandwidth(y = self.audio_data, sr = self.sample_rate)
        rolloff = librosa.feature.spectral_rolloff(y = self.audio_data, sr = self.sample_rate)
        zcr = librosa.feature.zero_crossing_rate(self.audio_data)
        mfcc = librosa.feature.mfcc(y = self.audio_data, sr = self.sample_rate)

        return chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc