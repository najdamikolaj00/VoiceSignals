'''
Class for feature extraction
Steps to cover:

4. Feature extraction
'''

import librosa

class Data_features(object):

    def __init__(self, data_for_features):
        '''
        Constructor takes dictionary of all audio_data, sample_rate and sobriety.
        '''
        self.data = data_for_features

    def features(self):
        '''
        Function for features extraction using librosa library

        returns ''six'' features from audio file
        '''
        for i in range(len(self.data['sobriety'])):

            chroma_stft = librosa.feature.chroma_stft(y = self.data['audio_data'][i], sr = self.data['sample_rate'][i])
            spec_cent = librosa.feature.spectral_centroid(y = self.data['audio_data'][i], sr = self.data['sample_rate'][i])
            spec_bw = librosa.feature.spectral_bandwidth(y = self.data['audio_data'][i], sr = self.data['sample_rate'][i])
            rolloff = librosa.feature.spectral_rolloff(y = self.data['audio_data'][i], sr = self.data['sample_rate'][i])
            zcr = librosa.feature.zero_crossing_rate(self.data['audio_data'][i])
            mfcc = librosa.feature.mfcc(y = self.data['audio_data'][i], sr = self.data['sample_rate'][i])

        return chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc
