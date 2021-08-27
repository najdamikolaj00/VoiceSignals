'''
Here I will change audio_preprocessing file the way I want it to look now 27.08.2021
'''
import numpy as np
import librosa
import librosa.display
import os


def import_audio(path):
    if os.path.exists(path):
        audio_data, sample_rate = librosa.load(path)
        return audio_data, sample_rate
    else:
        raise FileNotFoundError 

class AudioPreprocessing():

    def __init__(self, audioname, threshold):
        '''
        Constructor
        '''
        self.audioname = audioname
        self.threshold = threshold# Threshold should be calculate somehow
    def preprocessing(self):

        audio_data, sample_rate = librosa.load(self.audioname, mono = True, duration = 30)
        
        def delete_treshold_segments(audio_file, threshold):
            audio_data = audio_file
            while abs(audio_data[0]) < threshold:
                    audio_data = audio_data[1:]

            return audio_data

        audio_data = self.audio_file
        left_deleted = delete_treshold_segments(audio_data, self.threshold)
        right_deleted = delete_treshold_segments(np.flip(left_deleted), self.threshold)
        self.audio_file = np.flip(right_deleted)

    def features(self):

        chroma_stft = librosa.feature.chroma_stft(y = y, sr = sr)
        spec_cent = librosa.feature.spectral_centroid(y = y, sr = sr)
        spec_bw = librosa.feature.spectral_bandwidth(y = y, sr = sr)
        rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y = y, sr = sr)

        
        pass

   