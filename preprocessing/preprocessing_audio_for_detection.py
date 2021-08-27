'''
Here I will change audio_preprocessing file the way I want it to look now 27.08.2021

'''
import numpy as np
import librosa
from converter import converter

class AudioPreprocessing():

    def __init__(self, audioname, treshold):
        '''
        Constructor

        audioname -> name of audio file
        threshold -> treshold float value currently equal to 0.01
        '''
        self.audioname = audioname
        self.treshold = treshold# Threshold should be calculate somehow
    
    def preprocessing(self):
        '''
        Function for preprocessing
        
        importing audio using "load" function from librosa library
        converter() -> Converting files with different extension than .wav using our-made function
        delete_treshold_segments -> deleting unwanted sound from audio files

        returns croped audio data and sample rate
        '''
        converter()
        audio_data, sample_rate = librosa.load(self.audioname, mono = True, duration = 30)
        
        def delete_treshold_segments(audio_data):
            if audio_data.size != 0:
                while abs(audio_data[0]) < self.treshold:
                        audio_data = audio_data[1:]

                return audio_data
            else:
                return audio_data

        left_deleted = delete_treshold_segments(audio_data)
        right_deleted = delete_treshold_segments(np.flip(left_deleted))
        croped_audio_data = np.flip(right_deleted)
        
        return croped_audio_data, sample_rate

    def features(self, audio_file, sample_rate):
        '''
        Function for features extraction using librosa library

        returns six features from audio file
        '''

        chroma_stft = librosa.feature.chroma_stft(y = audio_file, sr = sample_rate)
        spec_cent = librosa.feature.spectral_centroid(y = audio_file, sr = sample_rate)
        spec_bw = librosa.feature.spectral_bandwidth(y = audio_file, sr = sample_rate)
        rolloff = librosa.feature.spectral_rolloff(y = audio_file, sr = sample_rate)
        zcr = librosa.feature.zero_crossing_rate(audio_file)
        mfcc = librosa.feature.mfcc(y = audio_file, sr = sample_rate)

        return chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc