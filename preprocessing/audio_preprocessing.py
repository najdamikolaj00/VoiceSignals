import csv
import os.path
import librosa
import librosa.display
import collections
import numpy as np
from pydub import AudioSegment
import numpy as np
import pandas as pd
import os
import csv
import warnings
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler



class AudioPreprocessing(object):

    def __init__(self):
        # Constructor
        self.data = collections.defaultdict(list)
        self.audio_file = []
        self.sample_rate = 0

    def import_audio(self, path):
        # import audio data and sample rate to dictionary
        # program sprawdzony działa w playground najda
        rootdir = path
        for subdir, dirs, files in os.walk(rootdir):
            for filename in files:
                if filename.endswith('.wav'):
                    pass
                else:
                    AudioSegment.from_file(os.path.join(subdir, filename), os.path.splitext(filename)[1][1:]).export(os.path.join(subdir
                    , filename.split(".")[0] + '.wav'), format = 'wav')
                    os.remove((os.path.join(subdir, filename)))

                audio_data, sample_rate = librosa.load(os.path.join(subdir, filename))
                self.data["audio_data"].append(audio_data)
                self.data["sample_rate"].append(sample_rate)
                self.data["sobriety"].append(filename.split('.wav')[0])
    

    def crop_audio(self, threshold, boundary = 'both'):
        # Crop audio to start with a threshold value
        def delete_treshold_segments(audio_file, threshold):
            audio_file = audio_file
            while abs(audio_file[0]) < threshold:
                    audio_file = audio_file[1:]

            return audio_file

        audio_file = self.audio_file
        reversed_audio_file = np.flip(audio_file)

        if boundary == 'left':
            self.audio_file = delete_treshold_segments(audio_file, threshold)
        elif boundary == 'right':
            self.audio_file = np.flip(delete_treshold_segments(reversed_audio_file, threshold))
        elif boundary == 'both':
            left_deleted = delete_treshold_segments(audio_file, threshold)
            right_deleted = delete_treshold_segments(np.flip(left_deleted), threshold)
            final = np.flip(right_deleted)
            self.audio_file = final
    
        return self.audio_file


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

    def convert_to_png(self, filename_path):
        cmap = plt.get_cmap('inferno')
        plt.specgram(self.audio_file, NFFT = 2048, Fs = 2, Fc = 0, noverlap = 128, cmap = cmap, sides = 'default', mode = 'default', scale = 'dB')
        plt.axis('off')
        plt.savefig(f'{filename_path[:-3].replace(".", "")}.png')
        plt.clf()


    def save_to_csv(self, path):
        # function will make a csv file with good headers of features names
        # program sprawdzony działa w playground najda

        header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
        for i in range(1, 21):
            header += f' mfcc{i}'
        header += ' label'
        header = header.split()

        with open(path, 'w', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow(header)

        

            for i in range(len(self.data['sobriety'])):

                chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc = self.features(self.data['audio_data'][i], self.data['sample_rate'][i])
                
                filename = self.data['sobriety'][i]
                to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
                
                for e in mfcc:
                    to_append += f' {np.mean(e)}'

                if any(substring in self.data['sobriety'][i] for substring in ["sober", "unsober"]):
                    label = ["sober" if "sober" in self.data['sobriety'][i] else "unsober"][0]
                else:
                    label = "unknown"
                
                to_append += f" {label}"
                                
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        
    def model_data_split(self, filename): 
        """ Method for performing train-test split on the data from the selected file """

        data = pd.read_csv(filename)
        data.head() # Dropping unneccesary columns
        sobriety_list = data.iloc[:, -1]
        encoder = LabelEncoder()
        y = encoder.fit_transform(sobriety_list)#Scaling the Feature columns
        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5)

        return X_train, X_test, y_train, y_test


    def plot_mfcc(self):
        # Plot mfcc transform
        mffc = self.mfcc()
        librosa.display.specshow(mffc)
        plt.xlabel("Time")
        plt.ylabel("MFCC")
        plt.title("MFFC")
        plt.colorbar()
        plt.show()

    def plot_fft(self):
        # Plots the fourier transform of the audio data
        fft = self.fft()
        magnitude = np.abs(fft)
        frequency = np.linspace(0, self.sample_rate, len(magnitude))
        plt.plot(frequency, magnitude)
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.title("Fourier Transform")
        plt.show()

    def plot_timeseries(self):
        # Plot timeseries
        plt.figure(figsize=(14, 6))
        librosa.display.waveplot(self.audio_file, sr = self.sample_rate)
        plt.title(f'"{self.filename}"" time-series (sample rate = {self.sample_rate})')
        plt.show()

    def plot_filtered_spectrogram_comparison(self):
    
        S_full, phase = librosa.magphase(librosa.stft(self.audio_file))

        plt.figure(figsize = (12, 4))
        
        librosa.display.specshow(librosa.amplitude_to_db(S_full, ref = np.max), y_axis = 'log', x_axis = 'time', sr = self.sample_rate)

        S_filter = librosa.decompose.nn_filter(S_full, aggregate = np.median, metric = 'cosine', width = int(librosa.time_to_frames(2, sr = self.sample_rate)))
        S_filter = np.minimum(S_full, S_filter)
        margin_i, margin_v = 2, 10
        power = 2

        mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power = power)
        mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power = power)

        S_foreground = mask_v * S_full
        S_background = mask_i * S_full

        # Plot spectrum
        plt.subplot(3, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(S_full, ref = np.max), y_axis = 'log', sr = self.sample_rate)
        plt.title('Full spectrum')
        plt.colorbar()

        plt.subplot(3, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(S_background, ref = np.max), y_axis = 'log', sr = self.sample_rate)
        plt.title('Background')
        plt.colorbar()

        plt.subplot(3, 1, 3)
        librosa.display.specshow(librosa.amplitude_to_db(S_foreground, ref = np.max), y_axis = 'log', x_axis = 'time', sr = self.sample_rate)
        plt.title('Foreground')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

