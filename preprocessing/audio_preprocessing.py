import matplotlib.pyplot as plt
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

    def __init__(self, filename, audio_file, sample_rate):
        # Constructor
        self.filename = filename
        self.audio_file = audio_file
        self.sample_rate = sample_rate

    def crop_audio(self, threshold, boundary = 'both'):
        # Crop audio to start with a threshold value
        def delete_treshold_segments(audio_file, threshold):
            audio_data = audio_file
            while abs(audio_data[0]) < threshold:
                    audio_data = audio_data[1:]

            return audio_data

        audio_data = self.audio_file
        reversed_audio_data = np.flip(audio_data)

        if boundary == 'left':
            self.audio_file = delete_treshold_segments(audio_data, threshold)
        elif boundary == 'right':
            self.audio_file = np.flip(delete_treshold_segments(reversed_audio_data, threshold))
        elif boundary == 'both':
            left_deleted = delete_treshold_segments(audio_data, threshold)
            right_deleted = delete_treshold_segments(np.flip(left_deleted), threshold)
            final = np.flip(right_deleted)
            self.audio_file = final
    
        return self

    def convert_to_png(self, filename_path):
        cmap = plt.get_cmap('inferno')
        plt.specgram(self.audio_file, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
        plt.axis('off')
        plt.savefig(f'{filename_path[:-3].replace(".", "")}.png')
        plt.clf()
    
    def mfcc(self):
        # Compute audio signal mfcc
        return librosa.feature.mfcc(self.audio_file, sr = self.sample_rate)
    
    def fft(self):
        # Compute Fourier transform
        return np.fft.fft(self.audio_file)
    
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

