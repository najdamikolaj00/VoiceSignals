import noisereduce as nr
import librosa
import soundfile as sf
import os

def noisereduce() -> None:
    rootdir = 'voice_data'
    for subdir, dirs, files in os.walk(rootdir):
        for filename in files:
            data, rate = librosa.load(os.path.join(subdir, filename))
            reduced_noise = nr.reduce_noise(y=data, sr=rate)
            sf.write(os.path.join(subdir, filename), reduced_noise, rate)

noisereduce()
