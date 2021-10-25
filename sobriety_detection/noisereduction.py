import noisereduce as nr
import librosa
import soundfile as sf
import os

def noisereduce() -> None:
    rootdir = 'audio_to_split025'
    for subdir, dirs, files in os.walk(rootdir):
        for filename in files:
            if 'bartek' in filename:
                data, rate = librosa.load(os.path.join(subdir, filename))
                reduced_noise = nr.reduce_noise(y=data, sr=rate)
                sf.write(os.path.join(subdir, filename), reduced_noise, rate)

noisereduce()
