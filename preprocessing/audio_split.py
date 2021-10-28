'''
Program for splitting audio by frames of length of 25% of the audio length.

1. Load audio with librosa (so numpy array),
2. iterate over array spliting by 25% of audio length,
3. after 25%'s split convert to audio.wav
4. save in directory

'''
import soundfile as sf
import numpy as np
import librosa
import os


def audio_split() -> None:
    rootdir = 'audio_to_split025'
    savedir = 'audio_025'

    for subdir, dirs, files in os.walk(rootdir):
        for filename in files:
            audio_data, sample_rate = librosa.load(os.path.join(subdir, filename))
            array_025 = []
            array_050 = []
            array_075 = []
            array_100 = []
            for i in range(0, len(audio_data)):

                if i <= len(audio_data)*0.25:
                    array_025.append(audio_data[i])
                    if len(array_025) == np.around(len(audio_data)*0.25):
                        array_025_numpy = np.array(array_025)
                        if 'unsober' in filename:
                            sf.write(savedir + '/unsober/' + "025" + filename, array_025_numpy, 22050)
                        else:
                            sf.write(savedir + '/sober/' + "025" + filename, array_025_numpy, 22050)
                
                elif i <= len(audio_data)*0.50 and i > len(audio_data)*0.25:
                    array_050.append(audio_data[i])
                    if len(array_050) == np.around(len(audio_data)*0.25):
                        array_050_numpy = np.array(array_050)
                        if 'unsober' in filename:
                            sf.write(savedir + '/unsober/' + "050" + filename, array_050_numpy, 22050)
                        else:
                            sf.write(savedir + '/sober/' + "050" + filename, array_050_numpy, 22050)

                elif i <= len(audio_data)*0.75 and i > len(audio_data)*0.50:
                    array_075.append(audio_data[i])
                    if len(array_075) == np.around(len(audio_data)*0.25):
                        array_075_numpy = np.array(array_075)
                        if 'unsober' in filename:
                            sf.write(savedir + '/unsober/' + "075" + filename, array_075_numpy, 22050)
                        else:
                            sf.write(savedir + '/sober/' + "075" + filename, array_075_numpy, 22050)

                elif i <= len(audio_data) and i > len(audio_data)*0.75:
                    array_100.append(audio_data[i])
                    if len(array_100) == np.around(len(audio_data)*0.25):
                        array_100_numpy = np.array(array_100)
                        if 'unsober' in filename:
                            sf.write(savedir + '/unsober/' + "100" + filename, array_100_numpy, 22050)
                        else:
                            sf.write(savedir + '/sober/' + "100" + filename, array_100_numpy, 22050)
    