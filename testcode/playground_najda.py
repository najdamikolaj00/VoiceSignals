import librosa
import os
import collections
import csv

data = collections.defaultdict(list) # import audio data and sample rate to dictionary
    

rootdir = 'data_test2'
for subdir, dirs, files in os.walk(rootdir):
    for filename in files:
        audio_data, sample_rate = librosa.load(os.path.join(subdir, filename))
        data["audio_data"].append(audio_data)
        data["sample_rate"].append(sample_rate)
        data["sobriety"].append(subdir.split('\\')[-1])

print(data)

# header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
# for i in range(1, 21):
#     header += f' mfcc{i}'
# header += ' label'
# header = header.split()

# file = open('dataset01.csv', 'w', newline='')
# with file:
#     writer = csv.writer(file)
#     writer.writerow(header)