'''
This program will show processing of audio files from scratch.
'''
'''
Importing necessary libraries
'''
import librosa
import numpy as np
import pandas as pd
import os
import csv
from pydub import AudioSegment 
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# our libraries
from converter import converter

'''
Converting files with different extension than .wav using our-made function converter()
'''

#converter() # This function doesn't need any arguments because it will check all files in the Data dict and it will return .wav files

'''
Next important step is to crop audio. After this audio file will include only sound without breakes at the beggining and in the end.
'''

'''
# Creating header file for .csv
# '''
# header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
# for i in range(1, 21):
#     header += f' mfcc{i}'
# header += ' label'
# header = header.split()

# file = open('dataset01.csv', 'w', newline='')
# with file:
#     writer = csv.writer(file)
#     writer.writerow(header)
# sobriety = 'sober unsober'.split()

# '''
# Feature extraction
# '''

# PARENT = os.path.dirname(os.getcwd())
# DIRECTORY = os.path.join(PARENT, "VoiceSignals\\Datano2")

# for s in sobriety:
#     for filename in os.listdir(os.path.join(DIRECTORY, s)):
#         audioname = os.path.join(DIRECTORY, s, filename)
#         y, sr = librosa.load(audioname, mono = True, duration = 30)

#         chroma_stft = librosa.feature.chroma_stft(y = y, sr = sr)
#         spec_cent = librosa.feature.spectral_centroid(y = y, sr = sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y = y, sr = sr)
#         rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)
#         zcr = librosa.feature.zero_crossing_rate(y)
#         mfcc = librosa.feature.mfcc(y = y, sr = sr)

#         to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
#         for e in mfcc:
#             to_append += f' {np.mean(e)}'
#         to_append += f' {s}'
#         file = open('dataset01.csv', 'a', newline='')
#         with file:
#             writer = csv.writer(file)
#             writer.writerow(to_append.split())

# '''
# Data preprocessing
# '''

def transform_data(filename): 
    data = pd.read_csv(filename)
    data.head() # Dropping unneccesary columns
    data = data.drop(['filename'], axis=1) #Encoding the Labels
    sobriety_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(sobriety_list)#Scaling the Feature columns
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5)
    return X_train, X_test, y_train, y_test

'''
Function for getting mean absolute error/trying to find best n_estimators number
'''

def get_mae(maximum_n_estimators, train_X, val_X, train_y, val_y):
    model = RandomForestClassifier(n_estimators = maximum_n_estimators, max_depth = 3, random_state = 8)
    model.fit(train_X, train_y)
    prediction_values = model.predict(val_X)
    mae = mean_absolute_error(val_y, prediction_values)
    return mae

best_one = []
candidate_n_estimators = list(range(25, 26))
X_train, X_test, y_train, y_test = transform_data('dataset01.csv')

for i in candidate_n_estimators:
    my_mae = get_mae(i, X_train, X_test, y_train, y_test)
    best_one.append(my_mae)

best = min(best_one)
index_best = 0

for i in range(len(best_one)):
    if best_one[i] == best:
        index_best = i
   
print('Smallest MAE:', np.around(best, decimals = 5)*100 ,'% for the:', candidate_n_estimators[index_best], 'number of trees in the forest.')
