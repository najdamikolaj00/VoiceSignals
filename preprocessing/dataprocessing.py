'''
Data preprocessing, building an ANN model and fit the model
'''
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from PIL import Image
# import pathlib
# import csv 
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import keras
# from keras import layers
# from keras.models import Sequential 
# import warnings
# warnings.filterwarnings('ignore')
# from save_spectograms import save_spectrograms
# from converter import converter


data = pd.read_csv('dataset.csv')
# data.head()# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)#Encoding the Labels
print(data.head())
# genre_list = data.iloc[:, -1]
# encoder = LabelEncoder()
# y = encoder.fit_transform(genre_list)#Scaling the Feature columns
# scaler = StandardScaler()
# X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = Sequential()
# model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# classifier = model.fit(X_train,
#                     y_train,
#                     epochs=10,
#                     batch_size=128)