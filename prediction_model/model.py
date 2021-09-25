""" 
This file stores a deep learning model for classifying the audio spectrogram data
from the collected samples
"""

# import librosa
# import librosa.display
# import random
# import warnings
import os
# from PIL import Image
import pathlib
import shutil
import splitfolders
# import csv
# # sklearn Preprocessing
# from sklearn.model_selection import train_test_split
# #Keras
# import keras
# import warnings
# warnings.filterwarnings('ignore')
# from keras import layers
# from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
# from keras.models import Sequential
# from keras.optimizers import SGD


# HOW DOES IT WORK?
# Step 1: Access the folder where all the spectrograms are stored after audio preprocessing
# Step 2: Walk through all the specotragrams and split them into sets comprising sober and intoxicated people samples
# Step 3: Perform label encoding for the entire dataset
# Step 4: Split the spectrograms into training and test datasets
# Step 5: Perform image augmentation to increase the number of training samples
# Step 6: Build a convolutional neural network
# Step 7: Compile the network and evalue the model
# Step 8: Display results 


class Sobriety():

    def __init__(self):
        """ Constructor """
        pass


     def load_data(self):
         """ Load the spectrogram data to the class """

        PATH = "../spectrograms"
        for instance in ['sober', 'intoxicated']:
            pathlib.Path(f'classification_data/{instance}').mkdir(parents = True, exist_ok = True)
            for file in os.listdir(PATH):
                filepath = lambda file: os.path.join(PATH, file)
                
                if "unsober" in file:
                    os.rename(filepath(file), filepath(file.replace("unsober", "intoxicated")))
                
                if instance in file:
                    shutil.copy(filepath(file), f'classification_data/{instance}/{file}')

        splitfolders.ratio('./classification_data/', output= "./model_input", seed = 1337, ratio = (.8, .2)) 
    
    def fit(self, x_train, y_train):
        """ Fit train data to the model """
        pass

    def predict(self, x_test, y_test):
        """ Classify new data based on the test dataset """
        pass

if __name__ == "__main__":
    
   

        

    load_data()