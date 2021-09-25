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
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


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
    
    def image_augmentation(self):
        """ Perform image augmentation and increase the size of the training set """
        
        train_datagen = ImageDataGenerator(
        rescale=1./255,       # Rescale all pixel values from 0-255, so after this step all the pixel values are in range (0,1)
        shear_range=0.2,      # Apply random transformations
        zoom_range=0.2,       # Apply zoom
        horizontal_flip=True) # Flip horizontally = ImageDataGenerator(rescale=1./255)

        training_set = train_datagen.flow_from_directory('./model_input/train', target_size=(64, 64),
                                                                         batch_size=32,
                                                                         class_mode='categorical',
                                                                         shuffle = False)
        
        test_set = test_datagen.flow_from_directory('./model_input/val', target_size=(64, 64),
                                                                  batch_size=32,
                                                                  class_mode='categorical',
                                                                  shuffle = False )

    def compile_network(self):
        """ Fit train data to the model """

        model = Sequential()
        input_shape=(64, 64, 3)#1st hidden layer
        model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=input_shape))
        model.add(AveragePooling2D((2, 2), strides=(2,2)))
        model.add(Activation('relu'))#2nd hidden layer
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(AveragePooling2D((2, 2), strides=(2,2)))
        model.add(Activation('relu'))#3rd hidden layer
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(AveragePooling2D((2, 2), strides=(2,2)))
        model.add(Activation('relu'))#Flatten
        model.add(Flatten())
        model.add(Dropout(rate=0.5))#Add fully connected layer.
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(rate=0.5))#Output layer
        model.add(Dense(10))
        model.add(Activation('softmax'))model.summary()

        epochs = 200
        batch_size = 8
        learning_rate = 0.01
        decay_rate = learning_rate / epochs
        momentum = 0.9
        sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
        model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])
        

    def predict(self, x_test, y_test):
        """ Classify new data based on the test dataset """
        pass

if __name__ == "__main__":
    
   

        
