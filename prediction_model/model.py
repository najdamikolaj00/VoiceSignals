""" 
This file stores a deep learning model for classifying the audio spectrogram data
from the collected samples
"""

# import librosa
# import librosa.display
# import random
# import warnings
import os
import numpy as np
# from PIL import Image
import pathlib
import shutil
import splitfolders
import csv
import pandas as pd
# # sklearn Preprocessing
# from sklearn.model_selection import train_test_split
# #Keras
import keras
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
        self.training_set = Sequential()
        self.test_set = Sequential()
        self.model = Sequential()


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
        
        train_datagen = ImageDataGenerator(rescale=1./255,       # Rescale all pixel values from 0-255, so after this step all the pixel values are in range (0,1)
                                           shear_range=0.2,      # Apply random transformations
                                           zoom_range=0.2,       # Apply zoom
                                           horizontal_flip=True) # Flip horizontally = ImageDataGenerator(rescale=1./255)

        test_datagen = ImageDataGenerator( rescale=1./255,       # Rescale all pixel values from 0-255, so after this step all the pixel values are in range (0,1)
                                           shear_range=0.2,      # Apply random transformations
                                           zoom_range=0.2,       # Apply zoom
                                           horizontal_flip = True) # Flip horizontally = ImageDataGenerator(rescale=1./255)

        self.training_set = train_datagen.flow_from_directory('./model_input/train', target_size = (64, 64),
                                                                         batch_size = 8,
                                                                         class_mode = 'binary',
                                                                         shuffle = True)
        
        self.test_set = test_datagen.flow_from_directory('./model_input/val', target_size = (64, 64),
                                                                  batch_size = 8,
                                                                  class_mode = 'binary',
                                                                  shuffle = True)


    def compile_network(self):
        """ Fit train data to the model """

        input_shape = (64, 64, 3)#1st hidden layer
        self.model.add(Conv2D(32, (3, 3), strides = (2, 2), input_shape = input_shape))
        self.model.add(AveragePooling2D((2, 2), strides = (2,2)))
        self.model.add(Activation('relu'))#2nd hidden layer
        self.model.add(Conv2D(64, (3, 3), padding = "same"))
        self.model.add(AveragePooling2D((2, 2), strides = (2,2)))
        self.model.add(Activation('relu'))#3rd hidden layer
        self.model.add(Conv2D(64, (3, 3), padding = "same"))
        self.model.add(AveragePooling2D((2, 2), strides = (2,2)))
        self.model.add(Activation('relu'))#Flatten
        self.model.add(Flatten())
        self.model.add(Dropout(rate = 0.5))#Add fully connected layer.
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(rate = 0.5))#Output layer
        self.model.add(Dense(1))
        self.model.add(Activation('softmax'))
        self.model.summary()

        epochs = 200
        batch_size = 2
        learning_rate = 0.01
        decay_rate = learning_rate / epochs
        momentum = 0.9
        sgd = SGD(lr = learning_rate, momentum = momentum, decay = decay_rate, nesterov = False)
        self.model.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics = ['accuracy'])
        
        self.model.fit(self.training_set,
                            epochs = 50,
                            validation_data = self.test_set,
                            validation_steps = 200)

        print(self.model.evaluate(self.test_set, steps = 50))

    def predict(self):
        """ Classify new data based on the test dataset """
        self.test_set.reset()
        pred = self.model.predict(self.test_set, steps=50, verbose=1)
        predicted_class_indices = np.argmax(pred, axis=1)
        labels = (self.training_set.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]
        predictions = predictions[:200]
        filenames = self.test_set.filenames
        results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})
        results.to_csv("prediction_results.csv",index = False)


if __name__ == "__main__":
    
    model = Sobriety()
    model.load_data()
    model.image_augmentation()
    model.compile_network()
    model.predict()
    
   

        
