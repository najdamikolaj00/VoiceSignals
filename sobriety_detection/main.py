from model import SobrietyModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from audio_split import audio_split
from noisereduction import noisereduce
import os
import sys
import xgboost as xgb
from sklearn.model_selection import train_test_split
sys.path.insert(0, '../preprocessing')
from audio_preprocessing import AudioPreprocessing

def main():
    
    '''
    Second value required dictionary name and .csv file name like: (id)(age)(sex)(number of  a test) from csv file idagesextest.csv

    Our data variables: 
    Input variables 'numerical': all features are numerical variables
    Output variable 'target(categorical)': sobriety =  sober or unsober

    It means that feature selection method should be ANOVA or Kendall's

    ANOVA correlation coefficient (linear)
    Kendall's rank coefficient (nonlinear)
    '''

    # #Sklearn approach 66.67 accuracy parameters: 12, 2, 5
    # obj_3 = Data_model(y, X, y_test, X_test)
    # obj_3.train_split(0.25, 1)
    # obj_3.feature_selection('ANOVA')
    # obj_3.model_class_forest(12, 2, 5)


    #Catboost approach accuracy//tmp out of use
    # obj_3 = Data_model(y, X, y_test, X_test
    # obj_3.train_split(0.25, 1)
    # obj_3.feature_selection('ANOVA')
    # obj_3.model_class_catboost()


    ''' Below is an example how to use the AudioPreprocessing class'''
   
    '''
    data = AudioPreprocessing() # Initializes the class instance

    data.import_audio(SOME_PATH) # Imports all the audio files contained in the folder specified by the path (converts non .wav to .wav too)

    data.save_csv(SOME_PATH_WITH_FILENAME) # Extracts all the features from the loaded audio files and stores them in a .csv where each loaded file has its own row with data
                                     # If a file from a given row didn't have "sober" or "unsober" in the filename, the label for that row will be "unknown"

    # Performs a train-test split on the data from the specified .csv file - data ready to be fed to the model
    X_train, X_test, y_train, y_test = data.model_data_split("../features_in_csv/025frames/mixed025all.csv")
    '''
    
    model = xgb.XGBClassifier()
    # model = RandomForestClassifier()
    # model = CatBoostClassifier()
    model.fit(X_train, y_train)
    prediction_values = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction_values)
    print('Accuracy: %.2f' % (accuracy*100))
    print(prediction_values)
    
if __name__ == '__main__':
    main()