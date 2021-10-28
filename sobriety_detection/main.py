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


'''
Steps to make:
1. Load files
2. Check if they are already converted to .wav extension if they are not --> convert them
3. Make a .csv file
4. Feature extraction
5. Save features to a csv file
6. Data preprocessing
7. Model approach
8. Check the scores, save results
9. Save the results and find different solutions of data preprocessing (so back to step 6.)

Program will contain classes responsible for different steps 
First class --> steps: 1, 2, 3, 5, 9.
Second class --> steps: 4.
Third class --> steps: 6.
Fourth class --> steps: 7, 8.
'''

from converter import converter
from data_operator import Data_operator
from data_preprocessing import Data_processing
from data_model import Data_model
import pandas as pd


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