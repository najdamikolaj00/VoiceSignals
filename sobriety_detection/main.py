from converter import converter
from data_operator import Data_operator
from data_preprocessing import Data_processing
from data_model import Data_model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from audio_split import audio_split
from noisereduction import noisereduce
import os
from sklearn.model_selection import train_test_split


def main():
    
    '''
    Second value required dictionary name and .csv file name like: (id)(age)(sex)(number of  a test) from csv file idagesextest.csv
    '''
    # converter()
    # noisereduce()
    # audio_split()

    
    # obj = Data_operator('folder_name', 'filename.csv', 'operating system')
    # obj.import_audio()
    # data = pd.read_csv('filename.csv')
    # data.to_csv('filename.csv', mode = 'a', header=False, index=False)
    

    '''
    Our data variables: 
    Input variables 'numerical': all features are numerical variables
    Output variable 'target(categorical)': sobriety =  sober or unsober

    It means that feature selection method should be ANOVA or Kendall's

    ANOVA correlation coefficient (linear)
    Kendall's rank coefficient (nonlinear)
    '''
    
    obj_test = Data_processing('textplusdescription025all.csv')
    y, X = obj_test.transform_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 12)

    #model = XGBoostClassifier()
    #model = RandomForestClassifier()
    model = CatBoostClassifier()
    model.fit(X_train, y_train)
    prediction_values = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction_values)
    print('Accuracy: %.2f' % (accuracy*100))
    print(prediction_values)
       
if __name__ == '__main__':
    main()