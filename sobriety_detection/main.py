from data_model import SobrietyModel
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

    

    # #Sklearn approach 66.67 accuracy parameters: 12, 2, 5
    # obj_3 = Data_model(y, X, y_test, X_test)
    # obj_3.train_split(0.25, 1)
    # obj_3.feature_selection('ANOVA')
    # obj_3.model_class_forest(12, 2, 5)


    #Catboost approach accuracy//tmp out of use
    # obj_3 = Data_model(y, X, y_test, X_test)
    # obj_3.train_split(0.25, 1)
    # obj_3.feature_selection('ANOVA')
    # obj_3.model_class_catboost()


    data = AudioPreprocessing()
    X_train, X_test, y_train, y_test = data.model_data_split("../features_in_csv/025frames/mixed025all.csv")

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