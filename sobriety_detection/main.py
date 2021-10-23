from converter import converter
from data_operator import Data_operator
from data_preprocessing import Data_processing
from data_model import Data_model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from catboost import CatBoostClassifier
import xgboost as xgb

def main():
    
    '''
    Second value required dictionary name and .csv file name like: (id)(age)(sex)(number of  a test) from csv file idagesextest.csv
    '''
    # converter()
    # obj = Data_operator('voice_data', '221M65.csv', 'windows')
    # obj.import_audio()

    '''
    Our data variables: 
    Input variables 'numerical': all features are numerical variables
    Output variable 'target(categorical)': sobriety =  sober or unsober

    It means that feature selection method should be ANOVA or Kendall's

    ANOVA correlation coefficient (linear)
    Kendall's rank coefficient (nonlinear)
    '''
    # rootdir = 'features_in_csv'
    # for subdir, dirs, files in os.walk(rootdir):
    #     for filename in files:
    #         # if 'F' in filename:
    #         data = pd.read_csv(os.path.join(subdir, filename))
    #         data.to_csv('AllVoicestextdescription.csv', mode = 'a', header=False, index=False)
                
    obj_test = Data_processing('AllVoicestextdescription.csv')
    y, X = obj_test.transform_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,  random_state = 12)

    #model = xgb.XGBClassifier()
    #model = RandomForestClassifier()
    model = CatBoostClassifier()
    model.fit(X_train, y_train)
    prediction_values = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction_values)
    print('Accuracy: %.2f' % (accuracy*100))
    
    print(prediction_values)

if __name__ == '__main__':
    main()