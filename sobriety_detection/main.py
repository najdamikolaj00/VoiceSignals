from converter import converter
from data_operator import Data_operator
from data_preprocessing import Data_processing
from data_model import Data_model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
    # data = pd.read_csv('features_in_csv/BadanieTekst/221M1.csv')
    # data_1 = pd.read_csv('features_in_csv/BadanieTekst/221M2.csv')
    # data_2 = pd.read_csv('features_in_csv/BadanieTekst/221M3.csv')
    # data.to_csv('221Mlearn.csv', mode = 'a', header=False, index=False)
    # data_1.to_csv('221Mlearn.csv', mode = 'a', header=False, index=False)
    # data_2.to_csv('221Mlearn.csv', mode = 'a', header=False, index=False)
    # data_1 = pd.read_csv('features_in_csv/BadanieTekst/221M65.csv')
    # data_2 = pd.read_csv('features_in_csv/BadanieTekst/221M5.csv')
    # data_1.to_csv('221Mlearn.csv', mode = 'a', header=False, index=False)
    # data_2.to_csv('221Mlearn.csv', mode = 'a', header=False, index=False)
    
    obj_train = Data_processing('221Mlearn.csv')
    y_train, X_train = obj_train.transform_data()

    obj_test = Data_processing('221MTest.csv')
    y_test, X_test = obj_test.transform_data()
    
    # model = Data_model(y_train, X_train, y_test, X_test)
    # model.model_class_forest()
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    prediction_values = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction_values)
    print('Accuracy: %.2f' % (accuracy*100))
    
    print(prediction_values)
       
    
    

if __name__ == '__main__':
    main()