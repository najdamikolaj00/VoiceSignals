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

from data_operator import Data_Operator


def main():

    obj = Data_operator('data_test2')
    obj.import_audio()

from data_operator import Data_operator
from data_preprocessing import Data_processing
from data_model import Data_model

def main():
    
    '''
    Test on men's voice: model learns on data collected one month ago before collecting test dataset.
    '''

    '''
    Data from August(model is learning on them):
    '''
    obj = Data_operator('voice_data\data_test5', 'features_in_csv\datasetwojtek_august.csv')
    obj.import_audio()

    '''
    Data from Semptember(we are going to check accuracy of predicion values)
    '''

    obj = Data_operator('voice_data\data_test6', 'features_in_csv\datasetwojtek_september.csv')
    obj.import_audio()

    '''
    Our data variables: 
    Input variables 'numerical': all features are numerical variables
    Output variable 'target(categorical)': sobriety =  sober or unsober

    It means that feature selection method should be ANOVA or Kendall's

    ANOVA correlation coefficient (linear)
    Kendall's rank coefficient (nonlinear)
    '''

    obj_2 = Data_processing('features_in_csv\datasetwojtek_august.csv')
    y, X = obj_2.transform_data()

    obj_test = Data_processing('features_in_csv\datasetwojtek_september.csv')
    y_test, X_test = obj_test.transform_data()

    #Sklearn approach 66.67 accuracy parameters: 12, 2, 5
    obj_3 = Data_model(y, X, y_test, X_test)
    # obj_3.train_split(0.25, 1)
    # obj_3.feature_selection('ANOVA')
    obj_3.model_class_forest(12, 2, 5)

    #Catboost approach accuracy//tmp out of use
    # obj_3 = Data_model(y, X, y_test, X_test)
    # obj_3.train_split(0.25, 1)
    # obj_3.feature_selection('ANOVA')
    # obj_3.model_class_catboost()

if __name__ == '__main__':
    main()