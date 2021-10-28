
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
    import matplotlib.pyplot as plt
    x = [86.36, 100, 96.43, 89.66]
    plt.figure()
    plt.hist(x)
    plt.show()           

if __name__ == '__main__':
    main()