import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Data_processing(object):

    def __init__(self, filename):
        self.file_name = filename

    def transform_data(self): 
        data = pd.read_csv(self.file_name)
        data.head() # Dropping unneccesary columns
        data = data.drop(['filename'], axis=1) #Encoding the Labels
        sobriety_list = data.iloc[:, -1]
        encoder = LabelEncoder()
        y = encoder.fit_transform(sobriety_list)#Scaling the Feature columns
        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
        return y, X
        
