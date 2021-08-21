'''
Data preprocessing, building an ANN model and fit the model
'''
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('dataset.csv')
data.head()# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)#Encoding the Labels
sobriety_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(sobriety_list)#Scaling the Feature columns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5)

def get_mae(maximum_n_estimators, train_X, val_X, train_y, val_y):
    model = RandomForestClassifier(n_estimators = maximum_n_estimators, max_depth = 3, random_state = 8)
    model.fit(train_X, train_y)
    prediction_values = model.predict(val_X)
    mae = mean_absolute_error(val_y, prediction_values)
    return mae

best_one = []
candidate_n_estimators = list(range(150, 160))

for i in candidate_n_estimators:
    my_mae = get_mae(i, X_train, X_test, y_train, y_test)
    best_one.append(my_mae)

best = min(best_one)
index_best = 0

for i in range(len(best_one)):
    if best_one[i] == best:
        index_best = i
   
print('Smallest MAE:', np.around(best, decimals = 5)*100 ,'% for the:', candidate_n_estimators[index_best], 'number of trees in the forest.')
