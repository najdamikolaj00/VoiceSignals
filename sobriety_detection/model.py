from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import sys
import xgboost as xgb
sys.path.insert(0, '../preprocessing')
#rom audio_preprocessing import AudioPreprocessing

#x = AudioPreprocessing()

class SobrietyModel(object):

    def __init__(self, y, X, y_test_data, X_test_data) -> None:
        
        '''
        Operator
        '''

        '''
        Program required this two variables:
        '''
        self.y = y # labels = categorical values
        self.X = X # features = numerical values

        '''
        If we want to check real predctions we use dataset from different time.
        So we need to provide this two variables to model (but they are not necessary if
        you are working only on training set):
        '''
        self.y_test_data = y_test_data
        self.X_test_data = X_test_data

        '''
        Program will use this variables if you are working only on training set and
        you are using train_test_split:
        '''
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        '''
        Program will use this variables if you are going to do feature_selection with ANOVA method:
        '''

        self.X_train_fs = None
        self.X_test_fs = None
        self.y_test_fs = None

    def train_split(self, test_size: float, random_state: int) -> None:

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = test_size, random_state = random_state)

    def feature_selection(self, method):

        if method == 'ANOVA':
            if self.X_test_data is not None:

                fs = SelectKBest(score_func = f_classif, k = 4)
                fs.fit(self.X, self.y)
                self.X_train_fs = fs.transform(self.X)
                self.X_test_fs = fs.transform(self.X_test_data)
                self.y_test_fs = self.y_test_data

            elif self.X_train is not None:

                fs = SelectKBest(score_func = f_classif, k = 4)
                fs.fit(self.X_train, self.y_train)
                self.X_train_fs = fs.transform(self.X_train)
                self.X_test_fs = fs.transform(self.X_test)
                self.y_test_fs =self.y_test

    def model_class_forest(self, n_estimators: int, max_depth: int, random_state: float) -> None:

        if self.X_train_fs is not None and self.X_train is not None:
            
            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state)
            model.fit(self.X_train_fs, self.y_train)
            prediction_values = model.predict(self.X_test_fs)
            accuracy = accuracy_score(self.y_test_fs, prediction_values)
            print('Accuracy: %.2f' % (accuracy*100))

        elif self.X_test_data is not None and self.X_train_fs is not None:

            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state)
            model.fit(self.X_train_fs, self.y)
            prediction_values = model.predict(self.X_test_fs)
            accuracy = accuracy_score(self.y_test_data, prediction_values)
            print('Accuracy: %.2f' % (accuracy*100))

        elif self.X_test_data is not None and self.X_train_fs is None:

            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state)
            model.fit(self.X, self.y)
            prediction_values = model.predict(self.X_test_data)
            accuracy = accuracy_score(self.y_test_data, prediction_values)
            print(prediction_values)
            print('Accuracy: %.2f' % (accuracy*100))

        else:

            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state)
            model.fit(self.X_train, self.y_train)
            prediction_values = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, prediction_values)
            print('Accuracy: %.2f' % (accuracy*100))

    def model_class_xgboost(self, n_estimators: int, max_depth: int, random_state: float) -> None:
        
        data_dmatrix = xgb.DMatrix(data = self.X_train, label = self.y_train)
        model = xgb.XGBClassifier(max_depth = max_depth, n_estimators = n_estimators)
        model.fit(self.X, self.y)
        prediction_values = model.predict(self.X_test_data)
        accuracy = accuracy_score(self.y_test_data, prediction_values)
        print(prediction_values)
        print('Accuracy: %.2f' % (accuracy * 100))

    #tmp out of use
    # def model_class_catboost(self) -> None:

    #     if self.X_train_fs is not None:

    #         model = CatBoostClassifier()
    #         model.fit(self.X_train_fs, self.y_train)
    #         prediction_values = model.predict(self.X_test_fs)
    #         print(self.y_test_data)
    #         print(prediction_values)
    #         accuracy = accuracy_score(self.y_test_fs, prediction_values)
    #         print('Accuracy: %.2f' % (accuracy*100))

    #     elif self.X_test_data is not None:

    #         model = CatBoostClassifier()
    #         model.fit(self.X, self.y)
    #         prediction_values = model.predict(self.X_test_data)
    #         print(self.y_test_data)
    #         print(prediction_values)
    #         accuracy = accuracy_score(self.y_test_data, prediction_values)
    #         print('Accuracy: %.2f' % (accuracy*100))

    #     else:

    #         model = CatBoostClassifier()
    #         model.fit(self.X_train, self.y_train)
    #         prediction_values = model.predict(self.X_test)
    #         accuracy = accuracy_score(self.y_test, prediction_values)
    #         print('Accuracy: %.2f' % (accuracy*100))
            