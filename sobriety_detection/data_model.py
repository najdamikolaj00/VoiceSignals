from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score

class Data_model(object):

    def __init__(self, y, X):
        
        self.y = y
        self.X = X
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_fs = None
        self.X_test_fs = None

    def train_split(self, test_size, random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = test_size, random_state = random_state)

    def feature_selection(self, method):

        if method == 'ANOVA':
            fs = SelectKBest(score_func = f_classif, k = 4)
            fs.fit(self.X_train, self.y_train)
            self.X_train_fs = fs.transform(self.X_train)
            self.X_test_fs = fs.transform(self.X_test)

    def model_class_forest(self, n_estimators, max_depth, random_state):
        if self.X_train_fs is not None:
            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state)
            model.fit(self.X_train_fs, self.y_train)
            prediction_values = model.predict(self.X_test_fs)
            accuracy = accuracy_score(self.y_test, prediction_values)
            print('Accuracy: %.2f' % (accuracy*100))
        else:
            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state)
            model.fit(self.X_train, self.y_train)
            prediction_values = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, prediction_values)
            print('Accuracy: %.2f' % (accuracy*100))
            