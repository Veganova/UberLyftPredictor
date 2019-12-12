from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# abstract class that represents a model using the sklearn library
class SklearnModel:
    def __init__(self):
        self.classifier = None
        self.trained_model = None
        self.name = None
        self.param_grid = {}

    def train(self, x_trn, y_trn, hyperparams):
        self.trained_model = self.classifier(**hyperparams)
        self.trained_model.fit(X=x_trn, y=y_trn)

    def tune_hyperparameters(self, x_trn, y_trn):
        grid_search = GridSearchCV(self.classifier(), self.param_grid)
        grid_search.fit(x_trn, y_trn)
        best_params = grid_search.best_params_
        return best_params

    def predict(self, x_vals):
        return self.trained_model.predict(X=x_vals)

    def accuracy(self, x_vals, y_vals):
        accuracy = self.trained_model.score(x_vals, y_vals)
        return accuracy

class NaiveBayes(SklearnModel):
    def __init__(self):
        self.classifier = GaussianNB
        self.name = "Gaussian Naive Bayes"
        self.param_grid = {}

class SVM(SklearnModel):
    def __init__(self):
        self.classifier = LinearSVC
        self.name = "Support Vector Machine"
        self.param_grid = {
            'C': [0.005, 0.01, 0.1, 1, 10, 20],
            #'penalty': ['l1', 'l2'],
            #'loss': ['hinge', 'squared_hinge'],
            'multi_class': ['ovr', 'crammer_singer'],
            'dual': [False], # recommended for large datasets
            'max_iter': 3000 # default was too low
        }

class DecisionTree(SklearnModel):
    def __init__(self):
        self.classifier = DecisionTreeClassifier
        self.name = "Decision Tree"
        self.param_grid = {'criterion': ['gini', 'entropy']}

class KNN(SklearnModel):
    def __init__(self):
        self.classifier = KNeighborsClassifier
        self.name = "k-Nearest Neighbor"
        self.param_grid = {'n_neighbors': [5, 10, 15, 18, 20, 22, 25]}