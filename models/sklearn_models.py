from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# abstract class that represents a model using the sklearn library
class SklearnModel:
    def __init__(self):
        self.classifier = None
        self.name = None

    def train(self, x_trn, y_trn):
        self.classifier.fit(X=x_trn, y=y_trn)

    def predict(self, x_vals):
        return self.classifier.predict(X=x_vals)

    def accuracy(self, x_vals, y_vals):
        accuracy = self.classifier.score(x_vals, y_vals)
        return accuracy

class NaiveBayes(SklearnModel):
    def __init__(self):
        self.classifier = GaussianNB()
        self.name = "Gaussian Naive Bayes"

class SVM(SklearnModel):
    def __init__(self):
        self.classifier = SVC()
        self.name = "Support Vector Machine"

class DecisionTree(SklearnModel):
    def __init__(self):
        self.classifier = DecisionTreeClassifier()
        self.name = "Decision Tree"

class KNN(SklearnModel):
    def __init__(self):
        self.classifier = KNeighborsClassifier()
        self.name = "k-Nearest Neighbor"