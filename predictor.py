import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from models.sklearn_models import NaiveBayes, SVM, DecisionTree, KNN, Logistic
from models.neural_network import NeuralNetwork
from sklearn.preprocessing import StandardScaler
import time

# hide annoying sklearn warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)

# read in data, output a dataframe
def read_data():
    # load cleaned/binned data
    data = pd.read_csv("uber_lyft_preference.csv")

    # map locations to integer values
    # consider source/destination -> geographic coordinates ?
    location_map = {
        "Haymarket Square": 0,
        "Back Bay": 1,
        "North End": 2,
        "North Station": 3,
        "Beacon Hill": 4,
        "Fenway": 5,
        "Boston University": 6,
        "South Station": 7,
        "Theatre District": 8,
        "West End": 9,
        "Financial District": 10,
        "Northeastern University": 11
    }
    data['source'] = data['source'].map(location_map)
    data['destination'] = data['destination'].map(location_map)

    # map product type to integers
    product_map = {
        "shared": 1,
        "standard": 2,
        "xl": 3,
        "black_xl": 4,
        "lux": 5,
    }
    data['type'] = data['type'].map(product_map)

    # map cab type to integers
    cab_type_map = {
        "Lyft": 0,
        "Uber": 1
    }
    data['preferred'] = data['preferred'].map(cab_type_map)

    return data

# separate out features from target vals
data = read_data()
# data = data[0:1000]
features = data[['day_of_week', 'hour', 'source', 'destination', 'type', 'avg_distance']]
target = data['preferred']
scaled_features = StandardScaler().fit_transform(features)
# create training and testing sets

x_trn, x_tst, y_trn, y_tst = train_test_split(scaled_features, target, random_state=300)

# train and test models
start_time = time.time()
models = [NaiveBayes(), DecisionTree(), KNN(), Logistic(), SVM(), NeuralNetwork()]
for model in models:
    print(model.name + ':')
    hyperparams = model.tune_hyperparameters(x_trn, y_trn)
    print('\tHyperparameters:', hyperparams)
    model.train(x_trn, y_trn, hyperparams)
    # evaluate accuracy & print results
    training_accuracy = model.accuracy(x_trn, y_trn)
    testing_accuracy = model.accuracy(x_tst, y_tst)
    print('\tTraining Accuracy:', training_accuracy)
    print("\tTesting Accuracy:", testing_accuracy)

elapsed_time = time.time() - start_time
print("Ran for ", elapsed_time, "seconds")