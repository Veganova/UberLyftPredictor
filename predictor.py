import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score

from models.logistic import Logistic
from models.sklearn_models import NaiveBayes, SVM, DecisionTree, KNN
from sklearn.preprocessing import StandardScaler

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

    # condense size (mostly for quick testing)
    #reduced_size = int(len(data) * 0.005)
    #condensed_data = data.iloc[:reduced_size]

    # include relevant columns and separate the classes from feature vector
    # filtered_data = condensed_data[['distance', 'source', 'destination', 'price', 'type', 'day_of_week', 'hour']]
    # classes = condensed_data['cab_type']

    # total_rows = len(filtered_data)
    # training_size = int(total_rows * 0.6)
    # test_size = total_rows - training_size

    return data

# separate out features from target vals
data = read_data()
features, target = (data.iloc[:,:6], data["preferred"])
scaled_features = StandardScaler().fit_transform(features)
# create training and testing sets
x_trn, x_tst, y_trn, y_tst = train_test_split(scaled_features, target, random_state=3000)

# train and test models
models = [NaiveBayes(), DecisionTree(), KNN()]#, SVM()]
for model in models:
    model.train(x_trn, y_trn)

    # evaluate accuracy & print results
    training_accuracy = model.accuracy(x_trn, y_trn)
    testing_accuracy = model.accuracy(x_tst, y_tst)
    print(model.name + ':')
    print(f'\tTraining Accuracy: {training_accuracy:.2%}')
    print(f'\tTesting Accuracy: {testing_accuracy:.2%}')

# test
#prediction = model.test
# get training errors
#model.classification_error(x_trn, y_trn)
# plot
#model.plot(training_data, training_classes)


