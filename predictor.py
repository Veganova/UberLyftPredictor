import pandas as pd
from models.logistic import Logistic

pd.set_option('display.max_columns', None)

# start with reformatted data
data = pd.read_csv("uber_lyft_preference_partial.csv")

# Mappings
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


# product type
product_map = {
    "shared": 1,
    "standard": 2,
    "xl": 3,
    "black_xl": 4,
    "lux": 5,
}
data['type'] = data['type'].map(product_map)

cab_type_map = {
    "Lyft": 0,
    "Uber": 1
}

data['cab_type'] = data['cab_type'].map(cab_type_map)

# condense size (mostly for quick testing)
reduced_size = int(len(data) * 0.05)
condensed_data = data.iloc[:reduced_size]

# include relevant columns and separate the classes from feature vector
filtered_data = condensed_data[['distance', 'source', 'destination', 'price', 'type', 'day_of_week', 'hour']]
classes = condensed_data['cab_type']

total_rows = len(filtered_data)
training_size = int(total_rows * 0.6)
test_size = total_rows - training_size

training_data = filtered_data.iloc[:training_size]
training_classes = classes.iloc[:training_size]
# test_data = data.iloc[training_size:]
# test_classes = classes.iloc[training_size:]


# set model
model = Logistic()

# train
model.train(training_data.values.tolist(), training_classes.values.tolist())

# test
# prediction = model.test

# get errors
# model.classification_error

# plot
# model.plot


