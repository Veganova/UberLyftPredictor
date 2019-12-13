from sklearn.metrics import accuracy_score
import torch
import numpy as np
from torch.autograd import Variable

# define some possible activation functions
relu = torch.nn.ReLU()
sigmoid = torch.nn.Sigmoid()
hyperbolic = torch.nn.Tanh()

# three layer neural net (2 hidden layers)
class FeedForward3(torch.nn.Module):
    # layer sizes = list of length 4, activation_functions = list of length 3
    def __init__(self, layer_sizes, activation_functions): 
        super(FeedForward3, self).__init__()
        self.layer_sizes = layer_sizes
        self.af1 = activation_functions[0]
        self.af2 = activation_functions[1]
        self.af3 = activation_functions[2]
        self.weights1 = torch.nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.weights2 = torch.nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.weights3 = torch.nn.Linear(self.layer_sizes[2], self.layer_sizes[3])

    def forward(self, x):
        hidden1_in = self.weights1(x)
        hidden1_out = self.af1(hidden1_in)
        hidden2_in = self.weights2(hidden1_out)
        hidden2_out = self.af2(hidden2_in)
        final_in = self.weights3(hidden2_out)
        final_out = self.af3(final_in)
        return final_out.flatten()


class NeuralNetwork:
    def __init__(self):
        self.model = self.init_model()
        self.name = "Neural Network"

    def init_model(self, hidden_layer_sizes=[12, 8], layer_functions=[relu, relu, sigmoid]):
        return FeedForward3([6] + hidden_layer_sizes + [1], layer_functions)

    def train(self, x_trn, y_trn, params):
        self.model.train()
        print('beginning training')
        x_trn_tensor = self.tensor(x_trn)
        y_trn_tensor = self.tensor(y_trn.to_numpy())
        # train model
        optimizer = torch.optim.SGD(self.model.parameters(), lr=params["learning_rate"]) # use SGD to optimize
        for i in range(0, params["loop_size"]): # how to determine range??
            optimizer.zero_grad()  # Forward pass
            output = self.model(x_trn_tensor)
            criterion = torch.nn.MSELoss()
            loss = criterion(output, y_trn_tensor)
            # print('Epoch {}: train loss: {}'.format(i, loss.item()))
            loss.backward()
            optimizer.step()

    def tune_hyperparameters(self, x_trn, y_trn):
        reduced_size = 5000
        X = x_trn[0: reduced_size]
        Y = y_trn[0: reduced_size]

        hidden_layer_sizes = np.arange(3, 6, 1)
        activation_functions = (relu, sigmoid)
        loops = np.arange(500, 5000, 3000)
        learning_rates = np.arange(0.05, 0.11, 0.02)

        best_accuracy = 0
        best_params = {}
        for hidden_layer_size in hidden_layer_sizes:
            for activation_function in activation_functions:
                for loop_size in loops:
                    for learning_rate in learning_rates:
                        temp_params = {
                            "hidden_layer_size": hidden_layer_size,
                            "loop_size": loop_size,
                            "activiation_function": activation_function,
                            "learning_rate": learning_rate
                        }
                        self.model = self.init_model()
                        self.train(X, Y, temp_params)
                        accuracy = self.accuracy(X, Y)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = temp_params
                            print(accuracy, best_params)

        return {} # do nothing

    def predict(self, x_vals):
        self.model.eval()
        x_tensor = self.tensor(x_vals)

        # predict and convert to 0 or 1
        t = Variable(torch.Tensor([0.5]))  # threshold
        prediction = (self.model(x_tensor) > t).float() * 1
        return prediction.int()

    def accuracy(self, x_vals, y_vals):
        predicted = self.predict(x_vals)
        return accuracy_score(y_vals, predicted)

    # convert data to the tensor type used by torch
    def tensor(self, data):
        return torch.tensor(data).float()