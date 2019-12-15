from sklearn.metrics import accuracy_score
import torch
import numpy as np
from torch.autograd import Variable

# define some possible activation functions
relu = torch.nn.ReLU
sigmoid = torch.nn.Sigmoid
hyperbolic = torch.nn.Tanh
linear = torch.nn.Linear

# three layer neural net (2 hidden layers)
class FeedForward3(torch.nn.Module):
    # layer sizes = list of length 4, activation_functions = list of length 3
    def __init__(self, input_size, layer_sizes, activation_functions):
        super(FeedForward3, self).__init__()
        self.input_size = input_size
        self.ac_funcs = []

        size = self.input_size
        for i in range(len(layer_sizes)):
            new_layer_size = layer_sizes[i]
            ac_func = activation_functions[i]

            layer_lin = linear(size, new_layer_size)
            # same as doing self.af1 = layer_lin (for for any i)
            self.__setattr__("af" + str(i + 1), layer_lin)
            layer_config = (layer_lin, ac_func())
            self.ac_funcs.append(layer_config)

            size = new_layer_size

    def forward(self, x):
        result = x
        for layer in self.ac_funcs:
            collector = layer[0]
            func = layer[1]
            result = func(collector(result))
        return result.flatten()


class NeuralNetwork:
    def __init__(self):
        self.model = self.init_model()
        self.name = "Neural Network"

    def init_model(self, hidden_layer_sizes=[12, 8], layer_functions=[relu, relu]):
        return FeedForward3(6, hidden_layer_sizes + [1], layer_functions + [sigmoid])

    def train(self, x_trn, y_trn, hyperparams):
        learning_rate = hyperparams["learning_rate"] or 0.08
        loop_size = hyperparams["loop_size"] or 2000
        each_layer_node_number = hyperparams["each_layer_node_number"] or [12, 8]
        activiation_functions = hyperparams["activiation_functions"] or [relu, relu]

        self.model = self.init_model(each_layer_node_number, activiation_functions)
        self.model.train()

        #print('beginning training')
        x_trn_tensor = self.tensor(x_trn)
        y_trn_tensor = self.tensor(y_trn.to_numpy())

        optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01) # use SGD to optimize
        criterion = torch.nn.MSELoss()

        # train model
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate) # use SGD to optimize
        for i in range(0, loop_size): # how to determine range??
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

        hidden_layer_sizes = np.arange(2, 6, 1)
        number_nodes = np.arange(3, 12, 3)
        activation_functions_set = (relu, sigmoid)
        loops = np.arange(500, 10000, 3000)
        learning_rates = np.arange(0.01, 0.11, 0.02)


        best_accuracy = 0
        best_params = {}

        for hidden_layer_size in hidden_layer_sizes:
            print("\tHIDDEN LAYER SIZE: ", hidden_layer_size)
            each_layer_node_number_combos = self.combos(number_nodes, hidden_layer_size)
            activation_function_combos = self.combos(activation_functions_set, hidden_layer_size)

            total_iters = len(each_layer_node_number_combos) * len(activation_function_combos)
            i = 0
            for each_layer_node_number in each_layer_node_number_combos:
                for activation_functions in activation_function_combos:
                    #print(i, total_iters)
                    i+=1
                    for loop_size in loops:
                        for learning_rate in learning_rates:
                            temp_params = {
                                "loop_size": loop_size,
                                "learning_rate": learning_rate,
                                "each_layer_node_number": each_layer_node_number,
                                "activiation_functions": activation_functions
                            }
                            self.train(X, Y, temp_params)
                            accuracy = self.accuracy(X, Y)
                            # only record if better accuracy found
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_params = temp_params
                                print('\tAccuracy:', accuracy)
                                print('\tParams:', best_params)

        return best_params

    # recursively generate all combinations
    def combos(self, els, l):
        comb = []
        if l == 1:
            for el in els:
                comb.append([el])
            return comb

        next_combs = self.combos(els, l - 1)
        for el in els:
            for next_comb in next_combs:
                copy = next_comb.copy()
                copy.append(el)
                #print(l, copy)
                comb.append(copy)

        return comb

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