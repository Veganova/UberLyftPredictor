from sklearn.metrics import accuracy_score
import torch 
from torch.autograd import Variable

# define some possible activation functions
relu = torch.nn.ReLU()
sigmoid = torch.nn.Sigmoid()

# three layer neural net (2 hidden layers)
class FeedForward3(torch.nn.Module):
    # layer sizes = list of length 2, activation_functions = list of length 3
    def __init__(self, hidden_layer_sizes, activation_functions): 
        super(FeedForward3, self).__init__()
        self.input_layer_size = 6
        self.hidden_layer_sizes = hidden_layer_sizes
        self.af1 = activation_functions[0]
        self.af2 = activation_functions[1]
        self.af3 = activation_functions[2]
        self.weights1 = torch.nn.Linear(self.input_layer_size, self.hidden_layer_sizes[0])
        self.weights2 = torch.nn.Linear(self.hidden_layer_sizes[0], self.hidden_layer_sizes[1])
        self.weights3 = torch.nn.Linear(self.hidden_layer_sizes[1], 1)

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
        self.model = FeedForward3([10, 8], [sigmoid, relu, sigmoid])
        self.name = "Neural Network"

    def train(self, x_trn, y_trn, hyperparams):
        print('beginning training')
        x_trn_tensor = self.tensor(x_trn)
        y_trn_tensor = self.tensor(y_trn.to_numpy())

        optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01) # use SGD to optimize
        criterion = torch.nn.MSELoss()

        # train model
        self.model.train()
        for i in range(0, 100): # how to determine range?
            optimizer.zero_grad()
            # feed forward
            predicted = self.model(x_trn_tensor)
            # compute loss
            loss = criterion(predicted, y_trn_tensor)
            #print('Iteration {}: train loss: {}'.format(i, loss.item()))
            # propogate backward
            loss.backward()
            optimizer.step() 

    def tune_hyperparameters(self, x_trn, y_trn):
        return {} # do nothing

    def predict(self, x_vals):
        self.model.eval()
        x_tensor = self.tensor(x_vals)
        #return np.rint(self.model(x_tensor).detach().numpy()).astype(int)
        prediction = self.model(x_tensor).int().numpy()
        return prediction

    def accuracy(self, x_vals, y_vals):
        predicted = self.predict(x_vals)
        return accuracy_score(y_vals, predicted)
        

    # convert data to the tensor type used by torch
    def tensor(self, data):
        return torch.tensor(data).float()