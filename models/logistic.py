import scipy.io as spio
import numpy as np
import math
import matplotlib.pyplot as plt


def gradient_descent(W, data, classes):
    size = len(data)
    w = W
    j = 0.001
    reached = False

    while not reached:
        dw = derivative(w, data, classes, size)
        new_w = w - j * dw
        reached = small_array(w - new_w, j)
        print(w)
        w = new_w

    return w


def small_array(T, threshold):
    for t in T:
        if t > threshold:
            return False

    return True


def add_bias(T):
    result = []
    for x in T:
        result.append(np.append(x, [1]))
    return np.array(result)


def derivative(w, data, classes, size):
    total = 0
    for i in range(size):
        y = classes[i]
        x = data[i]
        l = logistic(x, w)

        total += x * (y - l)

    return -total


def logistic(x, wT):
    print(x, wT)
    denom = 1 + math.exp(-1 * wT.dot(x))
    return 1.0 / denom


def sigmoid(x, i, w):
    a = []
    for item in x:
        term = 1.0/(1 + math.exp(-1 * item * w[i]))
        a.append(term)
    return a



class Logistic:
    def __init__(self):
        self.W = None
        pass

    def train(self, x_trn, y_trn):
        X = add_bias(x_trn)
        Y = y_trn
        w = np.ones(len(X[0]))
        W = gradient_descent(w, X, Y)

        self.W = W
        print(W)
        pass

    def test(self, x_tst, y_tst):
        pass

    def classification_error(self, data, classes, size):
        count = 0
        for i in range(size):
            y = classes[i]
            x = data[i]
            pred = 0
            if logistic(x, self.W) > 0.5:
                pred = 1

            # print(logistic(x, W), pred)

            if pred == y[0]:
                count += 1

        print("Accuracy: ", count, "/", size, " = ", count / (size * 1.0))
        return count

    def plot(self, data, classes, size):
        tem = np.arange(-4., 4., 0.2)
        sig = sigmoid(tem, 0, self.W)

        for i in range(size):
            y = classes[i]
            x = data[i]
            if y[0] == 1:
                plt.scatter(x[0], y[0], s=10, color='red')
            else:
                plt.scatter(x[0], y[0], s=10, color='blue')

        plt.ylabel('y')
        plt.xlabel('X')
        plt.plot(tem, sig)
        plt.show()