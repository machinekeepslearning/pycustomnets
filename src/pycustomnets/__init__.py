import copy

import numpy as np
import math
import random

MIN = -1
MAX = 1

#sigmoid works

def SOFTMAX(vec):

    x = np.exp(vec)

    return x/np.sum(x)

def D_SOFTMAX(vec):
    return SOFTMAX(vec)*(1 - SOFTMAX(vec))

def d_mega_min(x):
    return math.cos(x) / x + math.sin(x) * math.log(x, math.e) * -1


def SIGMOID(vec):
    return 1.0 / (1.0 + np.exp(-vec))


# First Derivative of Sigmoid
def D_SIGMOID(vec):
    return SIGMOID(vec) * (1 - SIGMOID(vec))


def RELU(vec):
    return np.fmax(0, vec)



# First derivative of Relu
def D_RELU(vec):
    d = []
    for x in vec:
        if x > 0:
            d.append(1.0)
        else:
            d.append(0.0)

    return np.array(d)


def LRELU(vec):
    return np.fmax(0.1 * vec, vec)


# First derivative of Leaky Relu
def D_LRELU(vec):
    d = []
    for x in vec:
        if x > 0:
            d.append(1.0)
        else:
            d.append(0.1)

    return np.array(d)


funcDict = {
    "relu": RELU,
    "lrelu": LRELU,
    "sigmoid": SIGMOID,
    "softmax": SOFTMAX,
}

d_funcDict = {
    "relu": D_RELU,
    "lrelu": D_LRELU,
    "sigmoid": D_SIGMOID,
    "softmax": D_SOFTMAX,
}

# Currently, Error Func is Mean Squared Error, this function works
def errorFunc(y_predicted_vec, y_actual_vec):
    Error_sum = 0.0

    if len(y_predicted_vec) != len(y_actual_vec):
        print("Unequal vectors")
        raise ValueError

    for out in range(len(y_predicted_vec)):
        Error_sum += math.pow((y_predicted_vec[out] - y_actual_vec[out]), 2)
    Error_sum /= float(len(y_predicted_vec))

    return Error_sum


def d_errorFunc(y_predicted, y_actual, length):
    return 2.0 * (y_predicted - y_actual) / length


# Single Layer of weights with x rows and y columns

def getRandWeights(x, y):
    return np.random.uniform(MIN, MAX, (x, y))


def getRandBias(x):
    bias = []
    for i in range(x):
        bias.append(random.uniform(MIN, MAX))

    return bias


def zeroList(list):
    for i in range(len(list)):
        list[i] *= 0


# Neural Network Model
class ModelStandard:
    def __init__(self):
        self.layers = [[0]]
        self.inactive = [[0]]
        self.inputVec = None
        self.weights = []
        self.d_weight_vec = []
        self.bias = []
        self.d_bias_vec = []
        self.func_vec = []
        self.d_func_vec = []

        # ADAM variables
        self.mw = []
        self.vw = []
        self.mb = None
        self.vb = []
        self.mw_hat = []
        self.vw_hat = []
        self.mb_hat = []
        self.vb_hat = []
        self.t = 0
        self.a_sgd = 0.001
        self.a_adam = 0.001
        self.B_1 = 0.9
        self.B_2 = 0.999
        self.e = math.pow(10, -8)

    def addLayer(self, x, func):
        self.inactive.append([x])
        self.layers.append([x])
        self.bias.append(getRandBias(x))
        self.d_bias_vec.append([0])

        '''match func:
            case "relu":
                self.func_vec.append(np.vectorize(RELU))
                self.d_func_vec.append(np.vectorize(d_RELU))
            elif func == "lrelu":
                self.func_vec.append(np.vectorize(LRELU))
                self.d_func_vec.append(np.vectorize(d_LRELU))
            elif func == "sigmoid":
                self.func_vec.append(np.vectorize(sigmoid))
                self.d_func_vec.append(np.vectorize(d_sigmoid))'''
        self.func_vec.append(funcDict.get(func))
        self.d_func_vec.append(d_funcDict.get(func))

        #print(self.bias)

    def initialize(self):
        for layer in range(len(self.layers) - 1):
            if len(self.weights) == 0:
                self.weights.append(getRandWeights(self.layers[layer + 1][0], len(self.inputVec)))
                self.d_weight_vec.append([1])
            else:
                self.weights.append(getRandWeights(self.layers[layer + 1][0], len(self.weights[-1])))
                self.d_weight_vec.append([1])

        self.mw = copy.deepcopy(self.weights)
        self.vw = copy.deepcopy(self.weights)
        self.mb = copy.deepcopy(self.bias)
        self.vb = copy.deepcopy(self.bias)
        self.mw_hat = copy.deepcopy(self.weights)
        self.vw_hat = copy.deepcopy(self.weights)
        self.mb_hat = copy.deepcopy(self.weights)
        self.vb_hat = copy.deepcopy(self.weights)

        for i in range(len(self.mw)):
            self.mw[i] = np.array(self.mw[i])
            self.vw[i] = np.array(self.vw[i])
            self.mb[i] = np.array(self.mb[i])
            self.vb[i] = np.array(self.vb[i])

        for i in range(len(self.mw)):
            self.mw[i] *= 0.0
            self.vw[i] *= 0.0
            self.mb[i] *= 0.0
            self.vb[i] *= 0.0

    def setInput(self, inputv):
        self.inputVec = inputv
        self.layers[0] = inputv
        self.inactive[0] = inputv

    def feedforward(self):
        for i in range(len(self.weights)):
            #print(self.bias)

            self.inactive[i + 1] = np.dot(self.weights[i], self.layers[i]) + self.bias[i]
            self.layers[i + 1] = self.func_vec[i](self.inactive[i + 1])

        #print(self.weights)
        #print(self.bias)
        #print(self.layers)

    # Current Optimization Method: First derivative backpropagation (I think this is sgd?)

    #The problem is in the optimization

    def computeGradient(self, expected):

        self.feedforward()

        d_Hidden = np.vectorize(d_errorFunc)(self.layers[-1], expected, len(expected))

        for layer_index in range(len(self.layers) - 1, 0, -1):

            d_activated_sum_vec = self.d_func_vec[layer_index-1](self.inactive[layer_index])

            self.d_weight_vec[layer_index - 1] = d_activated_sum_vec * d_Hidden
            self.d_bias_vec[layer_index - 1] = d_activated_sum_vec * d_Hidden

            d_eachW = np.array([self.layers[layer_index - 1]] * len(self.layers[layer_index])).T

            d_Hidden = np.dot(self.d_weight_vec[layer_index - 1], self.weights[layer_index - 1])

            self.d_weight_vec[layer_index - 1] = (self.d_weight_vec[layer_index - 1] * d_eachW).T

    def updateSGD(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.a_sgd * self.d_weight_vec[i]
            self.bias[i] -= self.a_sgd * self.d_bias_vec[i]

    def updateADAM(self):
        self.t += 1

        #Update weights and biases
        for k in range(len(self.mw)):
            #print(self.mw[k])

            self.mw[k] = self.d_weight_vec[k] * (1.0 - self.B_1) + self.mw[k] * self.B_1
            self.mb[k] = self.d_bias_vec[k] * (1.0 - self.B_1) + self.mb[k] * self.B_1

            #print(self.m[k])

            self.vw[k] = np.square(self.d_weight_vec[k]) * (1 - self.B_2) + self.vw[k] * self.B_2
            self.mw_hat = self.mw[k] / (1 - math.pow(self.B_1, self.t))
            self.vw_hat = self.vw[k] / (1 - math.pow(self.B_2, self.t))

            self.vb[k] = np.square(self.d_bias_vec[k]) * (1 - self.B_2) + self.vb[k] * self.B_2
            self.mb_hat = self.mb[k] / (1 - math.pow(self.B_1, self.t))
            self.vb_hat = self.vb[k] / (1 - math.pow(self.B_2, self.t))

            self.weights[k] -= self.a_adam * np.divide(self.mw_hat, np.sqrt(self.vw_hat) + self.e)
            self.bias[k] -= self.a_adam * np.divide(self.mb_hat, np.sqrt(self.vb_hat) + self.e)

    def optimize(self, expected, o_type):

        self.computeGradient(expected)

        match o_type:
            case "sgd":
                #print("optimizing using sgd")
                self.updateSGD()
            case "adam":
                self.updateADAM()
            case "adamax":
                pass

    def out(self):
        self.feedforward()
        print("Predicted: " + str(self.layers[-1]))

    def debug(self):
        print("Weights: " + str(self.weights))
        print("Layers: " + str(self.layers))
        print("derivatives: " + str(self.d_weight_vec))

    def error(self, expected):
        self.feedforward()
        return errorFunc(self.layers[-1], expected)


class ModelRL:
    def __init__(self):
        pass

'''def train_func(x):
    return 2 * x'''


'''if __name__ == "__main__":
    train_in = []
    train_out = []

    for i in range(20):
        train_in.append([i*0.05])
        train_out.append([train_func(i*0.05)])

    NN = ModelStandard()
    NN.setInput(train_in[0])
    NN.addLayer(1, "softmax")
    NN.initialize()

    for i in range(1000):
        for i in range(20):
            NN.setInput(train_in[i])
            NN.optimize(train_out[i], "sgd")

    NN.setInput([0.03])
    NN.out()'''
