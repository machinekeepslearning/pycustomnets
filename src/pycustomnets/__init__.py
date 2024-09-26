import copy

import numpy as np
import math
import random

MIN = 0
MAX = 1


#sigmoid works

def softmax(vec):
    activated = []
    sum = 0
    for j in range(len(vec)):
        sum += math.exp(vec[j])

    for i in range(len(vec)):
        activated.append(math.exp(vec[i]) / sum)

    return activated


def d_mega_min(x):
    return math.cos(x) / x + math.sin(x) * math.log(x, math.e) * -1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# First Derivative of Sigmoid
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def RELU(x):
    return float(max(0, x))


# First derivative of Relu
def d_RELU(x):
    if x > 0:
        return float(1)
    else:
        return float(0)


def LRELU(x):
    return float(max(0.1 * x, x))


# First derivative of Leaky Relu
def d_LRELU(x):
    if x > 0:
        return float(1)
    else:
        return float(0.1)


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
        self.a_sgd = 0.0001
        self.a_adam = 0.001
        self.B_1 = 0.9
        self.B_2 = 0.999
        self.e = math.pow(10, -8)

    def addLayer(self, x, func):
        self.inactive.append([x])
        self.layers.append([x])
        self.bias.append(getRandBias(x))
        self.d_bias_vec.append([0])

        if func == "relu":
            self.func_vec.append(np.vectorize(RELU))
            self.d_func_vec.append(np.vectorize(d_RELU))
        elif func == "lrelu":
            self.func_vec.append(np.vectorize(LRELU))
            self.d_func_vec.append(np.vectorize(d_LRELU))
        elif func == "sigmoid":
            self.func_vec.append(np.vectorize(sigmoid))
            self.d_func_vec.append(np.vectorize(d_sigmoid))
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
