import copy

import numpy as np
import math
import random

MIN = -0.01
MAX = 0.01

#Activation Functions and their derivatives

def SOFTMAX(vec):

    x = np.exp(vec - np.mean(vec))

    return x/np.sum(x)

def D_SOFTMAX(vec):
    s = SOFTMAX(vec)
    return s * (1 - s)

def d_mega_min(x):
    return math.cos(x) / x + math.sin(x) * math.log(x, math.e) * -1


def SIGMOID(vec):
    return 1.0 / (1.0 + np.exp(np.negative(vec)))


# First Derivative of Sigmoid
def D_SIGMOID(vec):
    s = SIGMOID(vec)
    return s * (1.0 - s)


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

#Loss Functions and their derivatives

def MEAN_SQUARED_ERROR(predicted_vec, true_vec):
    return np.sum(np.square(predicted_vec-true_vec))/len(true_vec)

def D_MEAN_SQUARED_ERROR(predicted_vec, true_vec):
    return 2.0*(predicted_vec-true_vec)/len(true_vec)

def CROSS_ENTROPY(predicted_vec, true_vec):
    return np.negative(np.sum(true_vec * np.log(predicted_vec)))

def D_CROSS_ENTROPY(predicted_vec, true_vec):
    return np.negative(true_vec/predicted_vec)

def BINARY_CROSS_ENTROPY(predicted_vec, true_vec):
    return np.negative(np.sum(true_vec*np.log(predicted_vec)+(1.0-true_vec)*np.log(1.0-predicted_vec)))

def D_BINARY_CROSS_ENTROPY(predicted_vec, true_vec):
    return np.negative(true_vec/predicted_vec - (1.0-true_vec)/(1.0-predicted_vec))


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

error_Dict = {
    "mse": MEAN_SQUARED_ERROR,
    "cross_entropy": CROSS_ENTROPY,
    "b_cross_entropy": BINARY_CROSS_ENTROPY,
}

d_error_Dict = {
    "mse": D_MEAN_SQUARED_ERROR,
    "cross_entropy": D_CROSS_ENTROPY,
    "b_cross_entropy": D_BINARY_CROSS_ENTROPY,
}


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

def vecify(x, length):
    vec = []
    for i in range(length):
        if i == x:
            vec.append(1)
        else:
            vec.append(0)

    return np.array(vec)

# Neural Network Model
class ModelStandard:
    def __init__(self):
        self.d_errorFunc = None
        self.errorFunc = None
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
        self.a_adam = 0.00001
        self.B_1 = 0.9
        self.B_2 = 0.999
        self.e = math.pow(10, -8)

    def addLayer(self, x, func):
        self.inactive.append([x])
        self.layers.append([x])
        self.bias.append(getRandBias(x))
        self.d_bias_vec.append([0])


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

    def setError(self, error):
        self.errorFunc = error_Dict.get(error)
        self.d_errorFunc = d_error_Dict.get(error)

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

    def computeGradient(self, true_l):

        self.feedforward()

        if len(true_l) != len(self.layers[-1]) and len(true_l) == 1:
            expected = vecify(true_l[0], len(self.layers[-1]))
        elif len(true_l) == len(self.layers[-1]):
            expected = true_l
        else:
            print("Incomparable Dimensions")
            exit(-1)

        d_Hidden = self.d_errorFunc(self.layers[-1], expected)

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

    def error(self, true_l):
        self.feedforward()

        if len(true_l) != len(self.layers[-1]) and len(true_l) == 1:
            expected = vecify(true_l[0], len(self.layers[-1]))
        elif len(true_l) == len(self.layers[-1]):
            expected = true_l
        else:
            print("Incomparable Dimensions")
            exit(-1)

        return self.errorFunc(self.layers[-1], expected)


class ModelRL:
    def __init__(self):
        pass
