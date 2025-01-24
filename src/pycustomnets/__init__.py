import copy
from idlelib.pyparse import trans
from logging import fatal

import numpy
import math
import random

import numpy as np

MIN = 0
MAX = 0.01


# Activation Functions and their derivatives

def SOFTMAX(vec):
    x = numpy.exp(vec - numpy.mean(vec))

    return x / numpy.sum(x)


def D_SOFTMAX(vec):
    s = SOFTMAX(vec)
    return s * (1 - s)


def d_mega_min(x):
    return math.cos(x) / x + math.sin(x) * math.log(x, math.e) * -1


def SIGMOID(vec):
    return 1.0 / (1.0 + numpy.exp(numpy.negative(vec)))


# First Derivative of Sigmoid
def D_SIGMOID(vec):
    s = SIGMOID(vec)
    return s * (1.0 - s)


def RELU(vec):
    return numpy.fmax(0, vec)


# First derivative of Relu
def D_RELU(vec):
    return numpy.fmin(numpy.fmax(vec, 0), 1)


def LRELU(vec):
    return numpy.fmax(0.1 * vec, vec)


# First derivative of Leaky Relu
def D_LRELU(vec):
    d = []
    for x in vec:
        if x > 0:
            d.append(1.0)
        else:
            d.append(0.1)

    return numpy.array(d)


# Loss Functions and their derivatives

def MEAN_SQUARED_ERROR(predicted_vec, true_vec):
    return numpy.sum(numpy.square(predicted_vec - true_vec)) / len(true_vec)


def D_MEAN_SQUARED_ERROR(predicted_vec, true_vec):
    return 2.0 * (predicted_vec - true_vec) / len(true_vec)


def CROSS_ENTROPY(predicted_vec, true_vec):
    return numpy.negative(numpy.sum(true_vec * numpy.log(predicted_vec)))


def D_CROSS_ENTROPY(predicted_vec, true_vec):
    return numpy.negative(true_vec / predicted_vec)


def BINARY_CROSS_ENTROPY(predicted_vec, true_vec):
    print(predicted_vec)
    print(true_vec)
    return numpy.negative(numpy.sum(true_vec * numpy.log(predicted_vec) + (1.0 - true_vec) * numpy.log(1.0 - predicted_vec)))


def D_BINARY_CROSS_ENTROPY(predicted_vec, true_vec):
    return numpy.negative(true_vec / predicted_vec - (1.0 - true_vec) / (1.0 - predicted_vec))

#Shortcuts

def BCE_SOFTMAX(inactive, true):
    return SOFTMAX(inactive) - true

def CE_SIGMOID(inactive, true):
    return numpy.negative(true * (1 - SIGMOID(inactive)))

#Dictionaries

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


# Misc Functions

#def getRandWeights(x, y):
#    return np.random.uniform(MIN, MAX, (x, y))


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

    return numpy.array(vec)


def unitTensor(x):
    return x / numpy.max(x)



# Neural Network Model
class ModelStandard:
    def __init__(self, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.iterations = 0
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
        self.meanw = None
        self.meanb = None
        self.eFuncName = None
        self.outFuncName = None

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

        #For Convolutional NNs
        self.wrt_cLayer = None

        # Misc
        self.quickbce = None
        self.quickce = None
        self.updaters = {
            "sgd": self.updateSGD,
            "adam": self.updateADAM
        }

    def addLayer(self, x, func):
        self.inactive.append([x])
        self.layers.append([x])
        self.bias.append(numpy.array(getRandBias(x)))
        self.d_bias_vec.append([0])

        self.func_vec.append(funcDict.get(func))
        self.d_func_vec.append(d_funcDict.get(func))

        self.outFuncName = func

    def initialize(self):
        for layer in range(len(self.layers) - 1):
            if len(self.weights) == 0:
                self.weights.append(numpy.random.uniform(MIN, MAX, (self.layers[layer + 1][0], len(self.inputVec))))
                self.d_weight_vec.append([1])
            else:
                self.weights.append(numpy.random.uniform(MIN, MAX, (self.layers[layer + 1][0], len(self.weights[-1]))))
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
            self.mw[i] = numpy.array(self.mw[i])
            self.vw[i] = numpy.array(self.vw[i])
            self.mb[i] = numpy.array(self.mb[i])
            self.vb[i] = numpy.array(self.vb[i])

        for i in range(len(self.mw)):
            self.mw[i] *= 0.0
            self.vw[i] *= 0.0
            self.mb[i] *= 0.0
            self.vb[i] *= 0.0

        # shortcuts

        self.quickbce = self.eFuncName == "b_cross_entropy" and self.outFuncName == "softmax"
        self.quickce = self.eFuncName == "cross_entropy" and self.outFuncName == "sigmoid"

    def setInput(self, arr_in, norm):

        if numpy.array(arr_in).shape[0] != 1:
            inputv = numpy.array(arr_in).flatten()
        else:
            inputv = numpy.array(arr_in)

        if norm:
            inputv = unitTensor(inputv)

        self.inputVec = inputv
        self.layers[0] = copy.deepcopy(inputv)
        self.inactive[0] = copy.deepcopy(inputv)

    def setError(self, error):
        self.errorFunc = error_Dict.get(error)
        self.d_errorFunc = d_error_Dict.get(error)
        self.eFuncName = error

    def feedforward(self):
        for i in range(len(self.weights)):

            self.inactive[i + 1] = numpy.dot(self.weights[i], self.layers[i]) + self.bias[i]
            self.layers[i + 1] = self.func_vec[i](self.inactive[i + 1])

    # Optimization start

    def computeGradient(self, true_l):
        self.feedforward()

        if len(true_l) != len(self.layers[-1]) and len(true_l) == 1:
            expected = vecify(true_l[0], len(self.layers[-1]))
        elif len(true_l) == len(self.layers[-1]):
            expected = true_l
        else:
            print(f"Incomparable Dimensions {len(true_l)} {len(self.layers[-1])}")
            exit(-1)

        d_Hidden = self.d_errorFunc(self.layers[-1], expected)

        for layer_index in range(len(self.layers) - 1, 0, -1):

            d_activated_sum_vec = self.d_func_vec[layer_index - 1](self.inactive[layer_index])

            if self.quickbce and layer_index == len(self.layers) - 1:
                self.d_weight_vec[layer_index - 1] = BCE_SOFTMAX(self.inactive[layer_index], expected)
                self.d_bias_vec[layer_index - 1] = BCE_SOFTMAX(self.inactive[layer_index], expected)
            elif self.quickce and layer_index == len(self.layers) - 1:
                self.d_weight_vec[layer_index - 1] = CE_SIGMOID(self.inactive[layer_index], expected)
                self.d_bias_vec[layer_index - 1] = CE_SIGMOID(self.inactive[layer_index], expected)
            else:
                self.d_weight_vec[layer_index - 1] = d_activated_sum_vec * d_Hidden
                self.d_bias_vec[layer_index - 1] = d_activated_sum_vec * d_Hidden

            d_eachW = numpy.array([self.layers[layer_index - 1]] * len(self.layers[layer_index])).T

            d_Hidden = numpy.dot(self.d_weight_vec[layer_index - 1], self.weights[layer_index - 1])

            self.d_weight_vec[layer_index - 1] = (self.d_weight_vec[layer_index - 1] * d_eachW).T

        #For use with CNN gradient
        self.wrt_cLayer = d_Hidden

    def updateSGD(self, d_weight_vec, d_bias_vec):
        for i in range(len(self.weights)):
            self.weights[i] -= self.a_sgd * d_weight_vec[i]
            self.bias[i] -= self.a_sgd * d_bias_vec[i]

    def updateADAM(self, d_weight_vec, d_bias_vec):
        self.t += 1

        # Update weights and biases
        for k in range(len(self.mw)):
            # print(self.mw[k])

            self.mw[k] = d_weight_vec[k] * (1.0 - self.B_1) + self.mw[k] * self.B_1
            self.mb[k] = d_bias_vec[k] * (1.0 - self.B_1) + self.mb[k] * self.B_1

            # print(self.m[k])

            self.vw[k] = numpy.square(d_weight_vec[k]) * (1 - self.B_2) + self.vw[k] * self.B_2
            self.mw_hat = self.mw[k] / (1 - math.pow(self.B_1, self.t))
            self.vw_hat = self.vw[k] / (1 - math.pow(self.B_2, self.t))

            self.vb[k] = numpy.square(d_bias_vec[k]) * (1 - self.B_2) + self.vb[k] * self.B_2
            self.mb_hat = self.mb[k] / (1 - math.pow(self.B_1, self.t))
            self.vb_hat = self.vb[k] / (1 - math.pow(self.B_2, self.t))

            self.weights[k] -= self.a_adam * numpy.divide(self.mw_hat, numpy.sqrt(self.vw_hat) + self.e)
            self.bias[k] -= self.a_adam * numpy.divide(self.mb_hat, numpy.sqrt(self.vb_hat) + self.e)

    def optimize(self, expected, o_type):
        self.computeGradient(expected)
        self.updaters.get(o_type)(self.d_weight_vec, self.d_bias_vec)

    def train(self, training_in, training_out, o_type, norm):
        print(len(training_out))
        print(self.batch_size)
        self.iterations = int(len(training_out) / self.batch_size)
        print(self.iterations)

        self.meanw = copy.deepcopy(self.weights)
        self.meanb = copy.deepcopy(self.bias)

        for _epoch in range(self.epochs):
            print("Epoch: " + str(_epoch))
            for i in range(self.iterations):
                print("Iteration: " + str(i))
                for a in range(len(self.meanw)):
                    self.meanw[a] *= 0
                    self.meanb[a] *= 0
                for j in range(self.batch_size):
                    for k in range(len(self.meanw)):
                        self.setInput(training_in[j], norm)
                        self.computeGradient([training_out[j]])
                        self.meanw[k] += numpy.array(self.d_weight_vec[k])
                        self.meanb[k] += numpy.array(self.d_bias_vec[k])
                for b in range(len(self.meanw)):
                    self.meanw[b] /= self.batch_size
                    self.meanb[b] /= self.batch_size

                self.updaters.get(o_type)(self.meanw, self.meanb)


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



class ConvModel(ModelStandard):
    def __init__(self, batch_size, epochs):
        super().__init__(batch_size, epochs)
        self.image_tensor = None #where the image is stored as a tensor
        self.sum_tuple = [] # The axes across which to sum
        self.patch_sizes = [] # array of patch sizes in (x,y) form
        self.patch_area = [] # array of patch areas with each element = x * y
        self.strides = [] #array of strides for each convolution
        self.patch_maps = [] #array of patch maps where a patch map is defined as a matrix containing patches that a kernel will affect
        self.convLayers = [[0]] # array of each layer to be filtered, including input
        self.inactiveConvLayers = [] # convLayers that have not been passed into an activation function
        self.kernels = [] # (x, y) size filters that act as weights
        self.wrt_kernels = [] # gradient wrt the kernel at each position
        self.meank = None

        self.updatersC = {
            "sgd": self.updateSGDC,
            "adam": self.updateADAMC
        }

        self.mk = None
        self.vk = None
        self.tk = 0
    def setImage(self, img_tensor):
        arr_in = None
        match len(img_tensor.shape):
            case 3:
                arr_in = numpy.swapaxes(numpy.swapaxes(img_tensor, 0, 2), 1, 2)
            case 2:
                arr_in = numpy.array([img_tensor]).reshape(28, 28, 1)

        self.image_tensor = arr_in
        self.convLayers[0] = arr_in

    def _getPatchMap(self, img, patch_x, patch_y):
        x_slide = img.shape[2] - patch_x + 1
        y_slide = img.shape[1] - patch_y + 1
        tensor_four = []

        for f in range(img.shape[0]):
            p_tensor = []
            for y in range(y_slide):
                p_mat = []
                for x in range(x_slide):
                    p_mat.append(img[f][y:y + patch_y, x:x + patch_x].flatten())
                p_tensor.append(p_mat)
            tensor_four.append(p_tensor)

        return numpy.array(tensor_four)


    def addLayerC(self, patch_length, stride, num_kernels):
        self.kernels.append(numpy.random.uniform(MIN, MAX, (num_kernels, patch_length ** 2, 1)))
        self.wrt_kernels.append([0])
        self.convLayers.append([0])
        self.inactiveConvLayers.append([0])
        self.patch_sizes.append(numpy.array([patch_length, patch_length, 0]))
        self.strides.append(stride)

        self.patch_area.append(patch_length**2)

    def initializeC(self):
        self.forwardC(False)
        self.initialize()

        self.mk = copy.deepcopy(self.kernels)
        self.vk = copy.deepcopy(self.kernels)

        for i in range(len(self.mw)):
            self.mk[i] = numpy.array(self.mk[i])
            self.vk[i] = numpy.array(self.vk[i])

        for i in range(len(self.mw)):
            self.mk[i] *= 0.0
            self.vk[i] *= 0.0


    def forwardC(self, norm):
        for i in range(len(self.convLayers)-1):
            self.patch_maps.append(self._getPatchMap(self.convLayers[i], self.patch_sizes[i][0], self.patch_sizes[i][1]))
            self.inactiveConvLayers[i] = numpy.dot(self.patch_maps[i], self.kernels[i])
            '''self.inactiveConvLayers[i] = np.swapaxes(np.swapaxes(np.dot(self.patch_maps[i], self.kernels[i]), 1, 3), 2, 3)
            dims = self.inactiveConvLayers[i].shape
            self.inactiveConvLayers[i] = np.reshape(self.inactiveConvLayers[i], (dims[0] * dims[1], dims[2], dims[3]))'''
            self.convLayers[i+1] = LRELU(self.inactiveConvLayers[i])
        self.setInput(self.convLayers[-1], norm)

    def forwardAll(self):
        self.forwardC(False)
        self.feedforward()

    def computeGradientC(self, true_l):
        self.forwardAll()
        self.computeGradient(true_l)
        for i in range(len(self.inactiveConvLayers) - 1, -1, -1):
            last_deriv = self.wrt_cLayer.reshape(self.inactiveConvLayers[i].shape) * D_RELU(self.inactiveConvLayers[i])
            this_deriv = numpy.moveaxis(numpy.swapaxes(last_deriv, 3, 4), 4, 0)
            this_deriv = numpy.multiply(this_deriv, self.patch_maps[i])
            self.wrt_kernels[i] = numpy.reshape(numpy.sum(this_deriv, (1, 2, 3)), self.kernels[i].shape)
            print(self.wrt_kernels[i].shape)

            #calculates derivative wrt the layer the current kernel activates to be used for the next kernel
            temp = []
            self.wrt_cLayer = []
            kernel_side = int(math.sqrt(self.kernels[i].shape[1]))
            padded_deriv = numpy.delete(numpy.delete(numpy.delete(numpy.pad(last_deriv, kernel_side - 1, "constant"), [0, last_deriv.shape[0]+1], 0), [0, last_deriv.shape[3]+1], 3), [0, last_deriv.shape[4]+1], 4)

            for p in range(self.kernels[i].shape[0]):
                temp.append(self._getPatchMap(padded_deriv[:, :, :, p, 0], kernel_side, kernel_side))
                self.wrt_cLayer.append(numpy.dot(temp[p], numpy.flip(self.kernels[i][p], 1)))
            self.wrt_cLayer = numpy.sum(self.wrt_cLayer, 0)
            self.wrt_cLayer = numpy.reshape(self.wrt_cLayer, self.wrt_cLayer.shape[0:len(self.wrt_cLayer.shape)-1])

    def updateADAMC(self, wrt_kernels):
        self.tk += 1

        for k in range(len(self.mk)):
            self.mk[k] = wrt_kernels[k] * (1.0 - self.B_1) + self.mk[k] * self.B_1
            self.vk[k] = numpy.square(wrt_kernels[k]) * (1 - self.B_2) + self.vk[k] * self.B_2

            mk_hat = self.mk[k] / (1.0 - math.pow(self.B_1, self.t))
            vk_hat = self.vk[k] / (1.0 - math.pow(self.B_2, self.t))

            self.kernels[k] -= self.a_adam * numpy.divide(mk_hat, numpy.sqrt(vk_hat) + self.e)

    def updateSGDC(self, wrt_kernels):
        for i in range(len(self.kernels)):
            self.kernels[i] -= self.a_sgd * wrt_kernels[i]

    def updateC(self, expected, o_type):
        self.computeGradientC(expected)
        self.updaters.get(o_type)(self.d_weight_vec, self.d_bias_vec)
        self.updatersC.get(o_type)(self.wrt_kernels)

    def train(self, training_in, training_out, o_type, norm):

        avg_error = 0

        print(len(training_out))
        print(self.batch_size)
        self.iterations = int(len(training_out) / self.batch_size)
        print(self.iterations)

        self.meanw = copy.deepcopy(self.weights)
        self.meanb = copy.deepcopy(self.bias)
        self.meank = copy.deepcopy(self.kernels)

        for _epoch in range(self.epochs):
            print("Epoch: " + str(_epoch))
            for i in range(self.iterations):
                print("Iteration: " + str(i))
                print(avg_error)
                for a in range(len(self.meanw)):
                    self.meanw[a] *= 0
                    self.meanb[a] *= 0
                for a in range(len(self.meank)):
                    self.meank[a] *= 0
                for j in range(self.batch_size):
                    self.setImage(training_in[j])
                    self.computeGradientC([training_out[j]])
                    avg_error += self.error([training_out[j]])
                    for k in range(len(self.meanw)):
                        self.meanw[k] += numpy.array(self.d_weight_vec[k])
                        self.meanb[k] += numpy.array(self.d_bias_vec[k])
                    for k in range(len(self.meank)):
                        self.meank[k] += numpy.array(self.wrt_kernels[k].reshape(self.kernels[k].shape))
                for b in range(len(self.meanw)):
                    self.meanw[b] /= self.batch_size
                    self.meanb[b] /= self.batch_size
                for b in range(len(self.meank)):
                    self.meank[b] /= self.batch_size
                avg_error /= self.batch_size

                self.updaters.get(o_type)(self.meanw, self.meanb)
                self.updatersC.get(o_type)(self.meank)


