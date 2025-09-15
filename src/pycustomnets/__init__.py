import copy

import numpy
import math
import random

import numpy as np

MIN = 0
MAX = 0.1
small_num = 1e-8

# Activation Functions and their derivatives

def SOFTMAX(vec):
    x = numpy.exp(vec - numpy.max(vec))

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
    return numpy.greater(vec, 0)


def LRELU(vec):
    return numpy.fmax(0.1 * vec, vec)


# First derivative of Leaky Relu
def D_LRELU(vec):
    a = numpy.greater(0, vec) * 0.1
    b = numpy.greater(vec, 0) * 1
    return a+b


# Loss Functions and their derivatives

def MEAN_SQUARED_ERROR(predicted_vec, true_vec):
    return numpy.sum(numpy.square(predicted_vec - true_vec)) / len(true_vec)


def D_MEAN_SQUARED_ERROR(predicted_vec, true_vec):
    return 2.0 * (predicted_vec - true_vec) / len(true_vec)


def CROSS_ENTROPY(predicted_vec, true_vec):
    return numpy.negative(numpy.sum(numpy.multiply(true_vec, numpy.log(predicted_vec + small_num))))


def D_CROSS_ENTROPY(predicted_vec, true_vec):
    return numpy.negative(true_vec / predicted_vec)


def BINARY_CROSS_ENTROPY(predicted_vec, true_vec):
    return numpy.negative(numpy.sum(true_vec * numpy.log(predicted_vec) + (1.0 - true_vec) * numpy.log(1.0 - predicted_vec)))


def D_BINARY_CROSS_ENTROPY(predicted_vec, true_vec):
    return numpy.negative(true_vec / predicted_vec - (1.0 - true_vec) / (1.0 - predicted_vec))

#Shortcuts

def FAST_BCE(predicted, true):
    return -true * (1 - predicted)

def FAST_CE(predicted, true):
    return predicted - true

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
    "ce": CROSS_ENTROPY,
    "bce": BINARY_CROSS_ENTROPY,
}

d_error_Dict = {
    "mse": D_MEAN_SQUARED_ERROR,
    "ce": D_CROSS_ENTROPY,
    "bce": D_BINARY_CROSS_ENTROPY,
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
    def __init__(self, batch_size, epochs, norm, dropout):
        self.training = True
        self.dropout = dropout
        self.norm = norm
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
        self.t = 0
        self.a_sgd = 0.001
        self.a_adam = 0.001
        self.B_1 = 0.9
        self.B_2 = 0.999
        self.e = math.pow(10, -8)

        #For Convolutional NNs
        self.d_layer = None

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

        self.mw = [0] * len(self.weights)
        self.vw = [0] * len(self.weights)
        self.mb = [0] * len(self.bias)
        self.vb = [0] * len(self.bias)


        # shortcuts

        self.quickbce = self.eFuncName == "bce" and self.outFuncName == "softmax" or self.outFuncName == "sigmoid"

        #print(self.quickbce)

        self.quickce = self.eFuncName == "ce" and self.outFuncName == "sigmoid" or self.outFuncName == "softmax"

    def setInput(self, arr_in):
        arr_in = numpy.array(arr_in)

        if arr_in.shape[0] != 1 or len(arr_in.shape) > 2:
            inputv = arr_in.flatten()
        else:
            inputv = arr_in

        if self.norm:
            #inputv = inputv/255
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

    def computeGradient(self, training_in, true_l):
        self.training = True
        self.setInput(training_in)
        self.feedforward()


        if len(true_l) == len(self.layers[-1]):
            expected = true_l
        elif len(true_l) == 1:
            expected = vecify(true_l[0], len(self.layers[-1]))
        else:
            print(f"Incomparable Dimensions {len(true_l)} {len(self.layers[-1])}")
            exit(-1)

        if self.quickce or self.quickbce:
            self.d_layer = 0
        else:
            self.d_layer = self.d_errorFunc(self.layers[-1], expected)

        for layer_index in range(len(self.layers) - 1, 0, -1):

            if self.quickbce and layer_index == len(self.layers) - 1:
                self.d_weight_vec[layer_index - 1] = FAST_BCE(self.layers[-1], expected)
                self.d_bias_vec[layer_index - 1] = FAST_BCE(self.layers[-1], expected)
            elif self.quickce and layer_index == len(self.layers) - 1:
                self.d_weight_vec[layer_index - 1] = FAST_CE(self.layers[-1], expected)
                self.d_bias_vec[layer_index - 1] = FAST_CE(self.layers[-1], expected)
            else:
                d_activated_sum_vec = self.d_func_vec[layer_index - 1](self.inactive[layer_index])
                self.d_weight_vec[layer_index - 1] = numpy.multiply(d_activated_sum_vec, self.d_layer)
                self.d_bias_vec[layer_index - 1] = numpy.multiply(d_activated_sum_vec, self.d_layer)

            d_eachW = numpy.array([self.layers[layer_index - 1]] * len(self.layers[layer_index])).T

            self.d_layer = numpy.dot(self.d_weight_vec[layer_index - 1], self.weights[layer_index - 1])

            self.d_weight_vec[layer_index - 1] = (self.d_weight_vec[layer_index - 1] * d_eachW).T

    def updateSGD(self, d_weight_vec, d_bias_vec):
        for i in range(len(self.weights)):
            self.weights[i] -= self.a_sgd * d_weight_vec[i]
            self.bias[i] -= self.a_sgd * d_bias_vec[i]

    def updateADAM(self, d_weight_vec, d_bias_vec):
        self.t += 1

        # Update weights and biases
        for k in range(len(self.mw)):

            self.mw[k] = d_weight_vec[k] * (1.0 - self.B_1) + self.mw[k] * self.B_1
            self.vw[k] = numpy.square(d_weight_vec[k]) * (1 - self.B_2) + self.vw[k] * self.B_2

            self.mb[k] = d_bias_vec[k] * (1.0 - self.B_1) + self.mb[k] * self.B_1
            self.vb[k] = numpy.square(d_bias_vec[k]) * (1 - self.B_2) + self.vb[k] * self.B_2

            a_t = self.a_adam * math.sqrt(1 - math.pow(self.B_2, self.t)) / (1 - math.pow(self.B_1, self.t))
            self.weights[k] -= a_t * numpy.divide(self.mw[k], numpy.sqrt(self.vw[k]) + self.e)
            self.bias[k] -= a_t * numpy.divide(self.mb[k], numpy.sqrt(self.vb[k]) + self.e)

    def update(self, expected_in, expected, o_type):
        self.computeGradient(expected_in, expected)
        self.updaters.get(o_type)(self.d_weight_vec, self.d_bias_vec)

    def train(self, training_in, training_out, o_type):
        #print(self.batch_size)
        self.iterations = int(len(training_out) / self.batch_size)
        #print(self.iterations)

        self.meanw = copy.deepcopy(self.weights)
        self.meanb = copy.deepcopy(self.bias)

        for _epoch in range(self.epochs):
            for i in range(self.iterations):
                for a in range(len(self.meanw)):
                    self.meanw[a] *= 0
                    self.meanb[a] *= 0
                for j in range(self.batch_size):
                    self.computeGradient(training_in[self.batch_size * i + j], [training_out[self.batch_size * i + j]])
                    for k in range(len(self.meanw)):
                        self.meanw[k] += self.d_weight_vec[k]
                        self.meanb[k] += self.d_bias_vec[k]
                self.updaters.get(o_type)(self.meanw, self.meanb)

    def error(self, true_in, true_l):
        self.training = True
        self.setInput(true_in)
        self.feedforward()

        if len(true_l) != len(self.layers[-1]) and len(true_l) == 1:
            expected = vecify(true_l[0], len(self.layers[-1]))
        elif len(true_l) == len(self.layers[-1]):
            expected = true_l
        else:
            print("Incomparable Dimensions")
            exit(-1)

        return self.errorFunc(self.layers[-1], expected)

    def singlePredict(self, test_in, test_out):
        error = self.error(test_in, test_out)
        print(f"Predicted {numpy.argmax(self.layers[-1])}, True label was {test_out[0]}")
        print(f"Accuracy: {numpy.array(self.layers[-1][int(test_out[0])])}")
        print(self.layers[-1])
        print(f"Error: {error}")
        return error

    def eval(self, test_in_arr, test_out_arr):
        avg_loss = 0
        avg_acc = 0

        for i in range(len(test_out_arr)):
            avg_loss += self.error(test_in_arr[i], [test_out_arr[i]])
            avg_acc += self.layers[-1][test_out_arr[i]]

        print(f"Average loss: {avg_loss/len(test_out_arr)}")
        print(f"Average accuracy: {avg_acc/len(test_out_arr)}")



class ConvModel(ModelStandard):
    def __init__(self, batch_size, epochs, norm, dropout):
        super().__init__(batch_size, epochs, norm, dropout)
        self.image_tensor = None #where the image is stored as a tensor
        self.patch_sizes = [] # array of patch sizes in (x,y) form
        self.patch_area = [] # array of patch areas with each element = x * y
        self.strides = [] #array of strides for each convolution
        self.patch_maps = [] #array of patch maps where a patch map is defined as a matrix containing patches that a kernel will affect
        self.convLayers = [[0]] # array of each layer to be filtered, including input
        self.inactiveConvLayers = [] # convLayers that have not been passed into an activation function
        self.kernels = [] # (x, y) size filters that act as weights
        self.wrt_kernels = [] # gradient wrt the kernel at each position
        self.cfunc_vec = []
        self.d_cfunc_vec = []
        self.meank = None
        self.cLayerCount = 1

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
                arr_in = numpy.expand_dims(img_tensor, 0)

        if self.norm:
            arr_in = unitTensor(arr_in)

        del self.image_tensor
        del self.convLayers[0]
        self.convLayers.insert(0, [0])

        self.image_tensor = arr_in
        self.convLayers[0] = arr_in

    def _getPatchMap(self, image, patch_x, patch_y):
        preserved = image.shape[0:-2]
        # process patch

        if len(image.shape) < 3:
            image = numpy.expand_dims(image, 0)
        elif len(image.shape) > 3:
            prod = numpy.prod(image.shape[0:-2])
            image = numpy.reshape(image, (prod, image.shape[-2], image.shape[-1]))

        # get the patch map

        x_slide = image.shape[2] - patch_x + 1
        y_slide = image.shape[1] - patch_y + 1
        final_tensor = []

        for f in range(image.shape[0]):
            p_tensor = []
            for y in range(y_slide):
                p_mat = []
                for x in range(x_slide):
                    p_mat.append(image[f][y:y + patch_y, x:x + patch_x].flatten())
                p_tensor.append(p_mat)
            final_tensor.append(p_tensor)

        final_tensor = numpy.array(final_tensor)

        final = numpy.append(preserved, final_tensor.shape[1:4])
        final_tensor = final_tensor.reshape(final)

        return final_tensor



    def addLayerC(self, patch_length, stride, num_kernels, function):
        self.kernels.append(numpy.random.uniform(MIN, MAX, (num_kernels, patch_length ** 2, 1)))
        self.wrt_kernels.append([0])
        self.convLayers.append([0])
        self.inactiveConvLayers.append([0])
        self.patch_sizes.append(numpy.array([patch_length, patch_length, 0]))
        self.strides.append(stride)

        self.patch_area.append(patch_length**2)
        self.cLayerCount += 1

        self.cfunc_vec.append(funcDict.get(function))
        self.d_cfunc_vec.append(d_funcDict.get(function))

    def initializeC(self):
        self.patch_maps = [0] * self.cLayerCount

        self.forwardC()
        self.initialize()


        self.mk = copy.deepcopy(self.kernels)
        self.vk = copy.deepcopy(self.kernels)

        for i in range(len(self.mk)):
            self.mk[i] = numpy.array(self.mk[i])
            self.vk[i] = numpy.array(self.vk[i])

        for i in range(len(self.mk)):
            self.mk[i] *= 0.0
            self.vk[i] *= 0.0


    def forwardC(self):

        # if self.training and self.dropout > 0:
        #     layer_size = numpy.size(self.convLayers[0])
        #     d_mask = numpy.zeros(layer_size)
        #
        #     d_mask[0:int((1 - self.dropout) * layer_size)] = 1
        #
        #     d_mask = d_mask.reshape((self.convLayers[0]).shape)
        #     numpy.random.shuffle(d_mask)
        #
        #     self.convLayers[0] = (d_mask * self.convLayers[0]) / self.dropout
        #     del d_mask

        for i in range(len(self.convLayers)-1):
            del self.patch_maps[i]
            del self.inactiveConvLayers[i]
            del self.convLayers[i+1]

            self.patch_maps.insert(i, None)
            self.inactiveConvLayers.insert(i, None)
            self.convLayers.insert(i + 1, [0])

            self.patch_maps[i] = self._getPatchMap(self.convLayers[i], self.patch_sizes[i][0], self.patch_sizes[i][1])

            self.inactiveConvLayers[i] = numpy.squeeze(numpy.dot(self.patch_maps[i], self.kernels[i]), -1)

            self.inactiveConvLayers[i] = numpy.moveaxis(self.inactiveConvLayers[i], -1, 0)

            self.convLayers[i+1] = self.cfunc_vec[i](self.inactiveConvLayers[i])

        self.setInput(self.convLayers[-1])

    def forwardAll(self):
        self.forwardC()
        self.feedforward()

    def computeGradientC(self, true_l):


        self.forwardAll()
        self.computeGradient(true_l)
        for i in range(len(self.inactiveConvLayers) - 1, -1, -1):
            # find derivative of loss wrt the ith kernel
            del self.wrt_kernels[i]
            self.wrt_kernels.insert(i, None)

            last_deriv = self.wrt_cLayer.reshape(self.inactiveConvLayers[i].shape) * self.d_cfunc_vec[i](self.inactiveConvLayers[i])

            this_deriv = numpy.expand_dims(last_deriv,-1)

            this_deriv = numpy.multiply(this_deriv, self.patch_maps[i])

            self.wrt_kernels[i] = numpy.sum(this_deriv, tuple(range(1, len(this_deriv.shape)-1)))

            self.wrt_kernels[i] = numpy.reshape(self.wrt_kernels[i], self.kernels[i].shape)

            # Compute grad wrt next layer for next kernel

            to_pad = []

            for v in range(len(last_deriv.shape)):
                if v == len(last_deriv.shape)-1 or v == len(last_deriv.shape)-2:
                    to_pad.append([self.patch_sizes[i][0]-1, self.patch_sizes[i][1]-1])
                else:
                    to_pad.append([0, 0])

            padded = numpy.pad(last_deriv, to_pad)

            r_kernel = numpy.flip(self.kernels[i], 1)

            d_map = self._getPatchMap(padded, self.patch_sizes[i][0], self.patch_sizes[i][1])

            temp = []

            for k in range(r_kernel.shape[0]):
                temp.append(numpy.dot(d_map[k], r_kernel[k]))

            self.wrt_cLayer = numpy.sum(temp, 0)





    def updateADAMC(self, wrt_kernels):
        self.tk += 1
        for k in range(len(self.mk)):
            self.mk[k] = wrt_kernels[k] * (1.0 - self.B_1) + self.mk[k] * self.B_1
            self.vk[k] = numpy.square(wrt_kernels[k]) * (1.0 - self.B_2) + self.vk[k] * self.B_2

            mk_hat = copy.deepcopy(self.mk[k] / (1.0 - math.pow(self.B_1, self.tk)))
            vk_hat = copy.deepcopy(self.vk[k] / (1.0 - math.pow(self.B_2, self.tk)))

            self.kernels[k] -= self.a_adam * numpy.divide(mk_hat, numpy.sqrt(vk_hat) + self.e)

            del mk_hat
            del vk_hat

    def updateSGDC(self, wrt_kernels):
        for i in range(len(self.kernels)):
            self.kernels[i] -= self.a_sgd * wrt_kernels[i]

    def updateC(self, expected_in, expected, o_type):
        self.setImage(expected_in)
        self.computeGradientC(expected)
        self.updaters.get(o_type)(self.d_weight_vec, self.d_bias_vec)
        self.updatersC.get(o_type)(self.wrt_kernels)

    def train(self, training_in, training_out, o_type, t_type):
        self.iterations = int(len(training_out) / self.batch_size)

        self.meanw = copy.deepcopy(self.weights)
        self.meanb = copy.deepcopy(self.bias)
        self.meank = copy.deepcopy(self.kernels)

        for _epoch in range(self.epochs):
            #print("Epoch: " + str(_epoch))
            for i in range(self.iterations):
            #    print("Iteration: " + str(i))
                for a in range(len(self.meanw)):
                    self.meanw[a] *= 0
                    self.meanb[a] *= 0
                for a in range(len(self.meank)):
                    self.meank[a] *= 0
                for j in range(self.batch_size):
                    self.setImage(training_in[self.batch_size * i + j])
                    self.computeGradientC([training_out[self.batch_size * i + j]])
                    for k in range(len(self.meanw)):
                        self.meanw[k] += numpy.array(self.d_weight_vec[k])
                        self.meanb[k] += numpy.array(self.d_bias_vec[k])
                    for k in range(len(self.meank)):
                        self.meank[k] += numpy.array(self.wrt_kernels[k])
                if t_type == "avg":
                    for b in range(len(self.meanw)):
                        self.meanw[b] /= self.batch_size
                        self.meanb[b] /= self.batch_size
                    for k in range(len(self.meank)):
                        self.meank[k] /= self.batch_size

                self.updaters.get(o_type)(self.meanw, self.meanb)
                self.updatersC.get(o_type)(self.meank)
                print(f"iteration: {i}", end="")
                print("\r", end="")
    def error(self, true_in, true_l):
        self.setImage(true_in)
        self.forwardAll()

        if len(true_l) != len(self.layers[-1]) and len(true_l) == 1:
            expected = vecify(true_l[0], len(self.layers[-1]))
        elif len(true_l) == len(self.layers[-1]):
            expected = numpy.array(true_l)
        else:
            print("Incomparable Dimensions")
            exit(-1)

        return self.errorFunc(self.layers[-1], expected)
    def singlePredict(self, test_in, test_out):
        error = self.error(test_in, test_out)
        print(f"Predicted {numpy.argmax(self.layers[-1])}")
        print(f"Accuracy: {numpy.array(self.layers[-1])}")
        return error
    def eval(self, test_in_arr, test_out_arr):
        avg_loss = 0
        avg_acc = 0

        for i in range(len(test_out_arr)):
            avg_loss += self.error(test_in_arr[i], [test_out_arr[i]])
            avg_acc += self.layers[-1][test_out_arr[i]]

        print(f"Average loss: {avg_loss/len(test_out_arr)}")
        print(f"Average accuracy: {avg_acc/len(test_out_arr)}")

