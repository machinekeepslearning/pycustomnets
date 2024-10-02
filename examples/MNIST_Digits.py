import pycustomnets as pc
import numpy

from keras._tf_keras.keras.datasets import mnist

import time


epoches = 2

def flatten(x):
    flat = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            flat.append(x[i][j])
    return flat

def vecify(x):
    vec = []
    for i in range(10):
        if i == x:
            vec.append(1)
        else:
            vec.append(0)

    return numpy.array(vec)

def training_func(x):
    return 2*x+1

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

def main():
    NN = pc.ModelStandard()
    NN.setInput(flatten(train_X[0]))
    NN.addLayer(128, "relu")
    NN.addLayer(10, "softmax")
    NN.setError("b_cross_entropy")
    NN.initialize()

    NN.setInput(flatten(test_X[0]))
    orig1 = NN.error([test_Y[0]])

    NN.setInput(flatten(test_X[10]))
    orig2 = NN.error([test_Y[10]])


    for i in range(epoches):
        for j in range(3000):
            NN.setInput(flatten(train_X[j]))
            #print(NN.weights)
            print(NN.error([train_Y[j]]))
            #NN.out()
            NN.optimize([train_Y[j]], "adam")

    NN.setInput(flatten(test_X[0]))
    NN.out()
    print("Before Optimizing: " + str(orig1))
    print(NN.error([test_Y[0]]))
    print(vecify(test_Y[0]))

    NN.setInput(flatten(test_X[10]))
    NN.out()
    print("Before optimizing: " + str(orig2))
    print(NN.error([test_Y[10]]))
    print(vecify(test_Y[10]))


if __name__ == "__main__":
    main()
