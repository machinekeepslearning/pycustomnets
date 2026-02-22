import numpy
import time

import src.pycustomnets as pc

from keras.datasets import mnist

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()


def main():
    NN = pc.ModelStandard(1000, 3, True, 0)
    NN.setInput(train_X[0])
    NN.addLayer(128, "relu")
    NN.addLayer(10, "softmax")
    NN.setError("ce")
    NN.initialize()

    '''for i in range(epoches):
        for j in range(10000):
            k = j #randint(0, 60000)
            NN.setInput(train_X[k])
            #print(NN.weights)
            print(NN.error([train_Y[k]]))
            #NN.out()
            NN.optimize([train_Y[k]], "adam")'''

    start = time.time()
    NN.train(train_X, train_Y, "adam")
    end = time.time()

    NN.eval(test_X, test_Y)
    print(f"training took {end - start} seconds")

    NN.probPredict(test_X[0], [test_Y[0]])


if __name__ == "__main__":
    main()
