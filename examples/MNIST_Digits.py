import matplotlib.pyplot as plt
import numpy

import pycustomnets as pc

from keras._tf_keras.keras.datasets import mnist

epoches = 2

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

def main():
    NN = pc.ModelStandard(1000, 3)
    NN.setInput(train_X[0])
    NN.addLayer(128, "relu")
    NN.addLayer(10, "softmax")
    NN.setError("b_cross_entropy")
    NN.initialize()


    '''for i in range(epoches):
        for j in range(10000):
            k = j #randint(0, 60000)
            NN.setInput(train_X[k])
            #print(NN.weights)
            print(NN.error([train_Y[k]]))
            #NN.out()
            NN.optimize([train_Y[k]], "adam")'''

    NN.train(train_X, train_Y, "adam")

    NN.setInput(train_X[3728])
    NN.out()
    print(NN.error([train_Y[3728]]))

    plt.imshow(train_X[3728])
    plt.pause(10)

if __name__ == "__main__":
    main()


