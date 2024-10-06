import pycustomnets as pc
import numpy

from keras._tf_keras.keras.datasets import mnist


epoches = 1

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

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

def main():
    NN = pc.ModelStandard()
    NN.setInput(flatten(train_X[0]))
    NN.addLayer(128, "relu")
    NN.addLayer(10, "softmax")
    NN.setError("b_cross_entropy")
    NN.initialize()


    for i in range(epoches):
        for j in range(10000):
            k = j #randint(0, 60000)
            NN.setInput(train_X[k])
            #print(NN.weights)
            print(NN.error([train_Y[k]]))
            #NN.out()
            NN.optimize([train_Y[k]], "adam")

    sum = 0

    for i in range(10000):


        NN.setInput(test_X[i])
        NN.out()
        print(NN.error([test_Y[i]]))
        sum += NN.error([test_Y[i]])
        print(vecify(test_Y[i]))

    print("avg error: " + str(sum/10000))

if __name__ == "__main__":
    main()

