import time

from pycustomnets import *

def training_func(x):
    return 2 * x - 2


def main():
    training_data_in = []
    training_data_out = []

    for i in range(40):
        training_data_in.append([i + 1])
        training_data_out.append([training_func(i + 1)])

    epoches = 500
    NN = ModelStandard(1, 1, False, False)
    NN.setInput(training_data_in[0])
    NN.setError("mse")
    NN.addLayer(1, "lrelu")
    NN.initialize()
    for _ in range(epoches):
        for i in range(len(training_data_in)):
            NN.update(training_data_in[i], training_data_out[i], "sgd")

    print("Predicting 50 -> " + str(training_func(50)))
    NN.singlePredict([50], [training_func(50)])


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(end - start)
