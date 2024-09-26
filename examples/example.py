from pycustomnets import *

def training_func(x):
    return 2 * x - 2


def main():
    training_data_in = []
    training_data_out = []

    for i in range(40):
        training_data_in.append([i + 1])
        training_data_out.append([training_func(i + 1)])

    #print(training_data_out)
    #print(training_data_in)

    epoches = 500
    NN = ModelStandard()
    NN.setInput(training_data_in[0])
    NN.addLayer(1, "lrelu")
    NN.initialize()
    for _ in range(epoches):
        for i in range(len(training_data_in)):
            NN.setInput(training_data_in[i])
            #print("Actual: " + str(training_data_out[i]))
            #NN.out()
            print("Error: " + str(NN.error(training_data_out[i])))
            NN.optimize(training_data_out[i], "adam")

    print("Predicting 50: " + str(training_func(50)))
    NN.setInput([50])
    NN.out()


if __name__ == "__main__":
    main()
