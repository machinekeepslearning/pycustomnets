import pycustomnets as pc

from keras._tf_keras.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def main():
    CNN = pc.ConvModel(1, 2)
    CNN.setImage(x_train[0])
    CNN.addLayerC(2, 1)
    CNN.addLayer(10, "softmax")
    CNN.setError("b_cross_entropy")
    CNN.initializeC()
    print(CNN.error([y_train[0]]))
    for _ in range(10):
        CNN.optimizeC([y_train[0]], "adam")
        #print(CNN.error([y_train[0]]))
    print(CNN.error([y_train[0]]))


if __name__ == "__main__":
    main()





