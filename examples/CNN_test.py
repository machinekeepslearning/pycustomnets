import pycustomnets as pc
from keras._tf_keras.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def main():
    CNN = pc.ConvModel(2, 4, True, 0.3)
    CNN.setImage(x_train[0])
    CNN.addLayerC(3, 1, 64)
    CNN.addLayerC(2, 1, 32)
    CNN.addLayer(128, "lrelu")
    CNN.addLayer(10, "softmax")
    CNN.setError("bce")
    CNN.initializeC()
    CNN.a_adam = 0.001

    CNN.singlePredict(x_test[8], [y_test[8]])

    CNN.train(x_train[0:10], y_train[0:10], "adam")

    CNN.singlePredict(x_test[8], [y_test[8]])



if __name__ == "__main__":
    main()