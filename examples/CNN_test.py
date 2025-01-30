import pycustomnets as pc
from keras._tf_keras.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def main():
    CNN = pc.ConvModel(100, 10)
    CNN.setImage(x_train[0])
    CNN.addLayerC(2, 1, 1)
    CNN.addLayer(128, "lrelu")
    CNN.addLayer(10, "softmax")
    CNN.setError("bce")
    CNN.initializeC()
    CNN.a_adam = 0.001
    avg_error = 0
    for i in range(1000):
        avg_error += CNN.singlePredict(x_train[i], [y_train[i]], True)
    avg_error_before = avg_error / 1000
    CNN.train(x_train[0:1000], y_train[0:1000], "adam", "none")
    avg_error = 0
    for i in range(1000):
        avg_error += CNN.singlePredict(x_train[i], [y_train[i]], True)
    print(avg_error_before)
    print(avg_error/1000)



if __name__ == "__main__":
    main()




