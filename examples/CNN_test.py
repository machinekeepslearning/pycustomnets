import pycustomnets as pc
import numpy as np
import PIL.Image
import matplotlib.pyplot as pyp

from keras._tf_keras.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_tensor = np.asarray(PIL.Image.open("PATH TO IMAGE HERE"))

def main():
    CNN = pc.ConvModel(1, 2)
    CNN.setImage(image_tensor)
    CNN.addLayerC(2, 1)
    CNN.addLayer(10, "softmax")
    CNN.setError("b_cross_entropy")
    CNN.initializeC()
    print(CNN.error([0]))
    print(CNN.kernels)
    for i in range(10):
        CNN.setImage(image_tensor)
        CNN.optimizeC([0], "adam")
    print(CNN.error([0]))
    print(CNN.kernels)


if __name__ == "__main__":
    main()




