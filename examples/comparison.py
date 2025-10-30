import os

import src.pycustomnets as pc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import models, layers, datasets
from keras import losses

import time

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

def main():
    NN = pc.ModelStandard(1, 1, True, 0)
    NN.setInput(x_train[0])
    NN.addLayer(128, "relu")
    NN.addLayer(10, "softmax")
    NN.setError("ce")
    NN.initialize()
    NN.a_adam = 0.001

    start = time.time()
    NN.train(x_train, y_train, "adam")
    end = time.time()
    print(f"pycustomnets train training ended after {end - start} seconds.")
    print("pycustomnets testing set evaluation:")
    NN.eval(x_test, y_test)


def benchmark():
    model = models.Sequential()
    model.add(layers.Input((28, 28)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss= losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    start = time.time()
    model.fit(x_train, y_train, epochs=1, batch_size=1)
    end = time.time()
    print(f"tensorflow training ended after {end - start} seconds.")
    print("tensorflow testing set evaluation:")
    model.evaluate(x_test, y_test, batch_size=1, verbose=2)


if __name__ == "__main__":
    main()
    benchmark()