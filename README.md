# pycustomnets
This is a little project that I've been working on for a while with the primary goal of gaining a more in depth understanding of deep learning and honing various skills

This is meant to be my version of tensorflow and while its definitely not as good as tensorflow for now, It does have some similar basic functions.

This project is also a package which can be downloaded through pip (however it might not very up to date)

`pip install pycustomnets`

IMPORTANT NOTE: If you plan on using this, expected values and input values must be arrays or else the model won't work. Single value expected values can be represented as a single element array (`[0]`, `[1]`, etc)

Activation functions
  - Leaky Relu ("lrelu")
  - Relu ("relu")
  - Sigmoid ("sigmoid")
  - Softmax ("softmax")

Error Functions
  - Mean Squared Error ("mse")
  - Cross Entropy ("cross_entropy")
  - Binary Cross Entropy ("b_cross_entropy")

Optimization algorithms
  - Stochastic Gradient Descent ("sgd")
  - ADAM ("adam")

In the future, I hope to add Reinforcement Learning algorithms, transformers and other stuff.

Example usage of pycustomnets can be found in the examples folder:
  - linearfunction: Trains a neural network to mimic a function
  - MNIST_Digits: Trains a neural network to identify handwritten digits from the mnist data base (Network architecture (784, 128, 10) was made similar to example shown in                   tensorflow demo for comparison purposes)

Convolutional Neural Networks have been added. Example usage coming soon

Better readme coming eventually
