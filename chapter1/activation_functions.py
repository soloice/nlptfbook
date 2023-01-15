import matplotlib.pyplot as plt
import numpy as np


def relu(x):
  return (x >= 0) * x, (x >= 0).astype(np.float)


def sigmoid(x):
  s = 1 / (1 + np.exp(-x))
  return s, s * (1 - s)


def tanh(x):
  s = sigmoid(x)[0]
  t = 2 * s - 1
  return t, 1 - t ** 2


xs = np.linspace(-5.0, 5.0, 1000)
relu_f, relu_g = relu(xs)
p1, = plt.plot(xs, relu_f)
p2, = plt.plot(xs, relu_g)

tanh_f, tanh_g = tanh(xs)
p3, = plt.plot(xs, tanh_f)
p4, = plt.plot(xs, tanh_g)

sigmoid_f, sigmoid_g = sigmoid(xs)
p5, = plt.plot(xs, sigmoid_f)
p6, = plt.plot(xs, sigmoid_g)

l1 = plt.legend([p1, p2, p3, p4, p5, p6],
                ["ReLU", "ReLU'", "tanh", "tanh'", "sigmoid", "sigmoid'"])
plt.show()
