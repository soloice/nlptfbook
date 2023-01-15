import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def peak_function_numpy(x, y):
  z = 3 * (1 - x) ** 2 * np.exp(- x ** 2 - (y + 1) ** 2) \
      - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2) \
      - 1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)
  return z


def peak_function(x):
  z = 3 * (1 - x[0]) ** 2 * tf.exp(- x[0] ** 2 - (x[1] + 1) ** 2) \
      - 10 * (x[0] / 5 - x[0] ** 3 - x[1] ** 5) * tf.exp(-x[0] ** 2 - x[1] ** 2) \
      - 1 / 3 * tf.exp(-(x[0] + 1) ** 2 - x[1] ** 2)
  return z


figure = plt.figure()
# ax = Axes3D(figure)
ax = figure.gca(projection="3d")

xs = []
ys = []

# x = tf.Variable([0.113, -1.6])
x = tf.Variable([-0.5, 1.5])
learning_rate = 1e-2

for step in range(100):
  with tf.GradientTape() as t:
    loss = peak_function(x)
  grad_x = t.gradient(loss, x)
  grad_norm = tf.sqrt(tf.reduce_sum(grad_x ** 2))
  x.assign_sub(learning_rate * grad_x)
  x_value = x.numpy()
  print('loss: {:.4f}, (x, y): ({:.4f}, {:.4f})'.format(loss, x_value[0], x_value[1]))
  print('  (dx, dy): ({:.4f}, {:.4f}), grad_norm: {:.4f}'
        .format(grad_x.numpy()[0], grad_x.numpy()[1], grad_norm.numpy()))
  xs.append(x_value[0])
  ys.append(x_value[1])

limit = 3
x1 = np.linspace(-limit, limit, 1000)
y1 = np.linspace(-limit, limit, 1000)
x, y = np.meshgrid(x1, y1)
z = peak_function_numpy(x, y)
ax.plot_surface(x, y, z, cmap="rainbow")

xs = np.array(xs)
ys = np.array(ys)
zs = peak_function_numpy(xs, ys)
print('zs: ', zs)
ax.plot3D(xs, ys, zs, 'o-', color='black')

# z_line = np.linspace(0, 15, 30)
# x_line = np.cos(z_line)
# y_line = np.sin(z_line)
# ax.plot3D(x_line, y_line, z_line, 'x-', color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

plt.show()
