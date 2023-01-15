import numpy as np
import tensorflow as tf


class LinearModel(object):
  def __init__(self):
    self.W = tf.Variable(1.0)
    self.b = tf.Variable(0.0)

  def __call__(self, x):
    return self.W * x + self.b


def loss_fn(y_pred, y_true):
  return tf.reduce_mean(tf.square(y_pred - y_true))


num_samples = 3000
data_xs = 10.0 * np.random.random(num_samples) - 5.0
noise = 0.01 * np.random.random(num_samples)
data_ys = 2.0 * data_xs - 1.0 + noise

model = LinearModel()
batch_size = 20
learning_rate = 0.1
for i in range(0, num_samples, batch_size):
  xs = data_xs[i:i + batch_size]
  ys = data_ys[i:i + batch_size]
  with tf.GradientTape() as t:
    prediction = model(xs, training=True)
    loss = loss_fn(prediction, ys)
  dW, db = t.gradient(loss, [model.W, model.b])
  model.W.assign_sub(learning_rate * dW)
  model.b.assign_sub(learning_rate * db)
  if i % 300 == 0:
    print("At step {}, loss = {:.4f}, W = {:.4f}, b = {:.4f}".
          format(i, loss, model.W.numpy(), model.b.numpy()))
