import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# Build an RNN
model = tf.keras.Sequential([tf.keras.layers.LSTM(80),
                             tf.keras.layers.Dense(10)])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32)
model.evaluate(x_test, y_test, batch_size=64)
