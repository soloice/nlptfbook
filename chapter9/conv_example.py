import tensorflow as tf


b, t, c, k = 1, 8, 1, 3
conv1 = tf.keras.layers.Conv1D(filters=1, kernel_size=k, strides=1,
                               padding='causal')
conv2 = tf.keras.layers.Conv1D(filters=1, kernel_size=k, strides=1,
                               padding='valid')

x = tf.random.normal(shape=(b, t, c))
# print(x)
print('conv1 output:', conv1(x))  # [b, t, c]
print('conv2 output:', conv2(x))  # [b, t-k+1, c]

print('Share parameter between conv1 and conv2!')
conv2.kernel = conv1.kernel

# causal conv = left zero padding + valid conv. Let's assert this!
x_padded = tf.concat([tf.zeros(shape=(b, k - 1, c), dtype=tf.float32), x],
                     axis=1)
tf.debugging.assert_near(conv2(x_padded), conv1(x))

conv3 = tf.keras.layers.Conv1D(filters=1, kernel_size=3, strides=1,
                               dilation_rate=5, padding='causal')
print(conv3(x))
