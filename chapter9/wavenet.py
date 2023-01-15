import numpy as np
import tensorflow as tf


# mu law encoding
def mu_law(x, mu=255):
  x = np.clip(x, -1, 1)
  x_mu = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
  return ((x_mu + 1) / 2 * mu).astype('int16')


# mu law decoding
def inv_mu_law(x, mu=255.0):
  x = np.array(x).astype(np.float32)
  y = 2. * (x - (mu + 1.) / 2.) / (mu + 1.)
  return np.sign(y) * (1. / mu) * ((1. + mu) ** np.abs(y) - 1.)


class ResidualConv1DGAU(tf.keras.layers.Layer):
  def __init__(self, channels, kernel_size, dropout, dilation_rate=1):
    super().__init__()
    self.dilated_conv = tf.keras.layers.Conv1D(channels * 2,
                                               kernel_size=kernel_size,
                                               padding='causal',
                                               dilation_rate=dilation_rate)
    self.dropout = tf.keras.layers.Dropout(rate=dropout)
    self.conv = tf.keras.layers.Conv1D(channels * 2,
                                       kernel_size=1,
                                       padding='causal')

  def call(self, inputs):
    # inputs: [batch_size, time_steps, channels], or [B, T, C] for short
    x = self.dropout(inputs)  # [B, T, C]
    x = self.dilated_conv(x)  # [B, T, 2C]
    # [B, T, 2C] -> [B, T, C], [B, T, C]
    x_tanh, x_sigmoid = tf.split(x, num_or_size_splits=2, axis=2)
    # Optional conditional inputs here (conv up-sample + split)
    x = tf.nn.tanh(x_tanh) * tf.nn.sigmoid(x_sigmoid)  # [B, T, C]
    x = self.conv(x)  # [B, T, 2C]
    # [B, T, 2C] -> [B, T, C], [B, T, C]
    out, skip = tf.split(x, num_or_size_splits=2, axis=2)
    return (out + inputs) / np.sqrt(2.0), skip


# block_size = 10 <-> dilation_rate = 1, 2, 4, ..., 512
block_size, num_blocks = 10, 4
batch_size, time_steps, vocab_size = 4, 16000, 256
initial_channels, conv_kernel_size, output_channels = 128, 2, 256
dropout_rate = 0.05

x_input = tf.keras.layers.Input(shape=(time_steps,))  # [B, T]
# Expand the channel dim for 1d convolution
x = tf.expand_dims(x_input, axis=-1)  # [B, T, C=1]
x = tf.keras.layers.Conv1D(filters=initial_channels,
                           kernel_size=1,
                           strides=1, padding='causal',
                           activation='relu')(x)  # [B, T, C]

skips = 0
for block_index in range(num_blocks):
  for stack_index in range(block_size):
    x, new_skips = ResidualConv1DGAU(channels=initial_channels,
                                     kernel_size=conv_kernel_size,
                                     dropout=dropout_rate,
                                     dilation_rate=2 ** stack_index)(x)
    skips += new_skips
skips = skips / np.sqrt(block_size * num_blocks)

out = tf.keras.Sequential(layers=[tf.keras.layers.ReLU(),
                                  tf.keras.layers.Conv1D(
                                    filters=output_channels,
                                    kernel_size=1,
                                    strides=1,
                                    padding='causal',
                                    use_bias=True
                                  ),
                                  tf.keras.layers.ReLU(),
                                  tf.keras.layers.Conv1D(filters=vocab_size,
                                                         kernel_size=1,
                                                         strides=1,
                                                         padding='causal',
                                                         use_bias=True
                                                         )
                                  ])(skips)  # [B, T, V]

model = tf.keras.Model(inputs=x_input, outputs=out)
model.summary()

wave_form = tf.random.uniform(shape=[batch_size, time_steps])
y = model(wave_form)
print(tf.shape(y), y)
