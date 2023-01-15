import tensorflow as tf

# SimpleRNN: Method I
rnn = tf.keras.layers.SimpleRNN(64, return_sequences=True)
model = tf.keras.Sequential([rnn])
model.build(input_shape=(None, None, 50))
print(model.summary())

# SimpleRNN: Method II
cell = tf.keras.layers.SimpleRNNCell(64)
model = tf.keras.Sequential([tf.keras.layers.RNN(cell, return_sequences=True)])
model.build(input_shape=(None, None, 50))
print(model.summary())

# LSTM: Method I
rnn = tf.keras.layers.LSTM(64, return_sequences=True)
model = tf.keras.Sequential([rnn])
model.build(input_shape=(None, None, 50))
print(model.summary())

# LSTM: Method II
cell = tf.keras.layers.LSTMCell(64)
model = tf.keras.Sequential([tf.keras.layers.RNN(cell, return_sequences=True)])
model.build(input_shape=(None, None, 50))
print(model.summary())

# GRU: Method I
rnn = tf.keras.layers.GRU(64, return_sequences=True)
model = tf.keras.Sequential([rnn])
model.build(input_shape=(None, None, 50))
print(model.summary())

# GRU: Method II
cell = tf.keras.layers.GRUCell(64)
model = tf.keras.Sequential([tf.keras.layers.RNN(cell, return_sequences=True)])
model.build(input_shape=(None, None, 50))
print(model.summary())
