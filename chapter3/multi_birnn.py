import tensorflow as tf

# Bidirectional RNN
model = tf.keras.Sequential()
cell = tf.keras.layers.SimpleRNN(128, return_sequences=True)
model.add(tf.keras.layers.Bidirectional(layer=cell))
# [batch_size, time_steps, input_dim]
model.build(input_shape=(None, None, 50))
print(model.summary())

# Multi-Layer RNN: Method I
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(128, return_sequences=True))
model.add(tf.keras.layers.SimpleRNN(64, return_sequences=True))
# [batch_size, time_steps, input_dim]
model.build(input_shape=(None, None, 50))
print(model.summary())

# Multi-Layer RNN: Method II
model = tf.keras.Sequential()
cell1 = tf.keras.layers.SimpleRNNCell(128)
cell2 = tf.keras.layers.SimpleRNNCell(64)
cell = tf.keras.layers.StackedRNNCells([cell1, cell2])
model.add(tf.keras.layers.RNN(cell, return_sequences=True))
model.build(input_shape=(None, None, 50))
print(model.summary())

# Multi-Layer BiRNN: Separated
forward_cells, backward_cells = [], []
for layer_size in [128, 64]:
  forward_cells.append(tf.keras.layers.SimpleRNNCell(layer_size))
  backward_cells.append(tf.keras.layers.SimpleRNNCell(layer_size))
forward_multi_cell = tf.keras.layers.StackedRNNCells(forward_cells)
forward_rnn = tf.keras.layers.RNN(forward_multi_cell, return_sequences=True)
backward_multi_cell = tf.keras.layers.StackedRNNCells(backward_cells)
backward_rnn = tf.keras.layers.RNN(backward_multi_cell, return_sequences=True,
                                   go_backwards=True)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Bidirectional(layer=forward_rnn,
                                        backward_layer=backward_rnn))
# [batch_size, time_steps, input_dim]
model.build(input_shape=(None, None, 50))
print(model.summary())

# Multi-Layer BiRNN: Fused
model = tf.keras.Sequential()
forward_layers, backward_layers = [], []
for layer_size in [128, 64]:
  forward_layer = tf.keras.layers.SimpleRNN(layer_size,
                                            return_sequences=True)
  backward_layer = tf.keras.layers.SimpleRNN(layer_size,
                                             return_sequences=True,
                                             go_backwards=True)
  model.add(tf.keras.layers.Bidirectional(layer=forward_layer,
                                          backward_layer=backward_layer))

model.build(input_shape=(None, None, 50))
print(model.summary())
