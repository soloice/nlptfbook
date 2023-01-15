import tensorflow as tf
from tensorflow.python.util import nest


class LayerNormGRUCell(tf.keras.layers.AbstractRNNCell):
  def __init__(self, units):
    super(LayerNormGRUCell, self).__init__()
    self.units = units
    zero_init = tf.zeros_initializer()
    self.b_z = self.add_weight(name='b_z', shape=(self.units,),
                               dtype='float32', initializer=zero_init,
                               trainable=True)
    self.b_r = self.add_weight(name='b_r', shape=(self.units,),
                               dtype='float32', initializer=zero_init,
                               trainable=True)
    self.b_xh = self.add_weight(name='b_xh', shape=(self.units,),
                                dtype='float32', initializer=zero_init,
                                trainable=True)
    self.b_hh = self.add_weight(name='b_hh', shape=(self.units,),
                                dtype='float32', initializer=zero_init,
                                trainable=True)
    u_init = tf.random_normal_initializer(stddev=(1 / self.units) ** 0.5)
    self.U_r = self.add_weight(name='U_r', shape=(self.units, self.units),
                               dtype='float32', initializer=u_init,
                               trainable=True)
    self.U_z = self.add_weight(name='U_z', shape=(self.units, self.units),
                               dtype='float32', initializer=u_init,
                               trainable=True)
    self.U_h = self.add_weight(name='U_h', shape=(self.units, self.units),
                               dtype='float32', initializer=u_init,
                               trainable=True)
    self.ln = [tf.keras.layers.LayerNormalization() for _ in range(6)]
    for ln in self.ln:
      ln.build(input_shape=(None, self.units))

  @property
  def state_size(self):
    return self.units

  @property
  def output_size(self):
    return self.units

  def build(self, input_shape):
    # input shape = [batch_size, dimension]
    dev = (2 / (input_shape[-1] + self.units)) ** 0.5
    w_init = tf.random_normal_initializer(stddev=dev)
    self.W_r = self.add_weight(name='W_r', shape=(input_shape[-1], self.units),
                               dtype='float32', initializer=w_init)
    self.W_z = self.add_weight(name='W_z', shape=(input_shape[-1], self.units),
                               dtype='float32', initializer=w_init)
    self.W_h = self.add_weight(name='W_h', shape=(input_shape[-1], self.units),
                               dtype='float32', initializer=w_init)
    self.built = True

  def call(self, inputs, states):
    # inputs: [batch_size, input_size], states: [batch_size, state_size]
    # output: [batch_size, output_size], new_states: [batch_size, state_size]
    is_nested = nest.is_nested(states)
    states = states[0] if is_nested else states
    # [batch_size, state_size]
    z = tf.sigmoid(self.ln[0](inputs @ self.W_z) +
                   self.ln[1](states @ self.U_z) + self.b_z)
    # [batch_size, state_size]
    r = tf.sigmoid(self.ln[2](inputs @ self.W_r) +
                   self.ln[3](states @ self.U_r) + self.b_r)
    # [batch_size, state_size]
    h_to_use = r * (self.ln[5](states @ self.U_h + self.b_hh))
    # [batch_size, state_size]
    h_cand = tf.tanh(self.ln[4](inputs @ self.W_h + self.b_xh + h_to_use))
    h_new = (1 - z) * states + z * h_cand
    return h_new, [h_new] if is_nested else h_new


ln_gru_cell = LayerNormGRUCell(64)
ln_gru = tf.keras.layers.RNN(ln_gru_cell, return_sequences=True)
model = tf.keras.Sequential([ln_gru])

# [batch_size, time_steps, input_dim]
model.build(input_shape=(None, None, 50))
print(model.summary())
