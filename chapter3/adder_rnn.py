import tensorflow as tf
from tensorflow.python.util import nest


class AdderRNNCell(tf.keras.layers.AbstractRNNCell):
  def __init__(self, **kwargs):
    super(AdderRNNCell, self).__init__(**kwargs)

  @property
  def state_size(self):
    return 1

  @property
  def output_size(self):
    return 1

  def call(self, inputs, states):
    # inputs: [batch_size, input_size], states: [batch_size, state_size]
    # output: [batch_size, output_size], new_states: [batch_size, state_size]
    is_nested = nest.is_nested(states)
    states = states[0] if is_nested else states
    current_sum = tf.reduce_sum(tf.concat([inputs, states], axis=1),
                                axis=1, keepdims=True)  # [B, 1]
    output = current_sum % 10
    carry = current_sum // 10
    return output, [carry] if is_nested else carry


adder_cell = AdderRNNCell()
print(adder_cell(tf.constant([[6, 3]]), tf.constant([[0]])))
adder_rnn = tf.keras.layers.RNN(adder_cell, return_sequences=True,
                                return_state=True, go_backwards=True)

# [B, T, D] = [1, 3, 2]
# Simulate:
#    6 5 8
#  + 3 2 4
# ---------
#    9 8 2
inputs = tf.constant([[[6, 3], [5, 2], [8, 4]]])
print(adder_rnn(inputs))
