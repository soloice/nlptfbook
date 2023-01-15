import tensorflow as tf


class SingleGate(tf.keras.layers.Layer):
  def __init__(self, input_dim, hidden_dim, num_child, activation='sigmoid'):
    super().__init__()
    self.num_child = num_child
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.kernel = self.add_weight('W', [input_dim, hidden_dim])
    self.bias = self.add_weight('b', [hidden_dim])
    self.activation = tf.keras.layers.Activation(activation)
    self.kernels = [self.add_weight(f'W_{i}', [hidden_dim, hidden_dim])
                    for i in range(num_child)]

  def call(self, inputs, hiddens):
    # inputs: [B, D_in], hiddens: [B, N, H]
    if inputs is None:
      hidden = 0.0
    else:
      hidden = inputs @ self.kernel
    if hiddens is not None:
      for i in range(self.num_child):
        hidden += hiddens[:, i, :] @ self.kernels[i]
    return self.activation(hidden + self.bias)  # [B, H]


class NaryTreeLSTM(tf.keras.layers.Layer):
  def __init__(self, input_dim, hidden_dim, num_child):
    super().__init__()
    self.input_gate = SingleGate(input_dim, hidden_dim, num_child)
    self.output_gate = SingleGate(input_dim, hidden_dim, num_child)
    self.candidate = SingleGate(input_dim, hidden_dim, num_child, 'tanh')
    self.num_child = num_child
    self.activation = tf.keras.layers.Activation('tanh')
    self.forget_gates = [SingleGate(input_dim, hidden_dim, num_child)
                         for _ in range(num_child)]

  def call(self, inputs=None, cell_states=None, hiddens=None):
    # inputs: [B, D_in], hiddens: [B, N, H], cell_states: [B, N, H]
    input_gate = self.input_gate(inputs, hiddens)
    candidate_input = self.candidate(inputs, hiddens)
    cell_state = input_gate * candidate_input
    if hiddens is not None:
      # Leaf nodes have no children
      for i in range(self.num_child):
        forget_gate = self.forget_gates[i](inputs, hiddens)
        cell_state += forget_gate * cell_states[:, i, :]
    hidden = self.output_gate(inputs, hiddens) * self.activation(cell_state)
    return cell_state, hidden


class Tree(object):
  def __init__(self):
    self.num_child = 0
    self.word_id = -1  # Current word ID (leaf nodes only)
    self.children = list()  # List of subtrees of type `Tree`
    self.state = None  # Used to store TreeLSTM state for current node


def tree_forward(tree: Tree, model: NaryTreeLSTM,
                 embs: tf.keras.layers.Embedding):
  if tree.num_child == 0:  # Leaf node
    word_emb = embs(tf.convert_to_tensor([tree.word_id]))  # [B, D]
    tree.state = model(inputs=word_emb, cell_states=None, hiddens=None)
  else:  # Inner node
    for idx in range(tree.num_child):
      tree.children[idx].state = tree_forward(tree.children[idx], model, embs)
    # List of [B, 1, D] tensors -> [B, N, D] tensor
    cell_states = tf.concat([tf.expand_dims(tree.children[idx].state[0], axis=1)
                             for idx in range(tree.num_child)], axis=1)
    # List of [B, 1, D] tensors -> [B, N, D] tensor
    hiddens = tf.concat([tf.expand_dims(tree.children[idx].state[1], axis=1)
                         for idx in range(tree.num_child)], axis=1)
    tree.state = model(inputs=None, cell_states=cell_states, hiddens=hiddens)
  return tree.state[1]  # hidden


if __name__ == '__main__':
  batch_size, input_dim, hidden_dim, num_child = 4, 5, 10, 2
  gate = SingleGate(input_dim, hidden_dim, num_child)
  print(gate(inputs=tf.random.uniform((batch_size, input_dim)),
             hiddens=tf.random.normal((batch_size, num_child, hidden_dim))))
  tlstm = NaryTreeLSTM(input_dim, hidden_dim, num_child)
  print(tlstm(inputs=None,
              cell_states=tf.random.normal((batch_size, num_child, hidden_dim)),
              hiddens=tf.random.normal((batch_size, num_child, hidden_dim))))
  print(tlstm(inputs=tf.random.uniform((batch_size, input_dim)),
              cell_states=None,
              hiddens=None))
  tf.keras.layers.Embedding
