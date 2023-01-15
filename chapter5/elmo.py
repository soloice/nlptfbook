import numpy as np
import tensorflow as tf

# PART I: ELMo for pretraining
batch_size, time_steps = 16, 20
embedding_dim, hidden_size, vocab_size = 256, 256, 30000
word_ids = tf.keras.layers.Input(shape=(None,), dtype='int64')  # [B, T]
embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size,
                                            output_dim=embedding_dim,
                                            mask_zero=True)
fwd_lstm1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
fwd_lstm2 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
bwd_lstm1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True,
                                 go_backwards=True)
bwd_lstm2 = tf.keras.layers.LSTM(hidden_size, return_sequences=True,
                                 go_backwards=True)
fwd_fc = tf.keras.layers.Dense(vocab_size)
bwd_fc = tf.keras.layers.Dense(vocab_size)
word_embs = embedding_layer(word_ids)  # [B, T, E]
fwd_output1 = word_embs + fwd_lstm1(word_embs)  # [B, T, H]
fwd_output2 = fwd_output1 + fwd_lstm2(fwd_output1)  # [B, T, H]
fwd_output = fwd_fc(fwd_output2)  # [B, T, V]
bwd_output1 = word_embs + bwd_lstm1(word_embs)  # [B, T, H]
bwd_output2 = bwd_output1 + bwd_lstm2(bwd_output1)  # [B, T, H]
bwd_output = bwd_fc(bwd_output2)  # [B, T, V]

elmo_train = tf.keras.Model(inputs=[word_ids],
                            outputs=[fwd_output, bwd_output])
elmo_train.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam()

# Do some fake training
for i in range(3):
  with tf.GradientTape() as tape:
    x = np.random.randint(0, vocab_size, size=(batch_size, time_steps + 2))
    fwd_y, x, bwd_y = x[:, 2:], x[:, 1:-1], x[:, :-2]
    fwd_pred, bwd_pred = elmo_train(x)
    fwd_loss = loss_fn(y_true=fwd_y, y_pred=fwd_pred)
    bwd_loss = loss_fn(y_true=bwd_y, y_pred=bwd_pred)
    total_loss = tf.reduce_mean(fwd_loss + bwd_loss)
    print('total_loss:', total_loss)
  opt.minimize(total_loss, elmo_train.trainable_variables, tape=tape)

# PART II: ELMo for downstream tasks
word_ids = tf.keras.layers.Input(shape=(None,), dtype='int64')  # [B, T]
embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size,
                                            output_dim=embedding_dim,
                                            mask_zero=True)
fwd_lstm1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
fwd_lstm2 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
bwd_lstm1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True,
                                 go_backwards=True)
bwd_lstm2 = tf.keras.layers.LSTM(hidden_size, return_sequences=True,
                                 go_backwards=True)
fwd_fc = tf.keras.layers.Dense(vocab_size)
bwd_fc = tf.keras.layers.Dense(vocab_size)
word_embs = embedding_layer(word_ids)  # [B, T, E]
fwd_output1 = word_embs + fwd_lstm1(word_embs)  # [B, T, H]
fwd_output2 = fwd_output1 + fwd_lstm2(fwd_output1)  # [B, T, H]
bwd_output1 = word_embs + bwd_lstm1(word_embs)  # [B, T, H]
bwd_output2 = bwd_output1 + bwd_lstm2(bwd_output1)  # [B, T, H]

all_outputs = [word_embs, fwd_output1, fwd_output2, bwd_output1, bwd_output2]
# Skiping this stack also works. In this case, 5 outputs are returned.
all_outputs = tf.stack(all_outputs, axis=2)  # [B, T, 5, D]
elmo_feature = tf.keras.Model(inputs=word_ids, outputs=all_outputs)
print('elmo feature:')
elmo_feature.summary()


class ELMoFeature(tf.keras.layers.Layer):
  def __init__(self, elmo_feature_model):
    super(ELMoFeature, self).__init__()
    self.elmo_feature_model = elmo_feature_model
    self.elmo_feature_model.trainable = False
    init_values = np.ones(5, dtype=np.float32).reshape((1, 1, 5, 1)) / 5
    w_init = tf.constant_initializer(init_values)
    self.elmo_weight = self.add_weight(name="elmo_weight",
                                       shape=init_values.shape,
                                       initializer=w_init,
                                       trainable=True)

  def call(self, inputs):
    # inputs: word ids of shape [B, T]
    all_elmo_embeddings = self.elmo_feature_model(inputs)  # [B, T, 5, H]
    weightd_sum = self.elmo_weight * all_elmo_embeddings  # [B, T, 5, H]
    final_elmo_embedding = tf.reduce_sum(weightd_sum, axis=2)  # [B, T, H]
    return final_elmo_embedding


# Use ELMo feature to replace the Embedding layer in the downstream model!
downstream_model = tf.keras.Sequential([ELMoFeature(elmo_feature),
                                        tf.keras.layers.GRU(512),
                                        tf.keras.layers.Dense(1)])
downstream_model.build(input_shape=(batch_size, time_steps))
print('downstream model:')
downstream_model.summary()

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
downstream_model.compile(loss=loss_fn, optimizer='adam')

# Do some fake training
for i in range(3):
  with tf.GradientTape() as tape:
    x = np.random.randint(0, vocab_size, size=(batch_size, time_steps))
    y = np.random.randint(0, 2, size=(batch_size,))
    downstream_model.fit(x, y)
    # Only elmo_weights changed in the ELMo layer
    print(downstream_model.layers[0].weights)
    # All other parameters in the backbone changed
    print(downstream_model.layers[1].weights)
