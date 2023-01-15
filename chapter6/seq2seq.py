import numpy as np
import random
import tensorflow as tf

max_len = 20
hidden_size = 64
batch_size = 32
embedding_dim = 30
vocab_size = 12  # 0, 1, ..., 9, 10, 11
BOS, EOS, PAD = 10, 11, 0

input_x = tf.keras.Input(batch_input_shape=(None, None))  # [B, T]
input_y = tf.keras.Input(batch_input_shape=(None, None))  # [B, U]
embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                      output_dim=embedding_dim,
                                      input_length=max_len,
                                      mask_zero=True)
encoder = tf.keras.layers.GRU(units=hidden_size, time_major=False,
                              return_state=True,
                              recurrent_dropout=0.1)
decoder = tf.keras.layers.GRU(units=hidden_size, time_major=False,
                              return_sequences=True, return_state=True,
                              recurrent_dropout=0.1)
fc = tf.keras.layers.Dense(units=vocab_size)
input_x_embeddings = embedding(input_x)  # [B, T, E]
input_y_embeddings = embedding(input_y)  # [B, U, E]
_, encoder_state = encoder(input_x_embeddings)  # [B, D], [B, D]
decoder_out, _ = decoder(inputs=input_y_embeddings,
                         initial_state=encoder_state)  # [B, U, D], [B, D]
logits = fc(decoder_out)  # [B, U, V]

model = tf.keras.Model(inputs=[input_x, input_y], outputs=[logits])
model.summary(line_length=120)
optim = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

odd_list, even_list = [1, 3, 5, 7, 9] * max_len, [2, 4, 6, 8] * max_len


def generate_batch_data(num_samples, copy_sequence=False):
  # Generate a batch of data with <BOS> & <EOS> added
  num_odds = np.random.randint(low=1, high=max_len // 2, size=num_samples)
  num_evens = np.random.randint(low=1, high=max_len // 2, size=num_samples)
  batch_len_x = num_odds + num_evens
  if copy_sequence:
    batch_len_y = num_evens * 2 + 1  # append <EOS> or prepend <BOS>
  else:
    batch_len_y = num_evens + 1  # append <EOS> or prepend <BOS>

  batch_max_length_x = np.max(batch_len_x)
  batch_max_length_y = np.max(batch_len_y)

  batch_x, batch_y_out = [], []
  for i in range(num_samples):
    odds = random.sample(odd_list, num_odds[i])
    evens = random.sample(even_list, num_evens[i])
    sample_x = odds + evens
    np.random.shuffle(sample_x)

    sample_y = list(filter(lambda x: x % 2 == 0, sample_x))
    if copy_sequence:
      sample_y += sample_y
    sample_x = np.r_[sample_x,
                     [PAD] * (batch_max_length_x - len(sample_x))]
    sample_y = np.r_[sample_y, [EOS],
                     [PAD] * (batch_max_length_y - len(sample_y) - 1)]

    batch_x.append(sample_x)
    batch_y_out.append(sample_y)

  batch_x = np.array(batch_x, dtype=np.int32)
  batch_y_out = np.array(batch_y_out, dtype=np.int32)
  bos_tokens = np.zeros([num_samples, 1], dtype=np.int32) + BOS
  batch_y_in = np.c_[bos_tokens, batch_y_out[:, :-1]]

  return batch_x, batch_y_in, batch_y_out, batch_len_x, batch_len_y


def loss_fn(labels, logits, masks):
  # labels: [B, U], logits: [B, U, V], masks: [B, U]
  # ce: [B, U]
  ce = tf.keras.losses.sparse_categorical_crossentropy(labels, logits,
                                                       from_logits=True)
  return tf.reduce_sum(ce * masks) / tf.reduce_sum(masks)


@tf.function(experimental_relax_shapes=True)
def train_step(x, y_in, y_out):
  mask = tf.cast(y_out > PAD, dtype=tf.float32)
  with tf.GradientTape() as tape:
    pred_logits = model([x, y_in], training=True)
    loss = loss_fn(labels=y_out, logits=pred_logits, masks=mask)
  optim.minimize(loss, model.trainable_variables, tape=tape)
  return loss


def greedy_decode(x):
  _, state = encoder(embedding(x))
  curr_batch_size = int(x.shape[0])
  step_inputs = tf.tile([[BOS]], [curr_batch_size, 1])  # [B, 1]
  seq_stopped = [False for _ in range(curr_batch_size)]
  decoding_results = [[] for _ in range(curr_batch_size)]
  for t in range(max_len * 2):
    # [B, 1, D], [B, D]
    step_out, state = decoder(embedding(step_inputs), state)
    step_logits = fc(step_out[:, 0, :])  # [B, V]
    most_probable_ids = tf.argmax(step_logits, axis=1)  # [B]
    for b in range(curr_batch_size):
      seq_stopped[b] = seq_stopped[b] or most_probable_ids[b] == EOS
      if not seq_stopped[b]:
        decoding_results[b].append(most_probable_ids[b])
    step_inputs = tf.reshape(most_probable_ids, [-1, 1])  # [B, 1]
    if all(seq_stopped):
      break
  return [tf.stack(res) for res in decoding_results]


for step in range(5000):
  x, y_in, y_out, len_x, len_y = generate_batch_data(batch_size)
  loss = train_step(x, y_in, y_out)  # Train with teacher forcing
  if step % 100 == 0:
    print('step', step, 'loss', loss)
    x, _1, _2, _3, _4 = generate_batch_data(num_samples=3)
    print('greedy:', x, greedy_decode(x))
