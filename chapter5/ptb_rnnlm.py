import numpy as np
import tensorflow as tf

batch_size, seq_len, num_epochs = 32, 50, 3
train_corpus = '../datasets/ptb/ptb.char.train.txt'
val_corpus = '../datasets/ptb/ptb.char.valid.txt'


def read_dataset(path_to_corpus, vocab=None):
  with open(path_to_corpus) as f:
    corpus = [l.strip(' ') for l in f.readlines() if len(l.strip()) > 0]
    text = ' '.join(corpus).split(' ')
    # number of characters in a training batch
    batch_chars = batch_size * seq_len
    effective_length = (len(text) - 1) // batch_chars * batch_chars + 1
    x, y = text[:effective_length - 1], text[1: effective_length]
    print(text[:1000])
    print(len(text))
    if vocab is None:
      vocab = sorted(set(text[:effective_length]))
      vocab = {char: i for i, char in enumerate(vocab)}
      print('vocab:', vocab)

    x_ids = np.array([vocab[char] for char in x]).reshape(
      [batch_size, -1, seq_len])
    y_ids = np.array([vocab[char] for char in y]).reshape(
      [batch_size, -1, seq_len])
    return x_ids, y_ids, vocab


def loss_fn(labels, logits):
  return tf.reduce_mean(
    tf.keras.losses.sparse_categorical_crossentropy(labels, logits,
                                                    from_logits=True))


train_x, train_y, char_vocab = read_dataset(train_corpus)
id2char = {i: char for char, i in char_vocab.items()}
vocab_size, embedding_dim, hidden = len(char_vocab), 20, 64
valid_x, valid_y, _ = read_dataset(val_corpus, char_vocab)
num_train_batches, num_valid_batches = train_x.shape[1], valid_x.shape[1]
print('num_batches per epoch:', num_train_batches, num_valid_batches)


def get_rnnlm(model_batch_size):
  return tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size,
                              output_dim=embedding_dim,
                              batch_size=model_batch_size),  # [B, T, E]
    tf.keras.layers.LSTM(hidden, stateful=True, return_sequences=True),
    tf.keras.layers.Dense(vocab_size)
  ])


def sample(language_model, start_text='\n', length=80, temperature=1.0):
  start_ids = [char_vocab[char] for char in start_text]
  # start_ids: [batch_size=1, time_steps]
  start_ids = np.array(start_ids).reshape(1, -1)
  language_model.reset_states()
  generated_sequence = start_text
  for i in range(length):
    # output: [batch_size=1, time_steps, vocab_size]
    output = language_model(start_ids)
    output = output / temperature
    # Sample from multinomial distribution.
    #   Use fp64 to deal with summation overflow
    probs = tf.nn.softmax(output[0, -1, :]).numpy().astype('float64')
    probs /= probs.sum()
    sampled_char_id = np.argmax(np.random.multinomial(n=1, pvals=probs))
    # print('sampled char:', sampled_char_id)
    generated_sequence += id2char[sampled_char_id]
    if generated_sequence[-1] == '\n':
      break
    # [batch_size=1, time_steps=1]
    start_ids = tf.reshape(sampled_char_id, [1, -1])
  return generated_sequence


def sample_sentences():
  sampling_model = get_rnnlm(model_batch_size=1)
  sampling_model.set_weights(model.get_weights())
  print(sample(sampling_model, start_text='this_is'))
  print(sample(sampling_model, start_text='the_move_was_made'))


model = get_rnnlm(batch_size)
print(model.summary())
optim = tf.keras.optimizers.Adamax(learning_rate=0.1, clipnorm=1.0)


def show_batch_text(batch):
  # batch: np.ndarray of shape [batch_size, seq_len]
  return [''.join([id2char[i] for i in row]) for row in batch]


for i in range(3):
  batch_x, batch_y = train_x[:, i, :], train_y[:, i, :]
  print('step', i, 'x:', show_batch_text(batch_x))
  print('step', i, 'y:', show_batch_text(batch_y))

for e in range(num_epochs):
  # Training
  print('Start epoch:', e)
  model.reset_states()
  for b in range(num_train_batches):
    batch_x, batch_y = train_x[:, b, :], train_y[:, b, :]
    with tf.GradientTape() as tape:
      pred_y = model(batch_x, training=True)
      loss = loss_fn(labels=batch_y, logits=pred_y)
    optim.minimize(loss, model.trainable_variables, tape=tape)
    if b % 100 == 0:
      print('step', b, 'loss:', loss.numpy(), 'ppl:', np.exp(loss.numpy()))
    if b % 500 == 0:
      sample_sentences()
  print('Finish Epoch', e)
  sample_sentences()
  # Validation
  model.reset_states()
  valid_loss = 0.0
  for b in range(num_valid_batches):
    batch_x, batch_y = valid_x[:, b, :], valid_y[:, b, :]
    pred_y = model(batch_x, training=False)
    valid_loss += loss_fn(labels=batch_y, logits=pred_y).numpy()
  valid_loss /= num_valid_batches
  print('Valid perplexity:', np.exp(valid_loss))
