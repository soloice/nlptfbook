import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size, max_seq_len = 30000, 300
embedding_dim, hidden1, hidden2 = 20, 32, 64
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
word2id = imdb.get_word_index()
for w in word2id:
  word2id[w] += 3
word2id['<pad>'] = 0
word2id['<sos>'] = 1
word2id['<unk>'] = 2
id2word = {i: w for w, i in word2id.items()}

print('total vocab size:', len(word2id))  # 88584
# Both train & test data have 25000 sentences.
print(x_train.shape, x_test.shape)
# Show an example sentence
print('x[0] ids:', x_train[0])
print('x[0] words:', ' '.join([id2word[i] for i in x_train[0]]))
# The longest training sentence has length 2494
seq_lens = [len(x) for x in x_train]
print(len([l for l in seq_lens if l <= max_seq_len]))

fwd1 = tf.keras.layers.GRU(units=hidden1, return_sequences=True,
                           time_major=False, go_backwards=False,
                           dropout=0.1, recurrent_dropout=0.1)
bwd1 = tf.keras.layers.GRU(units=hidden1, return_sequences=True,
                           time_major=False, go_backwards=True,
                           dropout=0.1, recurrent_dropout=0.1)
layer1 = tf.keras.layers.Bidirectional(layer=fwd1, backward_layer=bwd1,
                                       merge_mode="concat")
fwd2 = tf.keras.layers.GRU(units=hidden2, return_sequences=False,
                           time_major=False, go_backwards=False,
                           dropout=0.1, recurrent_dropout=0.1)
bwd2 = tf.keras.layers.GRU(units=hidden2, return_sequences=False,
                           time_major=False, go_backwards=True,
                           dropout=0.1, recurrent_dropout=0.1)
layer2 = tf.keras.layers.Bidirectional(layer=fwd2, backward_layer=bwd2,
                                       merge_mode="concat")

model = tf.keras.Sequential([
  tf.keras.layers.Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            input_length=max_seq_len,
                            mask_zero=True),  # [B, T, E]
  layer1,  # [B, T, 2*D1]
  layer2,  # [B, 2*D2]
  tf.keras.layers.Dense(units=64, activation='relu'),  # [B, D]
  tf.keras.layers.Dense(units=1)  # [B, 1]
])

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(loss=loss_fn, metrics=['accuracy'], optimizer='adam')
model.summary()

paded_x_train = pad_sequences(sequences=x_train, maxlen=max_seq_len,
                              padding="post", truncating="pre",
                              value=word2id['<pad>'])
paded_x_test = pad_sequences(sequences=x_test, maxlen=max_seq_len,
                             padding="post", truncating="pre",
                             value=word2id['<pad>'])
model.fit(x=paded_x_train, y=y_train, batch_size=32,
          epochs=3, validation_split=0.2, shuffle=True)
model.evaluate(paded_x_test, y_test)
