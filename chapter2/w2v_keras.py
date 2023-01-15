import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
import tensorflow.keras.preprocessing.sequence as keras_seq
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

SEED = 42
num_ns = 4
embedding_dim = 50
AUTOTUNE = tf.data.AUTOTUNE

path_to_corpus = '../datasets/ptb/ptb.train.txt'
text_ds = tf.data.TextLineDataset(path_to_corpus). \
  filter(lambda x: tf.cast(tf.strings.length(x), bool))

# Map & pad strings to integer (word ID) sequences
vectorize_layer = TextVectorization(
  standardize=None,
  output_mode='int',
  output_sequence_length=None)

# Read one epoch of training data to get the vocab
vectorize_layer.adapt(text_ds.batch(1024))
inverse_vocab = vectorize_layer.get_vocabulary()
vocab_size = len(inverse_vocab)
print(len(inverse_vocab), inverse_vocab[:20])

# Vectorize the data in text_ds.
text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE). \
  map(vectorize_layer).unbatch()
sequences = list(text_vector_ds.as_numpy_iterator())
print(len(sequences))
print(f"{sequences[0]} => {[inverse_vocab[i] for i in sequences[0]]}")


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  targets, contexts, labels = [], [], []
  sampling_table = keras_seq.make_sampling_table(vocab_size)

  for sequence in tqdm.tqdm(sequences):
    # list of (target_id, context_id) pairs
    positive_skip_grams, _ = keras_seq.skipgrams(
      sequence,
      vocabulary_size=vocab_size,
      sampling_table=sampling_table,
      window_size=window_size,
      negative_samples=0)

    # Generate negative samples one by one
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
        tf.constant([context_word], dtype="int64"), 1)
      # [num_ns]
      negative_sampling_candidates, _, _ = \
        tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # [num_ns, 1]
      negative_sampling_candidates = tf.expand_dims(
        negative_sampling_candidates, 1)
      # [1+num_ns, 1]
      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1.] + [0.] * num_ns, dtype="float32")

      # Finish sampling process for a target word
      targets.append(tf.constant([target_word]))
      contexts.append(context)
      labels.append(label)
  return targets, contexts, labels


targets, contexts, labels = generate_training_data(
  sequences=sequences,
  window_size=2,
  num_ns=num_ns,
  vocab_size=vocab_size,
  seed=SEED)
print(len(targets), len(contexts), len(labels))

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
# <PrefetchDataset shapes: (((B, 1), (B, 1+num_ns, 1)), (B, 1+num_ns))>
print(dataset)


class Word2Vec(Model):
  def __init__(self, vocab_size, embedding_dim, num_ns):
    super(Word2Vec, self).__init__()
    self.target_embedding = Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns + 1)
    self.dots = Dot(axes=(3, 2))
    self.flatten = Flatten()

  def call(self, pair):
    target_word, context_word = pair  # [B, 1], [B, 1+num_ns, 1]
    word_emb = self.target_embedding(target_word)  # [B, 1, D]
    context_emb = self.context_embedding(context_word)  # [B, 1+num_ns, 1, D]
    dots = self.dots([context_emb, word_emb])  # [B, 1+num_ns, 1, 1]
    return self.flatten(dots)  # [B, 1+num_ns]


def custom_loss(y_true, y_pred):
  return tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)


w2v = Word2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim,
               num_ns=num_ns)
w2v.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
w2v.fit(dataset, epochs=50)
