from chapter5.ner.config import Config
from chapter5.ner.data_util import CoNLLDataset
import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def build_bigru_layer(hidden_size, dropout, merge_mode="concat"):
  fwd = tf.keras.layers.GRU(units=hidden_size, return_sequences=True,
                            time_major=False, go_backwards=False,
                            dropout=dropout, recurrent_dropout=dropout)
  bwd = tf.keras.layers.GRU(units=hidden_size, return_sequences=True,
                            time_major=False, go_backwards=True,
                            dropout=dropout, recurrent_dropout=dropout)
  layer = tf.keras.layers.Bidirectional(layer=fwd, backward_layer=bwd,
                                        merge_mode=merge_mode)
  return layer


# build model
config = Config(load=True)
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(input_dim=config.nwords,
                            output_dim=config.dim_word,
                            mask_zero=True, trainable=True),  # [B, T, E]
  build_bigru_layer(config.hidden_size, config.dropout),  # [B, T, 2*D]
  tf.keras.layers.Dense(units=config.ntags)  # [B, T, V]
])
# Initialize with glove embeddings
with np.load(config.filename_trimmed) as data:
  model.layers[0].set_weights([data['embeddings']])
model.summary(line_length=120)
optim = tf.keras.optimizers.Adam(learning_rate=config.lr,
                                 clipnorm=config.clip)
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
valid_acc1 = tf.keras.metrics.SparseCategoricalAccuracy()
valid_acc2 = tf.keras.metrics.Accuracy()


# create datasets
def get_tf_dataset(data, shuffle=False):
  all_words, all_tags, lengths = [], [], []
  for words, tags in data:
    all_words.append(words)
    all_tags.append(tags)
  tensors = (tf.ragged.constant(all_words), tf.ragged.constant(all_tags))
  dataset = tf.data.Dataset.from_tensor_slices(tensors)
  dataset = dataset.map(lambda x, y: (x, y, tf.ones_like(x, dtype=tf.float32)))
  dataset = dataset.padded_batch(config.batch_size)
  if shuffle:
    dataset.shuffle(buffer_size=1000)
  return dataset


# len(train_data) = 14041, len(dev_data) = 3250
train_data = CoNLLDataset(config.filename_train, config.processing_word,
                          config.processing_tag, config.max_iter)
dev_data = CoNLLDataset(config.filename_dev, config.processing_word,
                        config.processing_tag, config.max_iter)
train_data = get_tf_dataset(train_data, shuffle=True)
dev_data = get_tf_dataset(dev_data, shuffle=False)

rand_matrix = np.random.uniform(-0.1, 0.1, size=(config.ntags, config.ntags))
transition_matrix = tf.Variable(rand_matrix, trainable=True)
all_trainable_vars = model.trainable_variables + [transition_matrix]
print('all_trainable_vars:', all_trainable_vars)


def loss_fn(labels, logits, masks):
  lengths = tf.reduce_sum(masks, axis=1)
  ll, _ = tfa.text.crf_log_likelihood(logits, labels,
                                      tf.cast(lengths, dtype=tf.int32),
                                      transition_matrix)
  # Average negative log likelihood across tokens to make it more stable
  return tf.reduce_sum(-ll) / tf.reduce_sum(lengths)


@tf.function(experimental_relax_shapes=True)
def train_step(words, tags, masks):
  with tf.GradientTape() as tape:
    pred_logits = model(words, training=True)
    loss = loss_fn(labels=tags, logits=pred_logits, masks=masks)
  train_acc.update_state(tags, pred_logits, sample_weight=masks)
  optim.minimize(loss, all_trainable_vars, tape=tape)
  return loss


@tf.function(experimental_relax_shapes=True)
def valid_step(words, tags, masks):
  # masks: [B, T]
  pred_logits = model(words, training=False)
  valid_acc1.update_state(tags, pred_logits, sample_weight=masks)
  lengths = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)
  decoded_tags, _ = tfa.text.crf_decode(potentials=pred_logits,
                                        transition_params=transition_matrix,
                                        sequence_length=lengths)
  valid_acc2.update_state(tags, decoded_tags, sample_weight=masks)
  return pred_logits


# train model
for e in range(config.nepochs):
  # Training
  print('Start epoch:', e)
  for i, (words, tags, masks) in tqdm(enumerate(train_data)):
    loss = train_step(words, tags, masks)
    if i % 100 == 0:
      print('step', i, 'loss:', loss.numpy(),
            'ppl:', np.exp(loss.numpy()),
            'acc:', train_acc.result())
  train_acc.reset_states()
  print('Finish Epoch', e)

  # Validation
  valid_acc1.reset_states()
  valid_acc2.reset_states()
  for words, tags, masks in tqdm(dev_data):
    valid_step(words, tags, masks)
  print('Valid accuracy:', valid_acc1.result(), valid_acc2.result())
