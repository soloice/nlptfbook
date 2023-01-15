from chapter5.ner.config import Config
from chapter5.ner.data_util import CoNLLDataset, minibatches, pad_sequences
import numpy as np
import tensorflow as tf


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
                            # input_length=max_seq_len,
                            mask_zero=True),  # [B, T, E]
  build_bigru_layer(config.hidden_size, config.dropout),  # [B, T, 2*D]
  build_bigru_layer(config.hidden_size, config.dropout),  # [B, T, 2*D]
  tf.keras.layers.Dense(units=64, activation='relu'),  # [B, D]
  tf.keras.layers.Dense(units=config.ntags)  # [B, V]
])
model.summary(line_length=120)

x = np.random.randint(1, 20000, size=(20, 64))
print(x.shape)
print(model.layers[0](x).shape)
print(model(x).shape)
