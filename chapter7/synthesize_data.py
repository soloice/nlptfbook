import random
import numpy as np


class Config:
  max_len = 20
  hidden_size = 64
  batch_size = 32
  inf_batch_size = 3
  embed_dim = 30
  attn_size = 64
  beam_width = 2
  vocab_size = 12  # 0, 1, ..., 9, 10, 11
  BOS = 10
  EOS = 11
  PAD = 0


config = Config()


def generate_data(num_samples, copy_sequence=False):
  # Generate a batch of data with <BOS> & <EOS> added
  odd_list = [1, 3, 5, 7, 9] * config.max_len
  even_list = [2, 4, 6, 8] * config.max_len
  num_odds = np.random.randint(low=1, high=config.max_len // 2,
                               size=num_samples)
  num_evens = np.random.randint(low=config.max_len // 5,
                                high=config.max_len // 2,
                                size=num_samples)
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
                     [config.PAD] * (batch_max_length_x - len(sample_x))]
    sample_y = np.r_[sample_y, [config.EOS],
                     [config.PAD] * (batch_max_length_y - len(sample_y) - 1)]

    batch_x.append(sample_x)
    batch_y_out.append(sample_y)

  batch_x = np.array(batch_x, dtype=np.int32)
  batch_y_out = np.array(batch_y_out, dtype=np.int32)
  bos_tokens = np.zeros([num_samples, 1], dtype=np.int32) + config.BOS
  batch_y_in = np.c_[bos_tokens, batch_y_out[:, :-1]]
  return batch_x, batch_y_in, batch_y_out, batch_len_x, batch_len_y

