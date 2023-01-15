import numpy as np
import tensorflow as tf

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


def create_masks(seq, causal=False):
  # seq: [batch_size, seq_len] tensor of input word ids
  # causal: boolean variable
  neg_inf = -1e9
  if causal:
    seq_len = tf.shape(seq)[1]
    # mask looks like [[0, 1], [0, 0]] and is of shape [seq_len, seq_len]
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask * neg_inf  # [seq_len, seq_len]
  else:
    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
    seq = tf.cast(tf.math.equal(seq, PAD_ID), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :] * neg_inf


class ScaledDotProductAttention(tf.keras.layers.Layer):
  def __init__(self, d_k):
    super(ScaledDotProductAttention, self).__init__()
    self.scale = tf.sqrt(tf.cast(d_k, dtype=tf.float32))

  def call(self, query, key, value, mask=None):
    # query: [..., N, D_k], key: [..., M, D_k], value: [..., M, D_v]
    # mask: [..., N, M]
    # where N is # of query vectors, M is # of key/value vectors,
    #   D_k is dimension for key & query vectors,
    #   D_v is dimension for value vectors. In most cases, D_k=D_v.
    scores = tf.matmul(query, key, transpose_b=True)  # [..., N, M]
    if mask is not None:
      scores += mask  # [..., N, M]
    p_attn = tf.nn.softmax(scores / self.scale, axis=-1)  # [..., N, M]
    return tf.matmul(p_attn, value), p_attn  # [..., N, D_v], [..., N, M]


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, n_heads, d_model):
    super(MultiHeadAttention, self).__init__()
    assert d_model % n_heads == 0, "d_model must be a multiple of n_heads"
    self.d_k = d_model // n_heads
    self.h = n_heads
    self.W_q = tf.keras.layers.Dense(d_model)
    self.W_k = tf.keras.layers.Dense(d_model)
    self.W_v = tf.keras.layers.Dense(d_model)
    self.W_o = tf.keras.layers.Dense(d_model)
    self.scaled_dot_product = ScaledDotProductAttention(self.d_k)

  def call(self, query, key, value, mask=None):
    # query: [B, N, D], key: [B, M, D], value: [B, M, D], mask: [..., M, N]
    # where B is batch_size, N is # of query vectors,
    #   M is # of key/value vectors, D is the model size.

    batch_size = tf.shape(query)[0]
    # [B, N, D], [B, M, D], [B, M, D]
    query, key, value = self.W_q(query), self.W_k(key), self.W_v(value)

    # [B, N, D] -> [B, N, H, D_k=D/H] -> [B, H, N, D_k]
    # where H is # of attention heads.
    query = tf.transpose(tf.reshape(query, (batch_size, -1, self.h, self.d_k)),
                         [0, 2, 1, 3])
    # [B, M, D] -> [B, M, H, D_k=D/H] -> [B, H, M, D_k]
    key = tf.transpose(tf.reshape(key, (batch_size, -1, self.h, self.d_k)),
                       [0, 2, 1, 3])
    # [B, M, D] -> [B, M, H, D_v=D/H] -> [B, H, M, D_v]
    value = tf.transpose(tf.reshape(value, (batch_size, -1, self.h, self.d_k)),
                         [0, 2, 1, 3])

    # [B, H, N, D_v], [B, H, N, M]
    x, attn = self.scaled_dot_product(query, key, value, mask=mask)
    # [B, H, N, D_v] -> [B, N, H, D_v] -> [B, N, D]
    x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]),
                   (batch_size, -1, self.h * self.d_k))

    return self.W_o(x), attn  # [B, N, D], [B, H, N, M]


class FeedForwardNetwork(tf.keras.layers.Layer):
  # f(x) = Linear(ReLU(Linear(x)))
  def __init__(self, d_model, d_ff):
    super(FeedForwardNetwork, self).__init__()
    self.fc1 = tf.keras.layers.Dense(d_ff)
    self.fc2 = tf.keras.layers.Dense(d_model)

  def call(self, x):
    # Both input & output size are [batch_size, time_steps, model_dim]
    return self.fc2(tf.nn.relu(self.fc1(x)))


class LayerNormalization(tf.keras.layers.Layer):
  def __init__(self, axis=-1, eps=1e-5):
    super(LayerNormalization, self).__init__()
    self.axis = axis
    self.eps = eps

  def build(self, input_shape):
    dim = input_shape[-1]
    self.gamma = self.add_weight(name='gamma', shape=(dim,),
                                 initializer='ones', trainable=True)
    self.beta = self.add_weight(name='beta', shape=(dim,),
                                initializer='zeros', trainable=True)
    return super(LayerNormalization, self).build(input_shape)

  def call(self, x, **kwargs):
    # x: [..., D]
    mean = tf.reduce_mean(x, axis=self.axis, keepdims=True)  # [..., 1]
    variance = tf.math.reduce_std(x, axis=self.axis, keepdims=True)  # [..., 1]
    normalized_inputs = (x - mean) / tf.sqrt(variance + self.eps)  # [..., D]
    return self.gamma * normalized_inputs + self.beta  # [..., D]


class EncoderBlock(tf.keras.layers.Layer):
  def __init__(self, heads, d_model, d_ff, dropout):
    super(EncoderBlock, self).__init__()
    self.self_attn = MultiHeadAttention(heads, d_model)
    self.ffn = FeedForwardNetwork(d_model, d_ff)
    self.ln1 = LayerNormalization()
    self.ln2 = LayerNormalization()
    self.drop1 = tf.keras.layers.Dropout(dropout)
    self.drop2 = tf.keras.layers.Dropout(dropout)

  def call(self, x, mask=None, training=False):
    # x: [B, T, D], where B is batch_size, T is the length of source sequence,
    #   D is model dimension

    # Self-attention sublayer
    input_x = x  # [B, T, D]
    x, attn = self.self_attn(x, x, x, mask=mask)  # [B, T, D], [B, H, T, T]
    attn_output = self.drop1(x, training=training)  # [B, T, D]
    x = self.ln1(input_x + attn_output)  # [B, T, D]

    # Feed forward sublayer
    ffn_output = self.drop2(self.ffn(x), training=training)  # [B, T, D]
    x = self.ln2(x + self.drop2(ffn_output))  # [B, T, D]
    return x, attn  # [B, T, D], [B, H, T, T]


class DecoderBlock(tf.keras.layers.Layer):
  def __init__(self, heads, d_model, d_ff, dropout):
    super(DecoderBlock, self).__init__()
    self.cross_attn = MultiHeadAttention(heads, d_model)
    self.causal_self_attn = MultiHeadAttention(heads, d_model)

    self.ffn = FeedForwardNetwork(d_model, d_ff)

    self.ln1 = LayerNormalization()
    self.ln2 = LayerNormalization()
    self.ln3 = LayerNormalization()

    self.drop1 = tf.keras.layers.Dropout(dropout)
    self.drop2 = tf.keras.layers.Dropout(dropout)
    self.drop3 = tf.keras.layers.Dropout(dropout)

  def call(self, x, enc_output, causal_mask, padding_mask, training=False):
    # x: [B, U, D] is the decoder input
    # enc_output: [B, T, D] is the encoder last-layer output,
    # causal_mask: [..., U, U] is used in decoder self-attention,
    # padding_mask: [..., U, T] is used in encoder-decoder cross-attention.
    # where B is batch_size, T/U is the length of source/target sequence,
    #   D is model dimension

    # Self-attention sublayer
    # [B, U, D], [B, H, U, U]
    attn_out, attn1 = self.causal_self_attn(x, x, x, causal_mask)
    x = self.ln1(x + self.drop1(attn_out, training=training))  # [B, U, D]

    # Cross-attention sublayer
    # [B, U, D], [B, H, U, T]
    attn_out, attn2 = self.cross_attn(x, enc_output, enc_output, padding_mask)
    x = self.ln2(x + self.drop2(attn_out, training=training))  # [B, U, D]

    # Feed forward sublayer
    x = self.ln3(x + self.drop3(self.ffn(x), training=training))  # [B, U, D]
    return x, attn1, attn2


class TiedEmbeddingDense(tf.keras.layers.Layer):
  def __init__(self, tied_to: tf.keras.layers.Layer, **kwargs):
    super(TiedEmbeddingDense, self).__init__(**kwargs)
    self.tied_to = tied_to

  def build(self, input_shape):
    # kernel: [vocab_size, model_dim]
    self.kernel = self.tied_to.weights[0]
    # bias: [vocab_size]
    self.bias = self.add_weight(name="bias", shape=[self.kernel.shape[0]])
    self.built = True

  def call(self, inputs):
    # inputs: [B, T, D], self.kernel: [V, D], self.bias: [V]
    # where B=batch_size, T=time_steps, D=d_model, V=vocab_size
    output = inputs @ tf.transpose(self.kernel) + self.bias  # [B, T, V]
    return output

  def get_config(self):
    return dict(list(super(TiedEmbeddingDense, self).get_config().items()))


class TransformerEmbedding(tf.keras.layers.Layer):
  def __init__(self, d_model, vocab_size, dropout, max_len=5000):
    super(TransformerEmbedding, self).__init__()
    self.emb = tf.keras.layers.Embedding(vocab_size, d_model)
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.drop = tf.keras.layers.Dropout(dropout)

    half_dim = d_model // 2
    pe = np.zeros((d_model, max_len))  # [D, T_max]
    pos = np.arange(max_len)  # [T_max]
    freq = 10000 ** (2 * np.arange(half_dim) / d_model)  # [D//2]
    pos_freq = pos.reshape((1, -1)) / freq.reshape((-1, 1))  # [D//2, T_max]
    pe[:d_model // 2, :] = np.sin(pos_freq)
    pe[d_model // 2:, :] = np.cos(pos_freq)
    self.pe = tf.constant(pe.T, dtype=tf.float32)  # [T_max, D]

  def build(self, input_shape):
    self.emb.build(input_shape)
    self.built = True

  def call(self, x, training=False):
    # x: [B, T] tensor of word ids, where B is batch_size, T is input length.
    # shape == (batch_size, max_len, d_model)
    x = self.emb(x) * tf.sqrt(tf.cast(self.d_model, tf.float32))  # [B, T, D]
    time_steps = x.get_shape()[1]
    return self.drop(x + self.pe[:time_steps], training=training)  # [B, T, D]


class Transformer(tf.keras.Model):
  def __init__(self, nlayers_enc, nlayers_dec, d_model, n_heads,
               d_ff, vocab_size, max_len=5000, dropout=0.1):
    super(Transformer, self).__init__()
    self.nlayers_enc = nlayers_enc
    self.nlayers_dec = nlayers_dec
    self.emb_layer = TransformerEmbedding(d_model, vocab_size,
                                          dropout=dropout, max_len=max_len)
    self.enc_blocks = [EncoderBlock(n_heads, d_model, d_ff, dropout)
                       for _ in range(nlayers_enc)]
    self.dec_blocks = [DecoderBlock(n_heads, d_model, d_ff, dropout)
                       for _ in range(nlayers_dec)]
    self.linear = TiedEmbeddingDense(tied_to=self.emb_layer)

  def call(self, src_ids, tgt_ids, training=False):
    # src_ids: [B, T] tensor of source word ids
    # tgt_ids: [B, U] tensor of target word ids
    # where B=batch size, T=source length, U=target length,
    #       D=model size, H=number of attention heads

    enc_x = self.emb_layer(src_ids, training=training)  # [B, T, D]
    # padding_mask.shape == [B, 1, 1, T]. It will be broadcasted
    #   into [B, H, T, T] in encoder, and into [B, H, U, T] in decoder.
    padding_mask = create_masks(src_ids, causal=False)  # [B, 1, 1, T]
    for i in range(self.nlayers_enc):
      # [B, T, D], [B, H, T, T]
      enc_x, enc_attn = self.enc_blocks[i](enc_x, padding_mask,
                                           training=training)

    dec_x = self.emb_layer(tgt_ids, training=training)  # [B, U, D]
    # causal_mask.shape == [U, U].
    #   It will be broadcasted into [B, H, U, U] later.
    causal_mask = create_masks(tgt_ids, causal=True)
    for i in range(self.nlayers_dec):
      # [B, U, D], [B, H, U, U], [B, H, U, T]
      dec_x, dec_attn1, dec_attn2 = self.dec_blocks[i](dec_x, enc_x,
                                                       causal_mask,
                                                       padding_mask,
                                                       training=training)
    # [B, U, V], [B, H, T, T], [B, H, U, U], [B, H, U, T]
    return self.linear(dec_x), enc_attn, dec_attn1, dec_attn2


batch_size, enc_len, dec_len = 20, 40, 50
vocab_size, d_model, nlayers_enc, nlayers_dec = 10000, 256, 2, 2
n_heads, d_ff = 8, d_model * 4

model = Transformer(nlayers_enc, nlayers_dec, d_model, n_heads,
                    d_ff, vocab_size)


def generate_data():
  # Assume PAD_ID=0, BOS_ID=1, EOS_ID=2
  # Step I: Generate word ids in range [1, vocab_size - 1]
  src_ids = np.random.randint(vocab_size - 1, size=[batch_size, enc_len]) + 1
  tgt_ids = np.random.randint(vocab_size - 1, size=[batch_size, dec_len]) + 1

  # Step II: Generate sequence lengths and deal with special tokens
  # Here sequence length include <BOS> token, but not <EOS> token
  src_lens = np.random.randint(low=enc_len//2, high=enc_len - 1,
                               size=[batch_size])
  tgt_lens = np.random.randint(low=dec_len // 2, high=dec_len - 1,
                               size=[batch_size])
  mask = np.zeros(shape=[batch_size, dec_len - 1], dtype=np.float32)
  for i in range(batch_size):
    src_ids[i, src_lens[i]:] = PAD_ID
    tgt_ids[i, 0] = BOS_ID
    tgt_ids[i, tgt_lens[i]] = EOS_ID
    tgt_ids[i, tgt_lens[i] + 1:] = PAD_ID
    mask[i, :tgt_lens[i]] = 1.0
  # tgt_ids[:, :-1] & tgt_ids[:, 1:] are for target input & output respectively
  return src_ids, tgt_ids[:, :-1], tgt_ids[:, 1:], src_lens, tgt_lens, mask


src_ids, tgt_ids_in, tgt_ids_out, *_ = generate_data()
y = model(src_ids, tgt_ids_in)
model.summary()

optim = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)

for i in range(5):
  src_ids, tgt_ids_in, tgt_ids_out, src_lens, tgt_lens, mask = generate_data()
  with tf.GradientTape() as tape:
    logits, _, _, _ = model(src_ids, tgt_ids_in)
    ce = tf.keras.losses.sparse_categorical_crossentropy(tgt_ids_out,
                                                         logits,
                                                         from_logits=True)
    loss = tf.reduce_sum(ce * mask) / tf.reduce_sum(mask)
    print(loss)
  optim.minimize(loss, model.trainable_variables, tape=tape)
  # Check tied weight
  print(tf.reduce_sum(model.emb_layer.weights[0] - model.linear.weights[0]))
