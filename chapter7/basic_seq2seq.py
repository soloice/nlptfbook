import tensorflow as tf
import tensorflow_addons as tfa
from chapter7.synthesize_data import generate_data, config


class Encoder(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.emb = tf.keras.layers.Embedding(input_dim=config.vocab_size,
                                         output_dim=config.embed_dim,
                                         mask_zero=True)
    self.rnn = tf.keras.layers.GRU(config.hidden_size,
                                   return_sequences=True,
                                   return_state=True)

  def call(self, inputs):
    # inputs: [B, T]
    inputs_embedded = self.emb(inputs)  # [B, T, E]
    output, state = self.rnn(inputs_embedded)  # [B, T, D], [B, D]
    return output, state


class Decoder(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.emb = tf.keras.layers.Embedding(input_dim=config.vocab_size,
                                         output_dim=config.embed_dim,
                                         mask_zero=False)
    # Note this is GRUCell instead of GRU!!!
    self.rnn_cell = tf.keras.layers.GRUCell(config.hidden_size)
    self.fc = tf.keras.layers.Dense(config.vocab_size)
    self.sampler = tfa.seq2seq.TrainingSampler()
    self.decoder = tfa.seq2seq.BasicDecoder(cell=self.rnn_cell,
                                            sampler=self.sampler,
                                            output_layer=self.fc)

  def call(self, inputs, init_state=None):
    # x: [B, U]
    inputs_embedded = self.emb(inputs)  # [B, U, E]
    # [B, U, D], [B, D]
    # final_outputs: tfa.seq2seq.BasicDecoderOutput
    #     (rnn_output: [B, U, D], sample_id: [B, D])
    # final_states: [B, D]
    # final_lengths: [B]
    output, _, _ = self.decoder(inputs_embedded, initial_state=init_state)
    return output.rnn_output  # [B, U, D]


encoder, decoder = Encoder(), Decoder()
optim = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)


def loss_fn(labels, logits, masks):
  # labels: [B, U], logits: [B, U, V], masks: [B, U]
  # ce: [B, U]
  ce = tf.keras.losses.sparse_categorical_crossentropy(labels, logits,
                                                       from_logits=True)
  return tf.reduce_sum(ce * masks) / tf.reduce_sum(masks)


@tf.function(experimental_relax_shapes=True)
def train_step(x, y_in, y_out):
  # x: [B, T], y_in: [B, U], y_out: [B, U]
  mask = tf.cast(y_out > config.PAD, dtype=tf.float32)
  with tf.GradientTape() as tape:
    _, encoder_final_state = encoder(x, training=True)  # _, [B, D]
    logits = decoder(y_in, encoder_final_state, training=True)  # [B, U, D]
    loss = loss_fn(labels=y_out, logits=logits, masks=mask)
  all_params = encoder.trainable_variables + decoder.trainable_variables
  optim.minimize(loss, all_params, tape=tape)
  return loss


def greedy_decode(x):
  # x: [B, T]
  greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler(decoder.emb)
  max_iter = config.max_len * 2
  basic_decoder = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell,
                                           sampler=greedy_sampler,
                                           output_layer=decoder.fc)
  curr_batch_size = int(x.shape[0])
  _, encoder_final_state = encoder(x)  # _, [B, D]
  start_tokens = tf.tile([config.BOS], [curr_batch_size])  # [B]
  kwargs = {
    'initial_state': encoder_final_state,
    'start_tokens': start_tokens,
    'end_token': config.EOS}
  # final_outputs: tfa.seq2seq.BasicDecoderOutput
  #   (rnn_output: [B, U, D], sample_id: [B, D])
  # final_states: [B, D] tensor
  # final_lengths: [B] tensor
  outputs, _, lens = tfa.seq2seq.dynamic_decode(basic_decoder,
                                                maximum_iterations=max_iter,
                                                decoder_init_input=None,
                                                decoder_init_kwargs=kwargs)
  return outputs.sample_id.numpy(), lens.numpy()


def beam_search_decode(x, beam_width):
  # x: [B, T]
  beam_decoder = tfa.seq2seq.BeamSearchDecoder(decoder.rnn_cell,
                                               beam_width=beam_width,
                                               embedding_fn=decoder.emb,
                                               output_layer=decoder.fc)
  max_iter = config.max_len * 2
  curr_batch_size = int(x.shape[0])
  _, encoder_final_state = encoder(x)  # _, [B, D]
  # set up decoder_initial_state
  decoder_initial_state = tfa.seq2seq.tile_batch(encoder_final_state,
                                                 multiplier=beam_width)

  start_tokens = tf.tile([config.BOS], [curr_batch_size])  # [B]
  kwargs = {
    'initial_state': decoder_initial_state,
    'start_tokens': start_tokens,
    'end_token': config.EOS}
  # final_outputs: tfa.seq2seq.FinalBeamSearchDecoderOutput
  #   beam_search_decoder_output: tfa.seq2seq.BeamSearchDecoderOutput
  #     (scores: [B, U, W], predicted_ids: [B, U, W], parent_ids: [B, U, W]))
  #   predicted_ids: [B, U, W] tensor
  # final_states: tfa.seq2seq.BeamSearchDecoderState object
  # final_lengths: [B, W] tensor
  outputs, _, lens = tfa.seq2seq.dynamic_decode(beam_decoder,
                                                maximum_iterations=max_iter,
                                                decoder_init_input=None,
                                                decoder_init_kwargs=kwargs)
  output_ids = outputs.predicted_ids.numpy()
  output_scores = outputs.beam_search_decoder_output.scores.numpy()
  return output_ids, output_scores, lens.numpy()


for step in range(5000):
  x, y_in, y_out, len_x, len_y = generate_data(config.batch_size,
                                               copy_sequence=True)
  loss = train_step(x, y_in, y_out)  # Train with teacher forcing
  if step % 50 == 0:
    print('step', step, 'loss', loss)
    x, _1, _2, _3, _4 = generate_data(num_samples=config.inf_batch_size,
                                      copy_sequence=True)
    print('input data:', x)
    greedy_ids, greedy_lens = greedy_decode(x)
    beam_search_ids, scores, beam_search_lens = \
      beam_search_decode(x, beam_width=config.beam_width)
    for b in range(config.inf_batch_size):
      print('-' * 50 + '\n', b, '-th sample:', x[b])
      print('greedy decoding result:\n', greedy_ids[b][:greedy_lens[b]])
      for w in range(config.beam_width):
        print(w + 1, '-th beam search result:\n',
              beam_search_ids[b, :beam_search_lens[b, w], w],
              'score:', scores[b, :beam_search_lens[b, w], w].sum())
