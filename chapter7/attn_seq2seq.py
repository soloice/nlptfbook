import numpy as np
import matplotlib.pyplot as plt
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

    self.fc = tf.keras.layers.Dense(config.vocab_size, name='decoder_fc')

    # Note: this is GRUCell instead of GRU!!!
    rnn_cell = tf.keras.layers.GRUCell(config.hidden_size)
    # Pass memory_layer=tf.keras.layers.Lambda(lambda x: x) to disable
    #   transformation on keys. i.e.: calculate <q, k> instead of <q, Wk>
    self.attn = tfa.seq2seq.LuongAttention(units=config.attn_size,
                                           memory=None,
                                           memory_sequence_length=None)
    self.rnn_cell = tfa.seq2seq.AttentionWrapper(
      cell=rnn_cell,
      attention_mechanism=self.attn,
      alignment_history=True)

    self.sampler = tfa.seq2seq.TrainingSampler()
    self.decoder = tfa.seq2seq.BasicDecoder(cell=self.rnn_cell,
                                            sampler=self.sampler,
                                            output_layer=self.fc)

  def call(self, inputs, init_state=None):
    # x: [B, U], memory: [B, T, D], init_state=[B, D]
    inputs_embedded = self.emb(inputs)  # [B, U, E]
    # final_outputs of type BasicDecoderOutput:
    #     (rnn_output: [B, U, D], sample_id: [B, D])
    # final_states: AttentionWrapperState of batch size B
    # final_lengths: [B]
    output, _, _ = self.decoder(inputs_embedded, initial_state=init_state)
    return output.rnn_output  # [B, U, D]


def build_model():
  # Do a forward step on synthesized data to instantiate the model
  encoder, decoder = Encoder(), Decoder()
  x, y_in, y_out, len_x, len_y = generate_data(config.batch_size,
                                               copy_sequence=True)
  enc_out, enc_state = encoder(x, training=True)  # [B, T, D], [B, D]
  decoder.attn.setup_memory(memory=enc_out, memory_sequence_length=len_x)
  zero_state = decoder.rnn_cell.get_initial_state(batch_size=x.shape[0],
                                                  dtype=tf.float32)
  init_state = zero_state.clone(cell_state=enc_state)
  _ = decoder(y_in, init_state, training=True)  # [B, U, D]
  return encoder, decoder


encoder, decoder = build_model()
optim = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)


def loss_fn(labels, logits, masks):
  # labels: [B, U], logits: [B, U, V], masks: [B, U]
  # ce: [B, U]
  ce = tf.keras.losses.sparse_categorical_crossentropy(labels, logits,
                                                       from_logits=True)
  return tf.reduce_sum(ce * masks) / tf.reduce_sum(masks)


@tf.function(experimental_relax_shapes=True)
def train_step(x, lx, y_in, y_out):
  # x: [B, T], lx: [B], y_in: [B, U], y_out: [B, U]
  mask = tf.cast(y_out > config.PAD, dtype=tf.float32)
  with tf.GradientTape() as tape:
    enc_out, enc_state = encoder(x, training=True)  # [B, T, D], [B, D]
    decoder.attn.setup_memory(memory=enc_out, memory_sequence_length=lx)
    zero_state = decoder.rnn_cell.get_initial_state(batch_size=x.shape[0],
                                                    dtype=tf.float32)
    init_state = zero_state.clone(cell_state=enc_state)
    logits = decoder(y_in, init_state, training=True)  # [B, U, D]
    loss = loss_fn(labels=y_out, logits=logits, masks=mask)
  all_params = encoder.trainable_variables + decoder.trainable_variables
  optim.minimize(loss, all_params, tape=tape)
  return loss


def greedy_decode(x, lx):
  # x: [B, T], lx: [B]
  greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler(decoder.emb)
  max_iter = config.max_len * 2
  basic_decoder = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell,
                                           sampler=greedy_sampler,
                                           output_layer=decoder.fc)
  curr_batch_size = int(x.shape[0])
  encoder_outputs, encoder_state = encoder(x)  # [B, T, D], [B, D]
  start_tokens = tf.tile([config.BOS], [curr_batch_size])  # [B]
  decoder.attn.setup_memory(memory=encoder_outputs, memory_sequence_length=lx)
  zero_state = decoder.rnn_cell.get_initial_state(batch_size=x.shape[0],
                                                  dtype=tf.float32)
  init_state = zero_state.clone(cell_state=encoder_state)

  # final_outputs: tfa.seq2seq.BasicDecoderOutput
  #     (rnn_output: [B, U, D], sample_id: [B, D])
  # final_states: AttentionWrapperState of batch size B
  # final_lengths: [B]
  outputs, _, lens = tfa.seq2seq.dynamic_decode(basic_decoder,
                                                maximum_iterations=max_iter,
                                                decoder_init_input=None,
                                                decoder_init_kwargs={
                                                  'initial_state': init_state,
                                                  'start_tokens': start_tokens,
                                                  'end_token': config.EOS})
  return outputs.sample_id.numpy(), lens.numpy()


def beam_search_decode(x, lx, beam_width):
  # x: [B, T], lx: [B]
  beam_decoder = tfa.seq2seq.BeamSearchDecoder(decoder.rnn_cell,
                                               beam_width=beam_width,
                                               embedding_fn=decoder.emb,
                                               output_layer=decoder.fc)
  max_iter = config.max_len * 2
  curr_batch_size = int(x.shape[0])
  encoder_outputs, encoder_final_state = encoder(x)  # _, [B, D]
  encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs,
                                           multiplier=beam_width)
  lx = tfa.seq2seq.tile_batch(lx, multiplier=beam_width)
  decoder.attn.setup_memory(encoder_outputs, memory_sequence_length=lx)
  # set up decoder_initial_state
  decoder_initial_state = decoder.rnn_cell.get_initial_state(
    batch_size=beam_width * curr_batch_size, dtype=tf.float32)
  tiled_states = tfa.seq2seq.tile_batch(encoder_final_state,
                                        multiplier=beam_width)
  decoder_initial_state = decoder_initial_state.clone(cell_state=tiled_states)

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


def plot_attention_weights():
  # Visualize a fixed batch of size 1
  x = np.array([[1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9]])
  y_in = np.array([[10, 2, 2, 4, 6, 8, 2, 2, 4, 6, 8]])
  y_out = np.array([[2, 2, 4, 6, 8, 2, 2, 4, 6, 8, 11]])
  len_x = np.array([len(i) for i in x])
  len_y = np.array([len(i) for i in y_in])
  print('plot attention:', x, y_in, y_out, len_x, len_y)
  greedy_ids, len_g = greedy_decode(x, len_x)
  print('greedy ids:', greedy_ids, len_g)
  enc_out, enc_state = encoder(x, training=False)  # [B, T, D], [B, D]
  decoder.attn.setup_memory(memory=enc_out, memory_sequence_length=len_x)
  zero_state = decoder.rnn_cell.get_initial_state(batch_size=x.shape[0],
                                                  dtype=tf.float32)
  init_state = zero_state.clone(cell_state=enc_state)
  greedy_ids_in = np.array([[config.BOS] + ids[:-1].tolist()
                            for ids in greedy_ids])
  inputs_embedded = decoder.emb(greedy_ids_in)  # [B, U, E]
  # outputs, states, lengths
  _, state, _ = decoder.decoder(inputs_embedded, initial_state=init_state)
  all_alignments = state.alignment_history.stack()  # [U, B, T]
  attn_heatmap = all_alignments.numpy()[:len_g[0], 0, :len_x[0]]  # [U, T]

  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(attn_heatmap, interpolation='nearest')
  fig.colorbar(cax)
  plt.xticks(np.arange(len(x[0])))
  plt.yticks(np.arange(len(greedy_ids[0])))
  ax.set_xticklabels([str(i) for i in x[0]])
  ax.set_yticklabels([str(i) for i in greedy_ids[0]])

  ax.set_xlabel('Input sequence')
  ax.set_ylabel('Output sequence')
  plt.title('Attention heatmap')
  plt.show()


if __name__ == '__main__':
  for step in range(5000):
    x, y_in, y_out, len_x, len_y = generate_data(config.batch_size,
                                                 copy_sequence=True)
    loss = train_step(x, len_x, y_in, y_out)  # Train with teacher forcing
    if step == 0:
      for p in decoder.trainable_variables:
        print(p.name, p.shape)
    if step % 50 == 0:
      print('step', step, 'loss', loss)
      x, _1, _2, len_x, _4 = generate_data(num_samples=config.inf_batch_size,
                                           copy_sequence=True)
      print('input data:', x)
      greedy_ids, greedy_lens = greedy_decode(x, len_x)
      beam_search_ids, scores, beam_search_lens = \
        beam_search_decode(x, len_x, beam_width=config.beam_width)
      for b in range(config.inf_batch_size):
        print('-' * 50 + '\n', b, '-th sample:', x[b])
        print('greedy decoding result:\n', greedy_ids[b][:greedy_lens[b]])
        for w in range(config.beam_width):
          print(w + 1, '-th beam search result:\n',
                beam_search_ids[b, :beam_search_lens[b, w], w],
                'score:', scores[b, :beam_search_lens[b, w], w].sum())

    if step % 1000 == 0:
      plot_attention_weights()
  plot_attention_weights()
