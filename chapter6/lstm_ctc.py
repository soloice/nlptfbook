import numpy as np
import tensorflow as tf

time_steps = 200
input_dim = 39  # MFCC features
vocab_size = 28  # 26 characters + white space + blank symbol

model = tf.keras.Sequential([
  tf.keras.Input(batch_input_shape=(None, None, input_dim)),  # [B, T, D]
  tf.keras.layers.LSTM(units=128, return_sequences=True,
                       time_major=False, go_backwards=False,
                       dropout=0.1, recurrent_dropout=0.1),  # [B, T, D']
  tf.keras.layers.Dense(units=vocab_size, activation='relu'),  # [B, T, V]
])
model.summary()
optim = tf.keras.optimizers.Adam(learning_rate=0.1, clipnorm=1.0)

# Training
batch_size = 2
x_lengths = [150, 180]
x = np.random.randn(batch_size, time_steps, input_dim)
y_lengths = [9, 12]
labels = np.random.randint(low=1, high=vocab_size,
                           size=(batch_size, time_steps))

with tf.GradientTape() as tape:
  logits = model(x)
  loss = tf.nn.ctc_loss(labels=labels, logits=logits,
                        label_length=y_lengths, logit_length=x_lengths,
                        blank_index=0,
                        logits_time_major=False)
optim.minimize(loss, model.trainable_variables, tape=tape)

# Inference
logits = model(x)  # [B, T, V]
logits_time_major = tf.transpose(logits, perm=[1, 0, 2])  # [T, B, V]
pred_labels, log_probs = \
  tf.nn.ctc_beam_search_decoder(inputs=logits_time_major,
                                sequence_length=x_lengths,
                                beam_width=50,
                                top_paths=1)
print([(pred_label.indices, pred_label.values)
       for pred_label in pred_labels])
print(log_probs)
