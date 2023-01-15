import tensorflow as tf

batch_size, time_steps = 32, 50
vocab_size, embedding_dim = 10000, 80
num_filters, num_classes = 64, 10
x = tf.keras.layers.Input(shape=(time_steps,))  # [B, T]
emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)(x)  # [B, T, E]
conv1 = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=3,
                               strides=1, padding='valid',
                               activation='relu')(emb)  # [B, T-2, F]
conv2 = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=4,
                               strides=1, padding='valid',
                               activation='relu')(emb)  # [B, T-3, F]
conv3 = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=5,
                               strides=1, padding='valid',
                               activation='relu')(emb)  # [B, T-4, F]
conv1p = tf.keras.layers.GlobalMaxPool1D()(conv1)  # [B, F]
conv2p = tf.keras.layers.GlobalMaxPool1D()(conv2)  # [B, F]
conv3p = tf.keras.layers.GlobalMaxPool1D()(conv3)  # [B, F]
conv_all = tf.keras.layers.Concatenate()([conv1p, conv2p, conv3p])  # [B, 3F]
logits = tf.keras.layers.Dense(units=num_classes)(conv_all)
model = tf.keras.Model(inputs=x, outputs=logits)
model.summary()

word_ids = tf.random.uniform(shape=[batch_size, time_steps],
                             minval=0, maxval=vocab_size - 1,
                             dtype=tf.int32)
print(model(word_ids))
