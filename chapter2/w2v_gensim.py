from time import time

from gensim.models import word2vec

# Training a skip-gram model
corpus_file = "../datasets/ptb/ptb.train.txt"
model = word2vec.Word2Vec(corpus_file=corpus_file,
                          min_count=100,  # discard infrequent words
                          window=5,  # max window size
                          vector_size=50,  # word embedding dimension
                          sample=6e-5,  # high-frequency words subsampling rate
                          alpha=0.03,  # initial learning rate
                          min_alpha=0.0007,  # final learning rate
                          negative=2,  # number of negative samples
                          workers=8,  # number of workers
                          seed=1234)

t = time()
model.train(corpus_file=corpus_file, total_examples=model.corpus_count,
            epochs=50, total_words=model.corpus_total_words, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

analogy_file = '../datasets/w2v/questions-words.txt'
result = model.wv.evaluate_word_analogies(analogy_file)
print('Analogy accuracy:', result[0])
for section_result in result[1]:
  print('section name:', section_result['section'],
        ', correct:', section_result['correct'])
