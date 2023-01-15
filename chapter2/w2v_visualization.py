import matplotlib.pyplot as plt
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

np.random.seed(12)
vectors = KeyedVectors.load_word2vec_format(
  '../pretrained/w2v/GoogleNews-vectors-negative300.bin',
  binary=True, limit=50000)
print('hello =', vectors['hello'])

words = [word for word in vectors.index_to_key[:200]]
embeddings = [vectors[word] for word in words]
words_embedded = TSNE(n_components=2).fit_transform(embeddings)

plt.figure(figsize=(10, 10))
for i, label in enumerate(words):
  x, y = words_embedded[i, :]
  plt.scatter(x, y)
  plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
               ha='right', va='bottom')
plt.show()

capitals = ['Beijing', 'Moscow', 'Washington', 'Tokyo', 'Rome',
            'Paris', 'Berlin', 'Athens', 'London', 'Ottawa']
countries = ['China', 'Russia', 'America', 'Japan', 'Italy',
             'France', 'Germany', 'Greece', 'England', 'Canada']
embeddings = [vectors[word] for word in capitals + countries]
words_embedded = PCA(n_components=2).fit_transform(embeddings)

num_pairs = len(capitals)
for i in range(num_pairs):
  x1, y1 = words_embedded[i]  # capital
  plt.scatter(x1, y1)
  plt.annotate(capitals[i], xy=(x1, y1), xytext=(5, 2),
               textcoords='offset points', ha='right', va='bottom')

  x2, y2 = words_embedded[i + num_pairs]  # country
  plt.scatter(x2, y2)
  plt.annotate(countries[i], xy=(x2, y2), xytext=(5, 2),
               textcoords='offset points', ha='right', va='bottom')
  plt.plot([x1, x2], [y1, y2])
plt.show()

print('cat', vectors.most_similar(positive=['cat'], topn=3))
print('king + woman - man = ?',
      vectors.most_similar(positive=['king', 'woman'],
                           negative=['man'], topn=3))

analogy_file = '../datasets/w2v/questions-words.txt'
result = vectors.evaluate_word_analogies(analogy_file)
print('Analogy accuracy:', result[0])
for section_result in result[1]:
  print('section name:', section_result['section'],
        ', correct:', section_result['correct'])
