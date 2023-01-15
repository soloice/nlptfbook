from chapter5.ner.data_util import get_trimmed_glove_vectors, load_vocab, \
  get_processing_word


class Config(object):
  def __init__(self, load=False):
    # dataset configs
    self.filename_dev = "../../datasets/conll2003/valid.txt"
    self.filename_test = "../../datasets/conll2003/test.txt"
    self.filename_train = "../../datasets/conll2003/train.txt"
    self.dim_word = 50
    self.filename_glove = f"../../pretrained/glove.6B/" \
                          f"glove.6B.{self.dim_word}d.txt"
    # trimmed embeddings (created from glove_filename with build_data.py)
    self.filename_trimmed = f"../../pretrained/glove.6B/" \
                            f"glove.6B.{self.dim_word}d.trimmed.npz"
    self.filename_words = "../../datasets/conll2003/words.txt"
    self.filename_tags = "../../datasets/conll2003/tags.txt"

    self.max_iter = None  # if not None, max number of examples in Dataset

    # training configs
    self.nepochs = 15
    self.dropout = 0.05
    self.batch_size = 20
    self.lr_method = "adam"
    self.lr = 1e-3
    # self.lr_decay = 0.3
    self.clip = 1.0  # gradient clipping
    self.hidden_size = 128  # bi-lstm hidden size
    if load:
      self.load()

  def load(self):
    # 1. vocabulary
    self.vocab_words = load_vocab(self.filename_words)
    self.vocab_tags = load_vocab(self.filename_tags)
    # self.vocab_chars = load_vocab(self.filename_chars)

    self.nwords = len(self.vocab_words)
    # self.nchars = len(self.vocab_chars)
    self.ntags = len(self.vocab_tags)

    # 2. get processing functions that map str -> id
    self.processing_word = get_processing_word(self.vocab_words,
                                               lowercase=True)
    self.processing_tag = get_processing_word(self.vocab_tags,
                                              lowercase=False,
                                              allow_unk=False)

    # 3. get pre-trained embeddings
    self.embeddings = get_trimmed_glove_vectors(self.filename_trimmed)
