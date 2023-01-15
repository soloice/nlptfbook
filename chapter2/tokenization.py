import jieba
import nltk

sentence = "There's beggary in the love that can be reckoned."
tokenizer = nltk.tokenize.NLTKWordTokenizer()
print(tokenizer.tokenize(sentence))

sentence = "其实地上本没有路，走的人多了，也便成了路。"
print(jieba.lcut(sentence))
