import glob
import os
import nltk
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
import numpy as np
from nltk.tokenize import word_tokenize
import gensim.corpora as corpora
from gensim.corpora.mmcorpus import MmCorpus

documents = []

# get the file paths for the books from the 'Books' directory
book_paths = glob.glob("./Books/*.txt")

# create LDABooks set to train LDA model
for book in book_paths:
    with open(book, 'r') as f:
        tokens = word_tokenize(f.read())
        if len(tokens) > 1000:
            documents.append(tokens)

# create a dictionary and save it
dictionary = Dictionary(documents)
dictionary.filter_extremes(5, 0.5, 100000)
dictionary.save('./lda_model/dictionary')

corpus = [dictionary.doc2bow(text) for text in documents]

# set the number of topics
n_topics = 100

# train the LDA model
lda = LdaModel(corpus=corpus,id2word=dictionary, num_topics=n_topics, passes = 2)

# save the trained LDA model to the 'lda_model' directory
lda.save('./lda_model/lda.model')
print('Model Saved...')
corpora.MmCorpus.serialize('./lda_model/corpus', corpus)

    


