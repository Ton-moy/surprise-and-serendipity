from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import gensim.corpora as corpora
from gensim.corpora.mmcorpus import MmCorpus
import glob
import os
from nltk.tokenize import word_tokenize
import numpy as np

# Load the pre-trained LDA model and dictionary from the 'lda_model' directory
model = LdaModel.load('./lda_model/lda.model')
dictionary = Dictionary.load('./lda_model/dictionary')

# get the file paths for the books from the 'Books' directory
book_dirs = glob.glob('./Books/*.txt')

for doc_path in book_dirs:
    with open(doc_path, 'r') as f:
        doc = f.read()
    
    # extract the book_ID
    file_name = os.path.basename(doc_path)

    bow = dictionary.doc2bow(word_tokenize(doc))
    
    topics = model[bow]
    
    distr = np.zeros(n_topics)
    for topic_id, value in topics:
        distr[topic_id] = value
        
    # create output path for each book
    out_path = f'./topic_distrs/{file_name}'

    # write the topic distribution to the topic_distrs directory
    with open(out_path, 'w') as f:
        for value in distr:
            f.write('{0:.4f}'.format(value) + ' ')
    
