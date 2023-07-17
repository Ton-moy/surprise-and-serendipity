import gzip
import json
import os
import time
import numpy as np

import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector

def get_lang_detector(nlp, name):
    return LanguageDetector(seed=42)

# load the English language model and add the language detector
nlp_model = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp_model.add_pipe('language_detector', last=True)

# specify the path of raw dataset file
data_file = './goodreads_reviews_dedup.json.gz'

document_dict = {}
token_count = {}

# open the gzip-compressed data file
with gzip.open(data_file) as f:
    for line in f:
        d = json.loads(line)
        book = d['book_id']
        text = d['review_text']
        
        # consider only English reviews
        if (nlp_model(text)._.language['language']) == 'en':
            tokens = text.split()
            len_tokens = len(tokens)

            # consider reviews with at least 50 tokens and add up to 10,000 tokens for each book
            if len_tokens > 50:
                if book not in document_dict:
                    document_dict[book] = text
                    token_count[book] = len_tokens
                else:
                    if token_count[book] < 10000:
                        document_dict[book] += ' ' + d['review_text']
                        token_count[book] += len_tokens
                        
# save each book's document as a separate text file in book directory
for book in document_dict:
    with open(f'./Books/{book}.txt', 'w') as out_file:
        out_file.write(document_dict[book])
        

