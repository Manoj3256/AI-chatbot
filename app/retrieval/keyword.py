from sklearn.feature_extraction import _stop_words
import string

def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

import pickle
bm25 = None

def load_bm25():
    global bm25
    with open("artifacts/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)