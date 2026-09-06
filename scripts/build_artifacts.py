import json
import numpy as np
import faiss
import pickle
from rank_bm25 import BM25Okapi
import string
from sklearn.feature_extraction import _stop_words
from sentence_transformers import SentenceTransformer

text = """
Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan.
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine.
Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.
Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007.
Caltech theoretical physicist and 2017 Nobel laureate in Physics Kip Thorne was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar.
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm.
Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles.
Interstellar uses extensive practical and miniature effects and the company Double Negative created additional digital effects.
Interstellar premiered on October 26, 2014, in Los Angeles.
In the United States, it was first released on film stock, expanding to venues using digital projectors.
The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014.
It received acclaim for its performances, direction, screenplay, musical score, visual effects, ambition, themes, and emotional weight.
It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics. Since its premiere, Interstellar gained a cult following, and now is regarded by many sci-fi experts as one of the best science-fiction films of all time.
Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades"""

texts = [t.strip(' \n') for t in text.split('.') if t.strip(' \n')]
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
embeds = embedder.encode(texts, convert_to_numpy=True)
dim = embeds.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.float32(embeds))
faiss.write_index(index, "artifacts/faiss.index")
with open("artifacts/texts.json", "w") as f:
    json.dump(texts, f)
print(f"Saved index with {index.ntotal} vectors and {len(texts)} texts.")

def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

tokenized_corpus = [bm25_tokenizer(passage) for passage in texts]
bm25 = BM25Okapi(tokenized_corpus)

with open("artifacts/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

print(f"Saved BM25 index over {len(tokenized_corpus)} documents.")