"""
Plain keyword search (BM25), the classic search-engine ranking.

Embeddings are great at "meaning" but can miss exact terms like a part
number, a name or an acronym. BM25 is the opposite. Using both together
(hybrid search) usually beats either one on its own.
"""

import math
import re
from collections import Counter

WORD = re.compile(r"[a-z0-9]+")

# very common words that don't help ranking
STOPWORDS = set("a an the and or of to in on for is are was were be by with as at it this that from what how why".split())


def stem(word):
    """
    A very rough stemmer so "exports", "exported" and "exporting" all match
    "export". Not linguistically perfect, but good enough for search.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix) and not word.endswith("ss"):
            return word[: -len(suffix)]
    return word


def tokenize(text):
    return [stem(w) for w in WORD.findall(text.lower()) if w not in STOPWORDS]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(1, len(self.docs))
        # how many chunks each word appears in
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query):
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                if t not in doc:
                    continue
                tf = doc[t]
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            out.append(s)
        return out


def rank(scores):
    """Indexes sorted best first."""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def reciprocal_rank_fusion(*rankings, k=60):
    """
    Merge several rankings into one. Each list gives 1 / (k + position) to
    every item, and we add them up. It only uses positions, so we don't have
    to worry about BM25 scores and cosine similarities being on different scales.
    """
    fused = Counter()
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            fused[idx] += 1.0 / (k + pos + 1)
    return [idx for idx, _ in fused.most_common()]
