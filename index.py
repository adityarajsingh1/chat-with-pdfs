"""
Turning chunks into embeddings and finding the ones closest to a question.

I'm using OpenAI's embedding model (cheap and good) and plain numpy
for the search. For a few hundred pages you really don't need a vector
database, a matrix and a dot product are plenty fast.

The index gets saved next to your PDFs so you only pay for embeddings once.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from bm25 import BM25, rank, reciprocal_rank_fusion

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
BATCH = 100  # the api accepts lots of inputs per call, this keeps requests a sane size


def embed(texts):
    from openai import OpenAI

    client = OpenAI()
    vectors = []
    for i in range(0, len(texts), BATCH):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + BATCH])
        vectors.extend(d.embedding for d in resp.data)
    arr = np.array(vectors, dtype=np.float32)
    # normalise so a dot product is the same as cosine similarity
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


class Index:
    def __init__(self, chunks, vectors=None):
        self.chunks = chunks
        self.vectors = vectors  # None when running keyword-only
        self.bm25 = BM25([c["text"] for c in chunks])

    @classmethod
    def build(cls, chunks, embed_fn=embed):
        print(f"embedding {len(chunks)} chunks...")
        return cls(chunks, embed_fn([c["text"] for c in chunks]))

    def search(self, question, k=5, mode="hybrid", embed_fn=embed):
        """mode is "hybrid" (default), "embedding" or "keyword"."""
        if self.vectors is None:
            mode = "keyword"

        keyword_scores = self.bm25.scores(question)
        if mode == "keyword":
            best = rank(keyword_scores)[:k]
            return [dict(self.chunks[i], score=keyword_scores[i]) for i in best]

        q = embed_fn([question])[0]
        sims = self.vectors @ q
        if mode == "embedding":
            best = list(np.argsort(sims)[::-1][:k])
        else:
            best = reciprocal_rank_fusion(list(np.argsort(sims)[::-1]), rank(keyword_scores))[:k]
        # report the cosine similarity, it's the easier number to read
        return [dict(self.chunks[i], score=float(sims[i])) for i in best]

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        return cls(chunks, np.load(folder / "vectors.npy"))


def fingerprint(pdf_folder):
    """A hash of the pdf names + sizes + edit times, so we know when to rebuild."""
    pdf_folder = Path(pdf_folder)
    files = [pdf_folder] if pdf_folder.is_file() else sorted(pdf_folder.glob("*.pdf"))
    h = hashlib.sha1(EMBED_MODEL.encode())
    for f in files:
        st = f.stat()
        h.update(f"{f.name}{st.st_size}{st.st_mtime}".encode())
    return h.hexdigest()[:12]


def get_index(pdf_folder, chunks_fn, embed_fn=embed, use_embeddings=True):
    """Load the cached index if the PDFs haven't changed, otherwise build a new one."""
    if not use_embeddings:
        # keyword search is cheap, no need to cache anything
        return Index(chunks_fn(pdf_folder))
    base = Path(pdf_folder)
    cache = (base.parent if base.is_file() else base) / ".pdf-index" / fingerprint(pdf_folder)
    if (cache / "vectors.npy").exists():
        print("using cached index")
        return Index.load(cache)
    index = Index.build(chunks_fn(pdf_folder), embed_fn)
    cache.parent.mkdir(exist_ok=True)
    index.save(cache)
    return index
