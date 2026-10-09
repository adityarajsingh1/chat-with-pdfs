import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bm25 import BM25, rank, reciprocal_rank_fusion  # noqa: E402
from index import Index  # noqa: E402

CHUNKS = [
    {"text": "The transformer uses multi-head attention.", "source": "a.pdf", "page": 1},
    {"text": "Error code E-4012 means the pump overheated.", "source": "b.pdf", "page": 7},
    {"text": "Attention lets each token look at the others.", "source": "a.pdf", "page": 2},
]


def test_bm25_finds_exact_terms():
    bm = BM25([c["text"] for c in CHUNKS])
    assert rank(bm.scores("what does E-4012 mean?"))[0] == 1


def test_bm25_unknown_words_score_zero():
    bm = BM25([c["text"] for c in CHUNKS])
    assert max(bm.scores("bananas")) == 0


def test_rrf_rewards_items_both_lists_like():
    fused = reciprocal_rank_fusion([0, 1, 2], [2, 1, 0], [1, 0, 2])
    assert fused[0] in (0, 1)
    assert set(fused) == {0, 1, 2}


def test_keyword_only_index_needs_no_embeddings():
    index = Index(CHUNKS)
    hits = index.search("pump overheated", k=1, mode="hybrid")  # falls back to keyword
    assert hits[0]["page"] == 7


def test_hybrid_combines_both():
    # fake embeddings where chunk 2 is semantically closest to the question
    vecs = np.eye(3, dtype=np.float32)
    index = Index(CHUNKS, vecs)
    fake_embed = lambda texts: np.array([[0.1, 0.0, 0.99]], dtype=np.float32)
    emb = index.search("how does attention work?", k=1, mode="embedding", embed_fn=fake_embed)
    assert emb[0]["page"] == 2
    hybrid = index.search("multi-head attention", k=2, mode="hybrid", embed_fn=fake_embed)
    assert {h["page"] for h in hybrid} == {1, 2}


def test_word_endings_still_match():
    from bm25 import tokenize
    assert tokenize("exports exported exporting") == ["export"] * 3
    assert tokenize("batteries class") == ["battery", "class"]
    bm = BM25(["the capital is Fredville", "Freedonia exports cheese"])
    assert rank(bm.scores("what does freedonia export?"))[0] == 1
