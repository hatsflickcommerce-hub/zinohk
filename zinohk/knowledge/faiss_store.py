"""
zinohk.knowledge.faiss_store
=============================
FAISS-powered vector store for ZINOHK.

Replaces linear scan retrieval with millisecond
nearest-neighbour search at any scale.

Linear scan : O(N) — fine for 1K facts, broken at 1M
FAISS index : O(log N) — milliseconds at 100M facts

Used by DynamicKnowledgeBase when facts > threshold.
Drops in as a replacement — same API, faster search.
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StoredFact:
    """A single stored fact with metadata."""
    fact_id:  int
    text:     str
    answer:   str
    category: str


class FAISSStore:
    """
    FAISS vector store for fast similarity search.

    Stores text embeddings and retrieves by
    cosine similarity in O(log N) time.

    Parameters
    ----------
    dim       : embedding dimension (384 for MiniLM)
    use_gpu   : use GPU FAISS if available

    Example
    -------
    >>> store = FAISSStore(dim=384)
    >>> store.add(vec, "Paris is the capital", "Paris", "geo")
    >>> results = store.search(query_vec, top_k=3)
    """

    def __init__(self, dim: int = 384, use_gpu: bool = False):
        self.dim     = dim
        self.use_gpu = use_gpu
        self.n_facts = 0

        # Flat L2 index — exact search, no approximation
        # For > 1M facts switch to IndexIVFFlat (approximate)
        self._index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalised vecs

        if use_gpu:
            try:
                res          = faiss.StandardGpuResources()
                self._index  = faiss.index_cpu_to_gpu(res, 0, self._index)
                print("✅ FAISS using GPU")
            except Exception:
                print("⚠️  FAISS GPU not available, using CPU")

        # Metadata storage (FAISS only stores vectors)
        self._facts: List[StoredFact] = []

    # ---------------------------------------------------------------- #
    # Adding facts
    # ---------------------------------------------------------------- #

    def add(
        self,
        vector:   np.ndarray,
        text:     str,
        answer:   str,
        category: str = 'general',
    ) -> int:
        """
        Add one fact to the store.

        Parameters
        ----------
        vector   : embedding vector (dim,) — will be L2-normalised
        text     : full fact text
        answer   : answer string
        category : topic category

        Returns
        -------
        int : fact_id assigned
        """
        vec = vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)   # cosine similarity via inner product
        self._index.add(vec)

        fact = StoredFact(
            fact_id  = self.n_facts,
            text     = text,
            answer   = answer,
            category = category,  # parameter name matches here
        )
        self._facts.append(fact)
        self.n_facts += 1
        return fact.fact_id

    def add_batch(
        self,
        vectors:    np.ndarray,
        texts:      List[str],
        answers:    List[str],
        categories: Optional[List[str]] = None,
    ) -> None:
        """
        Add many facts at once — much faster than one by one.

        Parameters
        ----------
        vectors    : (N, dim) embedding matrix
        texts      : N fact texts
        answers    : N answer strings
        categories : N category strings (optional)
        """
        assert len(vectors) == len(texts) == len(answers)
        if categories is None:
            categories = ['general'] * len(texts)

        vecs = vectors.astype(np.float32)
        faiss.normalize_L2(vecs)
        self._index.add(vecs)

        for i, (text, answer, cat) in enumerate(
                zip(texts, answers, categories)):
            self._facts.append(StoredFact(
                fact_id  = self.n_facts + i,
                text     = text,
                answer   = answer,
                category = cat,
            ))
        self.n_facts += len(texts)
        print(f"✅ Added {len(texts):,} facts | total={self.n_facts:,}")

    # ---------------------------------------------------------------- #
    # Search
    # ---------------------------------------------------------------- #

    def search(
        self,
        query_vec: np.ndarray,
        top_k:     int = 3,
        category:  Optional[str] = None,
    ) -> List[Tuple[float, StoredFact]]:
        """
        Find top-k most similar facts.

        Parameters
        ----------
        query_vec : query embedding (dim,)
        top_k     : number of results
        category  : optional filter by category

        Returns
        -------
        List of (score, StoredFact) sorted by relevance
        """
        if self.n_facts == 0:
            return []

        vec = query_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        # Search more than top_k if filtering by category
        k = min(top_k * 5 if category else top_k, self.n_facts)
        scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._facts):
                continue
            fact = self._facts[idx]
            if category and fact.category != category:
                continue
            results.append((float(score), fact))
            if len(results) >= top_k:
                break

        return results

    # ---------------------------------------------------------------- #
    # Save / Load
    # ---------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save index and metadata to disk."""
        import pickle
        import os
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        cpu_index = faiss.index_gpu_to_cpu(self._index) \
            if self.use_gpu else self._index
        faiss.write_index(cpu_index, f"{path}/faiss.index")

        # Save metadata
        with open(f"{path}/facts.pkl", 'wb') as f:
            pickle.dump(self._facts, f)

        print(f"✅ Saved {self.n_facts:,} facts to {path}/")

    def load(self, path: str) -> None:
        """Load index and metadata from disk."""
        import pickle
        self._index  = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/facts.pkl", 'rb') as f:
            self._facts = pickle.load(f)
        self.n_facts = len(self._facts)
        print(f"✅ Loaded {self.n_facts:,} facts from {path}/")

    # ---------------------------------------------------------------- #
    # Stats
    # ---------------------------------------------------------------- #

    def stats(self) -> dict:
        cats = {}
        for f in self._facts:
            cats[f.category] = cats.get(f.category, 0) + 1
        return {
            "n_facts"   : self.n_facts,
            "dim"       : self.dim,
            "categories": cats,
        }

    def __repr__(self) -> str:
        return (f"FAISSStore(facts={self.n_facts}, "
                f"dim={self.dim})")
