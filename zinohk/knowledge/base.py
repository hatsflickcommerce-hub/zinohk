"""
zinohk.knowledge.base
======================
Knowledge base — stores facts as spike vectors.

Facts come from:
  - Wikipedia sentences (offline)
  - Kaggle datasets (online, Phase 8b)
  - Manual entry

Each fact is stored as:
  - raw text         : the original sentence
  - vector           : TF-IDF encoded
  - spike pattern    : encoded via SpikeEncoder
  - category         : topic label (science, history, etc.)

Retrieval uses spike_overlap similarity — same mechanism
the brain uses for associative memory.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from zinohk.encoding.text  import TFIDFEncoder
from zinohk.encoding.spike import SpikeEncoder, Spike


@dataclass
class Fact:
    """
    A single piece of knowledge stored in the base.

    Attributes
    ----------
    fact_id  : unique identifier
    text     : raw text of the fact
    answer   : the answer this fact provides
    category : topic (science, history, general, etc.)
    vector   : TF-IDF vector
    spikes   : spike-encoded version
    """
    fact_id:  int
    text:     str
    answer:   str
    category: str
    vector:   np.ndarray
    spikes:   List[Spike]


class KnowledgeBase:
    """
    Spike-encoded knowledge store.

    Stores facts as spike patterns.
    Retrieves by temporal spike similarity.

    Parameters
    ----------
    vocab_size : TF-IDF vocabulary size
    T_max      : spike time window (ms)
    threshold  : minimum signal to spike

    Example
    -------
    >>> kb = KnowledgeBase(vocab_size=500)
    >>> kb.add_fact("The sky is blue",
    ...             answer="blue",
    ...             category="science")
    >>> results = kb.retrieve("What colour is the sky?", top_k=1)
    >>> results[0].answer
    'blue'
    """

    def __init__(
        self,
        vocab_size: int   = 1000,
        T_max:      float = 20.0,
        threshold:  float = 0.05,
    ):
        self.vocab_size = vocab_size
        self.T_max      = T_max
        self.threshold  = threshold

        self.text_enc  = TFIDFEncoder(max_vocab=vocab_size, min_freq=1)
        self.spike_enc = SpikeEncoder(T_max=T_max, threshold=threshold)

        self.facts:    List[Fact] = []
        self.is_built: bool       = False

    # ---------------------------------------------------------------- #
    # Building the knowledge base
    # ---------------------------------------------------------------- #

    def build(self, texts: List[str], answers: List[str],
              categories: Optional[List[str]] = None) -> None:
        """
        Fit encoder on corpus and encode all facts.

        Parameters
        ----------
        texts      : list of fact sentences
        answers    : corresponding answers
        categories : optional topic labels
        """
        if categories is None:
            categories = ['general'] * len(texts)

        assert len(texts) == len(answers), \
            "texts and answers must have same length"

        # Fit TF-IDF on full corpus
        self.text_enc.fit(texts)

        # Encode each fact
        self.facts = []
        for i, (text, answer, cat) in enumerate(
                zip(texts, answers, categories)):
            vec    = self.text_enc.encode(text)
            spikes = self.spike_enc.encode(vec)
            self.facts.append(Fact(
                fact_id  = i,
                text     = text,
                answer   = answer,
                category = cat,
                vector   = vec,
                spikes   = spikes,
            ))

        self.is_built = True
        print(f"✅ KnowledgeBase built: {len(self.facts)} facts | "
              f"vocab={self.text_enc.vocab_size}")

    def add_fact(
        self,
        text:     str,
        answer:   str,
        category: str = 'general',
    ) -> None:
        """Add a single fact after build."""
        assert self.is_built, "Call build() first"
        vec    = self.text_enc.encode(text)
        spikes = self.spike_enc.encode(vec)
        self.facts.append(Fact(
            fact_id  = len(self.facts),
            text     = text,
            answer   = answer,
            category = category,
            vector   = vec,
            spikes   = spikes,
        ))

    # ---------------------------------------------------------------- #
    # Retrieval
    # ---------------------------------------------------------------- #

    def retrieve(
        self,
        query:    str,
        top_k:   int   = 3,
        category: Optional[str] = None,
    ) -> List[tuple]:
        """
        Find most relevant facts for a query.

        Uses spike temporal overlap — same mechanism as
        biological associative memory.

        Parameters
        ----------
        query    : question text
        top_k    : number of top facts to return
        category : optional filter by topic

        Returns
        -------
        List of (score, Fact) sorted by relevance
        """
        assert self.is_built, "Call build() first"

        # Encode query
        q_vec    = self.text_enc.encode(query)
        q_spikes = self.spike_enc.encode(q_vec)

        # Score each fact
        scores = []
        for fact in self.facts:
            if category and fact.category != category:
                continue

            # Spike overlap similarity
            spike_score = self.spike_enc.spike_overlap(
                q_spikes, fact.spikes, tau=5.0)

            # Also compute cosine similarity as backup
            denom = (np.linalg.norm(q_vec) *
                     np.linalg.norm(fact.vector) + 1e-8)
            cos_score = float(q_vec @ fact.vector / denom)

            # Combined score
            combined = 0.6 * spike_score + 0.4 * cos_score
            scores.append((combined, fact))

        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

    # ---------------------------------------------------------------- #
    # Stats
    # ---------------------------------------------------------------- #

    def stats(self) -> dict:
        cats = {}
        for f in self.facts:
            cats[f.category] = cats.get(f.category, 0) + 1
        return {
            "total_facts": len(self.facts),
            "vocab_size" : self.text_enc.vocab_size,
            "categories" : cats,
            "is_built"   : self.is_built,
        }

    def __repr__(self) -> str:
        return (f"KnowledgeBase(facts={len(self.facts)}, "
                f"vocab={self.text_enc.vocab_size}, "
                f"built={self.is_built})")
