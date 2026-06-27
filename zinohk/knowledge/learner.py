"""
zinohk.knowledge.learner
=========================
Dynamic knowledge learner — no hardcoding.

ZINOHK learns from ANY source:
  - Wikipedia (live fetch)
  - Text files
  - Plain strings
  - Lists of facts

Key design:
  - Vocabulary rebuilds when new facts are added
  - No hardcoded knowledge
  - Grows continuously from any data source
  - SlowMemory consolidates over time
"""

import re
import os
from typing import List, Optional, Tuple
from zinohk.encoding.text  import TFIDFEncoder, tokenise
from zinohk.encoding.spike import SpikeEncoder
from zinohk.knowledge.retriever import (
    fetch_wikipedia, optimise_for_wikipedia,
    normalise_query, extract_answer
)


class DynamicKnowledgeBase:
    """
    A knowledge base that grows from any data source.

    No hardcoded facts. Vocabulary rebuilds automatically
    when new facts arrive so all facts are always searchable.

    Parameters
    ----------
    vocab_size : max vocabulary size
    T_max      : spike time window
    threshold  : spike threshold
    """

    def __init__(
        self,
        vocab_size: int   = 2000,
        T_max:      float = 20.0,
        threshold:  float = 0.05,
    ):
        self.vocab_size = vocab_size
        self.T_max      = T_max
        self.threshold  = threshold

        # Raw storage — before encoding
        self._raw_texts:    List[str] = []
        self._raw_answers:  List[str] = []
        self._raw_cats:     List[str] = []

        # Encoded storage — rebuilt when needed
        self._vectors   = None
        self._spikes    = None
        self._dirty     = True   # needs rebuild

        self.text_enc  = TFIDFEncoder(
            max_vocab=vocab_size, min_freq=1)
        self.spike_enc = SpikeEncoder(
            T_max=T_max, threshold=threshold)

        self.n_added   = 0
        self.n_queries = 0

    # ------------------------------------------------------------ #
    # Adding knowledge — from any source
    # ------------------------------------------------------------ #

    def learn(
        self,
        text:     str,
        answer:   str,
        category: str = 'general',
    ) -> None:
        """Learn one fact from any source."""
        # Avoid exact duplicates
        if text in self._raw_texts:
            return

        self._raw_texts.append(text)
        self._raw_answers.append(answer)
        self._raw_cats.append(category)
        self._dirty = True
        self.n_added += 1

    def learn_many(
        self,
        facts: List[Tuple[str, str, str]],
    ) -> None:
        """
        Learn many facts at once.

        Parameters
        ----------
        facts : list of (text, answer, category)
        """
        for text, answer, cat in facts:
            self.learn(text, answer, cat)
        print(f"✅ Learned {len(facts)} facts | "
              f"total={self.n_added}")

    def learn_from_file(self, path: str) -> int:
        """
        Learn from a text file.

        Expected format (one fact per line):
          fact text | answer | category

        Or just plain sentences (answer = sentence,
        category = 'text'):
          The sky is blue because of Rayleigh scattering.

        Returns number of facts learned.
        """
        assert os.path.exists(path), f"File not found: {path}"
        learned = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('|')
                if len(parts) == 3:
                    self.learn(parts[0].strip(),
                               parts[1].strip(),
                               parts[2].strip())
                else:
                    # Plain sentence — use as both text and answer
                    self.learn(line, line, 'text')
                learned += 1
        print(f"✅ Learned {learned} facts from {path}")
        return learned

    def learn_from_wikipedia(self, query: str) -> Optional[str]:
        """
        Fetch and learn a fact from Wikipedia.

        Parameters
        ----------
        query : what to search for

        Returns
        -------
        str : the fetched text, or None
        """
        opt  = optimise_for_wikipedia(query)
        text = fetch_wikipedia(opt, sentences=3)
        if text:
            answer = extract_answer(text, query)
            self.learn(text, answer, 'wikipedia')
            return text
        return None

    # ------------------------------------------------------------ #
    # Rebuild index — called automatically before retrieval
    # ------------------------------------------------------------ #

    def _rebuild(self) -> None:
        """Rebuild TF-IDF and spike encodings for all facts."""
        if not self._raw_texts:
            return

        import numpy as np

        # Refit encoder on ALL current texts
        self.text_enc = TFIDFEncoder(
            max_vocab=self.vocab_size, min_freq=1)
        self.text_enc.fit(self._raw_texts)

        # Re-encode everything
        self._vectors = []
        self._spikes  = []
        for text in self._raw_texts:
            vec = self.text_enc.encode(text)
            self._vectors.append(vec)
            self._spikes.append(self.spike_enc.encode(vec))

        self._dirty = False

    # ------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------ #

    def retrieve(
        self,
        query:    str,
        top_k:   int   = 3,
        category: Optional[str] = None,
    ) -> List[Tuple[float, dict]]:
        """
        Retrieve most relevant facts for a query.

        Rebuilds index automatically if new facts were added.

        Returns
        -------
        List of (score, fact_dict) sorted by relevance
        """
        import numpy as np

        self.n_queries += 1

        if self._dirty:
            self._rebuild()

        if not self._raw_texts:
            return []

        q_vec    = self.text_enc.encode(query)
        q_spikes = self.spike_enc.encode(q_vec)

        scores = []
        for i, (vec, spikes, text, answer, cat) in enumerate(
            zip(self._vectors, self._spikes,
                self._raw_texts, self._raw_answers,
                self._raw_cats)
        ):
            if category and cat != category:
                continue

            # Spike overlap
            spike_score = self.spike_enc.spike_overlap(
                q_spikes, spikes, tau=5.0)

            # Cosine similarity
            denom     = (np.linalg.norm(q_vec) *
                        np.linalg.norm(vec) + 1e-8)
            cos_score = float(q_vec @ vec / denom)

            combined = 0.6 * spike_score + 0.4 * cos_score
            scores.append((combined, {
                'text'    : text,
                'answer'  : answer,
                'category': cat,
                'index'   : i,
            }))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

    def ask(
        self,
        question:             str,
        confidence_threshold: float = 0.35,
        use_wikipedia:        bool  = True,
    ) -> dict:
        """
        Answer a question — local first, Wikipedia fallback.

        Parameters
        ----------
        question             : natural language question
        confidence_threshold : min score for direct answer
        use_wikipedia        : fetch from web if not confident

        Returns
        -------
        dict: answer, confidence, source, fact
        """
        expanded = normalise_query(question)
        results  = self.retrieve(expanded, top_k=1)

        if results and results[0][0] >= confidence_threshold:
            score, fact = results[0]
            return {
                'answer'    : fact['answer'],
                'confidence': round(score, 4),
                'source'    : fact['category'],
                'fact'      : fact['text'],
                'from_web'  : False,
            }

        # Wikipedia fallback
        if use_wikipedia:
            fetched = self.learn_from_wikipedia(expanded)
            if fetched:
                # Rebuild and retry
                results = self.retrieve(expanded, top_k=1)
                if results:
                    score, fact = results[0]
                    return {
                        'answer'    : fact['answer'],
                        'confidence': round(score, 4),
                        'source'    : 'wikipedia (just learned)',
                        'fact'      : fact['text'],
                        'from_web'  : True,
                    }

        # Best effort
        if results:
            score, fact = results[0]
            return {
                'answer'    : f"Not sure. Closest: {fact['answer']}",
                'confidence': round(score, 4),
                'source'    : fact['category'],
                'fact'      : fact['text'],
                'from_web'  : False,
            }

        return {
            'answer'    : "I don't know yet — no relevant facts found.",
            'confidence': 0.0,
            'source'    : None,
            'fact'      : None,
            'from_web'  : False,
        }

    def stats(self) -> dict:
        return {
            'total_facts'  : self.n_added,
            'vocab_size'   : self.text_enc.vocab_size,
            'n_queries'    : self.n_queries,
            'index_built'  : not self._dirty,
            'categories'   : {
                c: self._raw_cats.count(c)
                for c in set(self._raw_cats)
            },
        }

    def __repr__(self) -> str:
        return (f"DynamicKnowledgeBase("
                f"facts={self.n_added}, "
                f"vocab={self.text_enc.vocab_size})")
