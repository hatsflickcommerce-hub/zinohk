"""
zinohk.knowledge.smart_store
=============================
SmartKnowledgeBase — auto-switches to FAISS at scale.

Under FAISS_THRESHOLD facts  → uses numpy linear scan (simple)
Over  FAISS_THRESHOLD facts  → switches to FAISS (fast)

This gives the best of both worlds:
  - Simple for small KB (no FAISS overhead)
  - Blazing fast for large KB (millisecond search)

Drop-in replacement for DynamicKnowledgeBase.
Same API. Automatic scaling.
"""

import numpy as np
import time
from typing import List, Optional, Tuple
from zinohk.knowledge.faiss_store import FAISSStore
from zinohk.knowledge.retriever   import (
    normalise_query, smart_fetch,
    extract_entity, optimise_for_wikipedia,
)

FAISS_THRESHOLD = 1000   # switch to FAISS after this many facts


class SmartKnowledgeBase:
    """
    Auto-scaling knowledge base.

    Starts simple, upgrades to FAISS automatically.
    Learns from any source. Grows continuously.

    Parameters
    ----------
    encoder        : sentence encoder (SentenceTransformer or TFIDFEncoder)
    dim            : embedding dimension
    faiss_threshold: switch to FAISS after this many facts
    vocab_size     : TF-IDF vocab size (if no sentence encoder)

    Example
    -------
    >>> kb = SmartKnowledgeBase(encoder=model, dim=384)
    >>> kb.learn("Paris is capital of France", "Paris", "geo")
    >>> r = kb.ask("Capital of France?")
    >>> r['answer']
    'Paris'
    """

    def __init__(
        self,
        encoder         = None,
        dim:            int   = 384,
        faiss_threshold: int  = FAISS_THRESHOLD,
        vocab_size:      int  = 2000,
    ):
        self.encoder          = encoder
        self.dim              = dim
        self.faiss_threshold  = faiss_threshold

        # Raw storage
        self._texts:    List[str] = []
        self._answers:  List[str] = []
        self._cats:     List[str] = []
        self._vectors             = None   # (N, dim) numpy array

        # FAISS store — built when threshold reached
        self._faiss: Optional[FAISSStore] = None
        self._using_faiss = False

        # Fallback TF-IDF encoder if no sentence encoder given
        if encoder is None:
            from zinohk.encoding.text import TFIDFEncoder
            self._tfidf = TFIDFEncoder(
                max_vocab=vocab_size, min_freq=1)
            self._tfidf_fitted = False
        else:
            self._tfidf        = None
            self._tfidf_fitted = True

        self.n_added   = 0
        self.n_queries = 0

    # ---------------------------------------------------------------- #
    # Encoding
    # ---------------------------------------------------------------- #

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to vectors using available encoder."""
        if self.encoder is not None:
            return self.encoder.encode(
                texts,
                convert_to_numpy=True,
                batch_size=64,
                show_progress_bar=len(texts) > 500,
            )
        else:
            # TF-IDF fallback
            if not self._tfidf_fitted:
                self._tfidf.fit(texts)
                self._tfidf_fitted = True
            return np.array([self._tfidf.encode(t) for t in texts])

    def _encode_one(self, text: str) -> np.ndarray:
        """Encode a single text."""
        if self.encoder is not None:
            return self.encoder.encode(
                [text], convert_to_numpy=True)[0]
        else:
            if not self._tfidf_fitted:
                self._tfidf.fit([text])
                self._tfidf_fitted = True
            return self._tfidf.encode(text)

    # ---------------------------------------------------------------- #
    # Learning
    # ---------------------------------------------------------------- #

    def learn(
        self,
        text:     str,
        answer:   str,
        category: str = 'general',
    ) -> None:
        """Learn one fact from any source."""
        if text in self._texts:
            return

        vec = self._encode_one(text)

        self._texts.append(text)
        self._answers.append(answer)
        self._cats.append(category)

        if self._vectors is None:
            self._vectors = vec.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, vec])

        # Add to FAISS if already built
        if self._using_faiss:
            self._faiss.add(vec, text, answer, category)

        self.n_added += 1

        # Auto-upgrade to FAISS at threshold
        if (not self._using_faiss and
                self.n_added >= self.faiss_threshold):
            self._upgrade_to_faiss()

    def learn_many(
        self,
        facts: List[Tuple[str, str, str]],
    ) -> None:
        """Learn many facts at once — batch encoded."""
        new_facts = [(t, a, c) for t, a, c in facts
                     if t not in self._texts]
        if not new_facts:
            return

        texts   = [f[0] for f in new_facts]
        answers = [f[1] for f in new_facts]
        cats    = [f[2] for f in new_facts]

        t0   = time.time()
        vecs = self._encode(texts)
        t1   = time.time()

        self._texts.extend(texts)
        self._answers.extend(answers)
        self._cats.extend(cats)

        self._vectors = (vecs if self._vectors is None
                         else np.vstack([self._vectors, vecs]))
        self.n_added += len(new_facts)

        if self._using_faiss:
            self._faiss.add_batch(vecs, texts, answers, cats)
        elif self.n_added >= self.faiss_threshold:
            self._upgrade_to_faiss()

        print(f"✅ Learned {len(new_facts):,} facts | "
              f"total={self.n_added:,} | "
              f"encoded in {t1-t0:.1f}s | "
              f"mode={'FAISS' if self._using_faiss else 'numpy'}")

    def learn_from_wikipedia(self, query: str) -> Optional[str]:
        """Fetch and learn a fact from Wikipedia."""
        import re
        text = smart_fetch(query, sentences=3)
        if not text:
            return None

        q_words = set(re.findall(r"\b\w+\b", query.lower())) - {
            "what","is","the","who","how","when","where",
            "was","did","a","an","of","in","on","does"}
        sents = [s.strip() for s in
                 re.split(r"(?<=[.!?])\s+", text)
                 if len(s.strip()) > 15]
        scored = []
        for s in sents:
            sw = set(re.findall(r"\b\w+\b", s.lower()))
            scored.append((len(q_words & sw), len(s), s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        answer = scored[0][2] if scored else text[:200]

        self.learn(
            query + " " + query + " " + answer,
            answer,
            "wikipedia"
        )
        return answer

    # ---------------------------------------------------------------- #
    # Upgrade to FAISS
    # ---------------------------------------------------------------- #

    def _upgrade_to_faiss(self) -> None:
        """Migrate from numpy to FAISS index."""
        print(f"⚡ Upgrading to FAISS at {self.n_added:,} facts...")
        t0 = time.time()

        self._faiss = FAISSStore(dim=self._vectors.shape[1])
        self._faiss.add_batch(
            self._vectors,
            self._texts,
            self._answers,
            self._cats,
        )
        self._using_faiss = True

        print(f"✅ FAISS index built in {time.time()-t0:.2f}s")

    # ---------------------------------------------------------------- #
    # Retrieval
    # ---------------------------------------------------------------- #

    def retrieve(
        self,
        query:    str,
        top_k:   int = 3,
        category: Optional[str] = None,
    ) -> List[Tuple[float, dict]]:
        """Retrieve most relevant facts."""
        self.n_queries += 1

        if self.n_added == 0:
            return []

        q_vec = self._encode_one(query)

        if self._using_faiss:
            raw = self._faiss.search(q_vec, top_k, category)
            return [(score, {
                'text'    : fact.text,
                'answer'  : fact.answer,
                'category': fact.category,
            }) for score, fact in raw]

        # Numpy cosine search
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(
            q_vec.reshape(1, -1), self._vectors)[0]

        top_idx = np.argsort(sims)[::-1]
        results = []
        for idx in top_idx:
            if category and self._cats[idx] != category:
                continue
            results.append((float(sims[idx]), {
                'text'    : self._texts[idx],
                'answer'  : self._answers[idx],
                'category': self._cats[idx],
            }))
            if len(results) >= top_k:
                break
        return results

    def ask(
        self,
        question:             str,
        confidence_threshold: float = 0.35,
        use_wikipedia:        bool  = True,
    ) -> dict:
        """Answer a question — local first, Wikipedia fallback."""
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

        if use_wikipedia:
            fetched = self.learn_from_wikipedia(expanded)
            if fetched:
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

        if results:
            score, fact = results[0]
            return {
                'answer'    : f"Not sure. Best: {fact['answer']}",
                'confidence': round(score, 4),
                'source'    : fact['category'],
                'fact'      : fact['text'],
                'from_web'  : False,
            }

        return {
            'answer'    : "I don't know yet.",
            'confidence': 0.0,
            'source'    : None,
            'fact'      : None,
            'from_web'  : False,
        }

    def stats(self) -> dict:
        cats = {}
        for c in self._cats:
            cats[c] = cats.get(c, 0) + 1
        return {
            'total_facts' : self.n_added,
            'n_queries'   : self.n_queries,
            'mode'        : 'FAISS' if self._using_faiss else 'numpy',
            'categories'  : cats,
        }

    def save(self, path: str) -> None:
        """Save to disk."""
        import pickle, os
        os.makedirs(path, exist_ok=True)
        if self._using_faiss:
            self._faiss.save(f"{path}/faiss")
        with open(f"{path}/meta.pkl", 'wb') as f:
            pickle.dump({
                'texts'  : self._texts,
                'answers': self._answers,
                'cats'   : self._cats,
                'n_added': self.n_added,
            }, f)
        print(f"✅ Saved to {path}/")

    def __repr__(self) -> str:
        return (f"SmartKnowledgeBase("
                f"facts={self.n_added}, "
                f"mode={'FAISS' if self._using_faiss else 'numpy'})")
