"""
zinohk.encoding.text
=====================
Text encoder — converts raw text into a dense vector
that ZINOHK's spike encoder can process.

Two modes:
  1. TF-IDF (default, no GPU needed, works offline)
  2. Embedding-ready (slot for sentence transformers, Phase 9)

Pipeline:
  raw text → clean → tokenise → TF-IDF vector → normalise → [0,1]

The output vector feeds directly into SpikeEncoder.
"""

import re
import numpy as np
from collections import Counter
from typing import List, Dict, Optional


# Common English stopwords — skip these for better signal
STOPWORDS = {
    'the','a','an','is','it','in','of','to','and','i',
    'this','that','was','for','on','are','with','as','at',
    'be','by','from','or','have','had','not','but','they',
    'he','she','we','you','his','her','their','my','me',
    'him','so','if','do','its','been','all','one','more',
    'has','what','there','who','would','about','no','up',
    'out','were','when','which','will','been','than','then',
    'them','these','some','into','could','after','other',
}


def clean(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenise(text: str) -> List[str]:
    """Clean and split into tokens, removing stopwords."""
    return [t for t in clean(text).split()
            if t not in STOPWORDS and len(t) > 2]


class TFIDFEncoder:
    """
    TF-IDF text encoder — converts sentences to vectors.

    Build vocabulary from a corpus, then encode any
    sentence as a normalised TF-IDF vector in [0, 1].

    Parameters
    ----------
    max_vocab : maximum vocabulary size
    min_freq  : minimum word frequency to include

    Example
    -------
    >>> enc = TFIDFEncoder(max_vocab=500)
    >>> enc.fit(["the sky is blue", "grass is green"])
    >>> vec = enc.encode("sky is blue")
    >>> vec.shape
    (500,)
    """

    def __init__(self, max_vocab: int = 1000, min_freq: int = 2):
        self.max_vocab  = max_vocab
        self.min_freq   = min_freq
        self.vocab:     Dict[str, int] = {}
        self.idf:       np.ndarray     = np.array([])
        self.is_fitted: bool           = False

    def fit(self, corpus: List[str]) -> None:
        """
        Build vocabulary and IDF weights from corpus.

        Parameters
        ----------
        corpus : list of raw text strings
        """
        n_docs = len(corpus)

        # Count word frequencies across all docs
        word_freq  = Counter()
        doc_freq   = Counter()

        for doc in corpus:
            tokens = tokenise(doc)
            word_freq.update(tokens)
            doc_freq.update(set(tokens))   # unique per doc

        # Build vocabulary — top max_vocab by frequency
        vocab_words = [
            w for w, c in word_freq.most_common(self.max_vocab * 2)
            if c >= self.min_freq
        ][:self.max_vocab]

        self.vocab = {w: i for i, w in enumerate(vocab_words)}

        # IDF weights
        self.idf = np.zeros(len(self.vocab))
        for w, i in self.vocab.items():
            df = doc_freq.get(w, 0)
            self.idf[i] = np.log((n_docs + 1) / (df + 1)) + 1.0

        self.is_fitted = True

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text as a normalised TF-IDF vector.

        Parameters
        ----------
        text : raw text string

        Returns
        -------
        np.ndarray of shape (vocab_size,) in [0, 1]
        """
        assert self.is_fitted, "Call fit() first"

        tokens = tokenise(text)
        tf     = Counter(tokens)
        vec    = np.zeros(len(self.vocab), dtype=np.float32)

        for word, count in tf.items():
            if word in self.vocab:
                i      = self.vocab[word]
                vec[i] = count * self.idf[i]

        # Normalise to [0, 1]
        if vec.max() > 0:
            vec = vec / vec.max()

        return vec

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts. Returns (n, vocab_size)."""
        return np.array([self.encode(t) for t in texts])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return (f"TFIDFEncoder(vocab={self.vocab_size}, "
                f"fitted={self.is_fitted})")
