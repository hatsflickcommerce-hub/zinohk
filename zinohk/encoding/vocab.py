"""
zinohk.encoding.vocab
======================
Vocabulary builder for ZINOHK text generation.

Converts a text corpus into a word vocabulary with:
  - word → index mapping
  - index → word mapping
  - frequency filtering
  - special tokens: <PAD>, <UNK>, <START>, <END>

Used by the decoder to convert spike patterns
back into word sequences.
"""

import re
from collections import Counter
from typing import List, Dict, Optional


# Special tokens
PAD_TOKEN   = '<PAD>'
UNK_TOKEN   = '<UNK>'
START_TOKEN = '<START>'
END_TOKEN   = '<END>'

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]

PAD_IDX   = 0
UNK_IDX   = 1
START_IDX = 2
END_IDX   = 3


def tokenise(text: str) -> List[str]:
    """Simple word tokeniser — lowercase, split on punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\'\-]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()


class Vocabulary:
    """
    Word vocabulary for ZINOHK decoder.

    Parameters
    ----------
    max_size : maximum vocabulary size (default 10000)
    min_freq : minimum word frequency to include

    Example
    -------
    >>> vocab = Vocabulary(max_size=1000)
    >>> vocab.build(["the sky is blue", "the grass is green"])
    >>> vocab.encode("the sky")
    [2, 5, 6]  # START + word indices
    """

    def __init__(self, max_size: int = 10000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.freq:     Dict[str, int] = {}
        self.is_built: bool           = False

    def build(self, corpus: List[str]) -> None:
        """
        Build vocabulary from a list of sentences.

        Parameters
        ----------
        corpus : list of raw text strings
        """
        # Count word frequencies
        counter = Counter()
        for text in corpus:
            counter.update(tokenise(text))

        self.freq = dict(counter)

        # Start with special tokens
        self.word2idx = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.idx2word = {i: t for i, t in enumerate(SPECIAL_TOKENS)}

        # Add top words by frequency
        top_words = [
            w for w, c in counter.most_common(self.max_size * 2)
            if c >= self.min_freq and w not in self.word2idx
        ][:self.max_size - len(SPECIAL_TOKENS)]

        for word in top_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word

        self.is_built = True
        print(f"✅ Vocabulary built: {len(self.word2idx):,} words "
              f"(min_freq={self.min_freq})")

    def encode(
        self,
        text:        str,
        add_special: bool = True,
    ) -> List[int]:
        """
        Convert text to list of word indices.

        Parameters
        ----------
        text        : raw text string
        add_special : wrap with START/END tokens

        Returns
        -------
        List[int] of word indices
        """
        assert self.is_built, "Call build() first"
        tokens = tokenise(text)
        ids    = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        if add_special:
            ids = [START_IDX] + ids + [END_IDX]
        return ids

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Convert list of indices back to text.

        Parameters
        ----------
        indices      : list of word indices
        skip_special : remove special tokens from output
        """
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, UNK_TOKEN)
            if skip_special and word in SPECIAL_TOKENS:
                continue
            if word == END_TOKEN:
                break
            words.append(word)
        return ' '.join(words)

    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __repr__(self) -> str:
        return (f"Vocabulary(size={len(self.word2idx)}, "
                f"built={self.is_built})")
