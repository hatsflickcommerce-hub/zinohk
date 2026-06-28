"""
zinohk.encoding.decoder
========================
SpikeDecoder — converts sparse node activations
into word probability distributions.

This closes the final gap between ZINOHK and LLMs:
  encode  : text → embeddings → sparse nodes  (done)
  decode  : sparse nodes → word probs → text  (this file)

Architecture:
  sparse_activations (N_HIDDEN,)
      → decoder_weights (N_HIDDEN, vocab_size)
      → logits (vocab_size,)
      → softmax → word probabilities
      → argmax / beam search → next word

Learning:
  Decoder weights update via local Hebbian rule.
  No backprop. No global gradient.
  "Which sparse nodes were active when this word appeared?"
  → strengthen those node→word connections.
"""

import numpy as np
from typing import List, Optional, Tuple
from zinohk.encoding.vocab import Vocabulary, START_IDX, END_IDX, UNK_IDX


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    x = x / (temperature + 1e-8)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-8)


class SpikeDecoder:
    """
    Converts sparse node activations to word probabilities.

    Parameters
    ----------
    n_hidden  : number of sparse hidden nodes (input dim)
    vocab     : Vocabulary instance
    lr        : Hebbian learning rate
    temperature: softmax temperature (higher = more random)

    Example
    -------
    >>> dec = SpikeDecoder(n_hidden=512, vocab=vocab)
    >>> h   = sparse_activations  # (512,)
    >>> probs = dec.forward(h)    # (vocab_size,)
    >>> word_idx = np.argmax(probs)
    >>> word = vocab.idx2word[word_idx]
    """

    def __init__(
        self,
        n_hidden:    int,
        vocab:       Vocabulary,
        lr:          float = 0.01,
        temperature: float = 1.0,
    ):
        self.n_hidden    = n_hidden
        self.vocab       = vocab
        self.vocab_size  = len(vocab)
        self.lr          = lr
        self.temperature = temperature

        # Decoder weight matrix: (n_hidden, vocab_size)
        # Each row = one hidden node's contribution to each word
        self.W = np.random.uniform(
            0.0, 0.1,
            (n_hidden, self.vocab_size)
        ).astype(np.float32)

        self.n_updates = 0

    def forward(self, h: np.ndarray) -> np.ndarray:
        """
        Convert sparse activation to word probabilities.

        Parameters
        ----------
        h : sparse activation vector (n_hidden,)

        Returns
        -------
        probs : word probability distribution (vocab_size,)
        """
        assert h.shape == (self.n_hidden,), \
            f"Expected ({self.n_hidden},) got {h.shape}"

        logits = h @ self.W                        # (vocab_size,)
        probs  = softmax(logits, self.temperature)
        return probs

    def predict_word(self, h: np.ndarray) -> Tuple[int, float]:
        """
        Predict the most likely next word.

        Returns
        -------
        (word_idx, confidence)
        """
        probs    = self.forward(h)
        word_idx = int(np.argmax(probs))
        conf     = float(probs[word_idx])
        return word_idx, conf

    def hebbian_update(
        self,
        h:        np.ndarray,
        word_idx: int,
    ) -> None:
        """
        Local Hebbian update: strengthen h→word connection.

        When node i is active and word w appears:
          W[i, w] += lr * h[i]

        No backprop. No global gradient.

        Parameters
        ----------
        h        : sparse activation that produced the word
        word_idx : the correct word index
        """
        self.n_updates += 1

        # Strengthen connections to correct word
        self.W[:, word_idx] += self.lr * h

        # Slight decay on all other words (competition)
        decay_mask = np.ones(self.vocab_size, dtype=np.float32)
        decay_mask[word_idx] = 0.0
        self.W -= self.lr * 0.001 * np.outer(h, decay_mask)

        # Clip weights
        self.W[:] = np.clip(self.W, 0.0, 5.0)

    def train_on_pair(
        self,
        h:    np.ndarray,
        text: str,
    ) -> float:
        """
        Train decoder on one (activation, text) pair.

        For each word in text:
          - predict next word from h
          - update weights toward correct word

        Parameters
        ----------
        h    : sparse activation vector
        text : target text to learn

        Returns
        -------
        float : fraction of words predicted correctly
        """
        word_ids = self.vocab.encode(text, add_special=False)
        if not word_ids:
            return 0.0

        correct = 0
        for word_idx in word_ids:
            pred_idx, _ = self.predict_word(h)
            if pred_idx == word_idx:
                correct += 1
            self.hebbian_update(h, word_idx)

        return correct / len(word_ids)

    def generate(
        self,
        h:        np.ndarray,
        max_len:  int   = 20,
        stop_at_end: bool = True,
    ) -> str:
        """
        Generate text from sparse activation.

        Greedy decoding — always pick highest prob word.

        Parameters
        ----------
        h           : sparse activation vector
        max_len     : maximum words to generate
        stop_at_end : stop when END token predicted

        Returns
        -------
        str : generated text
        """
        words = []
        for _ in range(max_len):
            word_idx, conf = self.predict_word(h)

            if stop_at_end and word_idx == END_IDX:
                break

            word = self.vocab.idx2word.get(word_idx, '<UNK>')
            if word not in ['<PAD>', '<START>', '<END>', '<UNK>']:
                words.append(word)

        return ' '.join(words)

    def stats(self) -> dict:
        return {
            "n_hidden"   : self.n_hidden,
            "vocab_size" : self.vocab_size,
            "n_updates"  : self.n_updates,
            "weight_norm": round(float(np.linalg.norm(self.W)), 4),
            "temperature": self.temperature,
        }

    def __repr__(self) -> str:
        return (f"SpikeDecoder(hidden={self.n_hidden}, "
                f"vocab={self.vocab_size}, "
                f"updates={self.n_updates})")


class NGramDecoder:
    """
    Next-word predictor using Hebbian n-gram learning.

    Learns: given current word → predict next word.
    No backprop. Pure local Hebbian update.

    Parameters
    ----------
    vocab       : Vocabulary instance
    lr          : Hebbian learning rate
    temperature : sampling temperature (lower = more focused)
    """

    def __init__(
        self,
        vocab:       Vocabulary,
        lr:          float = 0.1,
        temperature: float = 0.8,
    ):
        self.vocab       = vocab
        self.V           = len(vocab)
        self.lr          = lr
        self.temperature = temperature

        # Transition matrix: W[i,j] = strength of word_i → word_j
        self.W = np.random.uniform(
            0.0, 0.05, (self.V, self.V)
        ).astype(np.float32)

        self.n_updates = 0

    def train(self, corpus: List[str], epochs: int = 300) -> None:
        """
        Train on corpus using Hebbian next-word prediction.

        Parameters
        ----------
        corpus : list of text sentences
        epochs : number of training epochs
        """
        for epoch in range(epochs):
            for text in corpus:
                ids = self.vocab.encode(text, add_special=True)
                for i in range(len(ids) - 1):
                    curr = ids[i]
                    nxt  = ids[i + 1]
                    # Strengthen curr→next
                    self.W[curr, nxt] += self.lr
                    # Small decay everywhere
                    self.W[curr]      -= self.lr * 0.01
                    self.W[:]          = np.clip(self.W, 0.0, 20.0)
                    self.n_updates    += 1

    def generate(
        self,
        seed:    str,
        max_len: int = 15,
    ) -> str:
        """
        Generate text from a seed word.

        Parameters
        ----------
        seed    : starting word
        max_len : maximum words to generate

        Returns
        -------
        str : generated text
        """
        curr_idx = self.vocab.word2idx.get(
            seed.lower(), START_IDX)
        words    = [seed] if seed.lower() in self.vocab else []

        for _ in range(max_len):
            probs    = softmax(self.W[curr_idx], self.temperature)
            next_idx = int(np.argmax(probs))
            word     = self.vocab.idx2word.get(next_idx, '')

            if word in [
                '<END>', '<PAD>', '<UNK>', '<START>'
            ]:
                break

            words.append(word)
            curr_idx = next_idx

            # Stop if we've repeated 3 times
            if len(words) >= 3 and len(set(words[-3:])) == 1:
                break

        return ' '.join(words)

    def perplexity(self, text: str) -> float:
        """
        Measure how well the model predicts this text.
        Lower = better.
        """
        ids  = self.vocab.encode(text, add_special=True)
        if len(ids) < 2:
            return float('inf')

        log_prob = 0.0
        for i in range(len(ids) - 1):
            probs     = softmax(self.W[ids[i]], self.temperature)
            log_prob += np.log(probs[ids[i+1]] + 1e-10)

        return float(np.exp(-log_prob / (len(ids) - 1)))

    def stats(self) -> dict:
        return {
            "vocab_size" : self.V,
            "n_updates"  : self.n_updates,
            "weight_norm": round(float(np.linalg.norm(self.W)), 2),
        }

    def __repr__(self) -> str:
        return (f"NGramDecoder(vocab={self.V}, "
                f"updates={self.n_updates})")
