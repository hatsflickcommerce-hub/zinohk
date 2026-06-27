"""
zinohk.feedback.predictor
==========================
Predictive Feedback Loop — Pillar 4 of ZINOHK.

How it works:
  1. High-level node maintains a prediction of what
     low-level input should look like
  2. Low-level node computes prediction error:
       error = actual_input - prediction
  3. Only fires if |error| > epsilon
     (expected input = silent = zero cost)
  4. High-level node updates prediction from error

Why this matters:
  - Familiar inputs cost almost nothing
  - Novel/surprising inputs trigger deep processing
  - Model gets faster and cheaper as it learns
  - Implements biological predictive coding

Math:
  prediction(t) = EMA of past inputs
  error(t)      = actual(t) - prediction(t)
  fire if |error(t)| > epsilon
  prediction(t+1) = prediction(t) + lr * error(t)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PredictionState:
    """Holds the running prediction for one node."""
    prediction: np.ndarray
    error:      np.ndarray
    n_updates:  int = 0
    n_silent:   int = 0   # times input matched prediction
    n_fired:    int = 0   # times error exceeded threshold


class PredictiveNode:
    """
    A node that uses top-down prediction to gate its output.

    Only fires when incoming signal differs from prediction.
    Updates prediction after every step.

    Parameters
    ----------
    node_id   : unique name
    n_inputs  : input dimension
    epsilon   : error threshold — below this = silent
    lr        : prediction learning rate
    decay     : how fast prediction forgets old inputs

    Example
    -------
    >>> pnode = PredictiveNode('p0', n_inputs=4, epsilon=0.1)
    >>> # First input — no prediction yet, fires fully
    >>> err = pnode.forward(np.array([0.8, 0.2, 0.6, 0.1]))
    >>> # Same input again — prediction matches, mostly silent
    >>> err2 = pnode.forward(np.array([0.8, 0.2, 0.6, 0.1]))
    >>> np.sum(np.abs(err2)) < np.sum(np.abs(err))
    True
    """

    def __init__(
        self,
        node_id:  str,
        n_inputs: int,
        epsilon:  float = 0.1,
        lr:       float = 0.2,
        decay:    float = 0.95,
    ):
        self.node_id  = node_id
        self.n_inputs = n_inputs
        self.epsilon  = epsilon
        self.lr       = lr
        self.decay    = decay

        # Start with zero prediction
        self.prediction = np.zeros(n_inputs, dtype=np.float32)

        # Stats
        self.n_updates = 0
        self.n_silent  = 0
        self.n_fired   = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute prediction error and update prediction.

        Parameters
        ----------
        x : np.ndarray of shape (n_inputs,)
            Incoming signal (normalised to [0,1])

        Returns
        -------
        np.ndarray : error signal (zeros where predicted correctly)
                     Only non-zero where surprise exceeds epsilon.
        """
        assert x.shape == (self.n_inputs,), \
            f"Expected ({self.n_inputs},) got {x.shape}"

        self.n_updates += 1

        # Compute prediction error
        error = x - self.prediction

        # Gate: only pass error where surprise exceeds epsilon
        mask        = np.abs(error) > self.epsilon
        gated_error = error * mask

        # Track firing vs silence
        if mask.any():
            self.n_fired += 1
        else:
            self.n_silent += 1

        # Update prediction (exponential moving average)
        self.prediction = (
            self.decay * self.prediction +
            self.lr    * x
        )
        self.prediction = np.clip(self.prediction, 0.0, 1.0)

        return gated_error

    @property
    def silence_rate(self) -> float:
        """Fraction of inputs that were fully predicted."""
        if self.n_updates == 0:
            return 0.0
        return self.n_silent / self.n_updates

    @property
    def surprise_rate(self) -> float:
        """Fraction of inputs that exceeded prediction."""
        if self.n_updates == 0:
            return 0.0
        return self.n_fired / self.n_updates

    def stats(self) -> dict:
        return {
            "node_id"     : self.node_id,
            "n_updates"   : self.n_updates,
            "silence_rate": round(self.silence_rate, 4),
            "surprise_rate": round(self.surprise_rate, 4),
            "pred_norm"   : round(float(np.linalg.norm(
                                self.prediction)), 4),
        }

    def __repr__(self) -> str:
        return (
            f"PredictiveNode(id={self.node_id!r}, "
            f"epsilon={self.epsilon}, "
            f"silence={self.silence_rate:.2f})"
        )
