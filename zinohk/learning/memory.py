"""
zinohk.learning.memory
=======================
Complementary Learning System (CLS) — fixes GAP-04.

Catastrophic forgetting problem:
  When a network learns new things, it overwrites old
  knowledge. Pure Hebbian learning has no protection.

Brain solution — two memory systems:
  1. Hippocampus (fast memory):
       Learns quickly from single experiences.
       Small capacity. Replays to cortex during sleep.

  2. Cortex (slow memory):
       Learns slowly via repeated replay.
       Large capacity. Permanent storage.
       Never overwritten by single new experience.

ZINOHK implementation:
  FastMemory  : high lr, small buffer, learns immediately
  SlowMemory  : low lr, large buffer, learns from replay
  MemorySystem: coordinates both, runs replay consolidation

Math:
  fast update : w += lr_fast * delta   (lr_fast = 0.1)
  slow update : w += lr_slow * delta   (lr_slow = 0.001)
  replay      : slow learns from fast buffer samples
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class MemoryTrace:
    """A single experience stored in fast memory."""
    x:     np.ndarray   # input
    label: int          # target
    step:  int          # when it was stored


class FastMemory:
    """
    Hippocampus-like fast memory.

    Learns immediately from new experiences.
    Limited capacity — old memories are replaced.
    Acts as a buffer for slow memory consolidation.

    Parameters
    ----------
    capacity : max number of experiences to hold
    lr       : fast learning rate (high)
    """

    def __init__(self, capacity: int = 200, lr: float = 0.1):
        self.capacity = capacity
        self.lr       = lr
        self.buffer   = deque(maxlen=capacity)
        self.step     = 0

    def store(self, x: np.ndarray, label: int) -> None:
        """Store a new experience."""
        self.step += 1
        self.buffer.append(MemoryTrace(
            x     = x.copy(),
            label = label,
            step  = self.step,
        ))

    def sample(self, n: int) -> List[MemoryTrace]:
        """Sample n random experiences for replay."""
        n = min(n, len(self.buffer))
        if n == 0:
            return []
        idx = np.random.choice(len(self.buffer), size=n, replace=False)
        return [self.buffer[i] for i in idx]

    @property
    def size(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return (f"FastMemory(capacity={self.capacity}, "
                f"stored={self.size})")


class SlowMemory:
    """
    Cortex-like slow memory.

    Learns slowly via repeated replay from fast memory.
    Weights are protected — single experiences cannot
    overwrite accumulated knowledge.

    Parameters
    ----------
    n_inputs  : input dimension
    n_outputs : output dimension
    lr        : slow learning rate (very low)
    """

    def __init__(
        self,
        n_inputs:  int,
        n_outputs: int,
        lr:        float = 0.001,
    ):
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs
        self.lr        = lr

        # Slow weights — initialised small
        self.weights = np.random.uniform(
            0.0, 0.1, (n_outputs, n_inputs))

        self.n_replays = 0

    def predict(self, x: np.ndarray) -> int:
        """Predict label from slow weights."""
        scores = self.weights @ x
        return int(np.argmax(scores))

    def replay_update(self, trace: MemoryTrace) -> float:
        """
        Update slow weights from one replayed experience.

        Uses slow lr — single replay barely changes weights.
        Knowledge accumulates only from REPEATED replay.

        Returns
        -------
        float : prediction error (1.0 if wrong, 0.0 if right)
        """
        self.n_replays += 1
        pred = self.predict(trace.x)
        err  = 1.0 if pred != trace.label else 0.0

        # Strengthen correct output, weaken others
        for j in range(self.n_outputs):
            if j == trace.label:
                self.weights[j] += self.lr * trace.x
            else:
                self.weights[j] -= self.lr * 0.5 * trace.x

        self.weights = np.clip(self.weights, 0.0, 2.0)
        return err

    def __repr__(self) -> str:
        return (f"SlowMemory(inputs={self.n_inputs}, "
                f"outputs={self.n_outputs}, "
                f"replays={self.n_replays})")


class MemorySystem:
    """
    Full Complementary Learning System.

    Coordinates fast + slow memory.
    Runs replay consolidation after each batch.

    Parameters
    ----------
    n_inputs    : input dimension
    n_outputs   : number of classes
    capacity    : fast memory buffer size
    replay_k    : how many traces to replay per step
    lr_fast     : fast memory learning rate
    lr_slow     : slow memory learning rate

    Example
    -------
    >>> ms = MemorySystem(n_inputs=10, n_outputs=2)
    >>> ms.learn(x, label=1)       # fast store + slow replay
    >>> pred = ms.predict(x)       # predict from slow memory
    """

    def __init__(
        self,
        n_inputs:  int,
        n_outputs: int,
        capacity:  int   = 200,
        replay_k:  int   = 10,
        lr_fast:   float = 0.1,
        lr_slow:   float = 0.001,
    ):
        self.fast = FastMemory(capacity=capacity, lr=lr_fast)
        self.slow = SlowMemory(
            n_inputs=n_inputs, n_outputs=n_outputs, lr=lr_slow)
        self.replay_k = replay_k
        self.n_steps  = 0

    def learn(self, x: np.ndarray, label: int) -> None:
        """
        Learn from one experience.

        1. Store in fast memory immediately
        2. Replay k random past experiences to slow memory
        """
        self.n_steps += 1

        # Fast store
        self.fast.store(x, label)

        # Slow replay — consolidate past experiences
        traces = self.fast.sample(self.replay_k)
        for trace in traces:
            self.slow.replay_update(trace)

    def predict(self, x: np.ndarray) -> int:
        """Predict from slow (permanent) memory."""
        return self.slow.predict(x)

    def stats(self) -> dict:
        return {
            "steps"       : self.n_steps,
            "fast_stored" : self.fast.size,
            "slow_replays": self.slow.n_replays,
            "slow_w_norm" : round(float(
                np.linalg.norm(self.slow.weights)), 4),
        }

    def __repr__(self) -> str:
        return (f"MemorySystem(fast={self.fast}, "
                f"slow={self.slow})")
