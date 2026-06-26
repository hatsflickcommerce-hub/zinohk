"""
zinohk.encoding.spike
======================
Spike Temporal Encoding — Pillar 3 of ZINOHK.

Converts float signals into spike timing patterns.

Key idea:
  Strong signal → fires early  (small spike_time)
  Weak signal   → fires late   (large spike_time)
  No signal     → silent       (spike_time = None)

Why this matters:
  - Replaces 32-bit floats with timing events
  - Preserves signal ORDER (fixes word order loss)
  - Binary spikes = 1000x less energy than float multiply
  - Silent = zero cost (true sparse compute)

Math:
  spike_time(x) = T_max * (1 - x)     for x in [0, 1]
  decode(t)     = 1 - (t / T_max)

  Two spikes close together = strong correlation
  Two spikes far apart      = weak correlation
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Spike:
    """
    A single spike event.

    Attributes
    ----------
    channel    : which input dimension fired
    spike_time : when it fired (ms) — smaller = stronger signal
    strength   : original signal strength [0, 1]
    """
    channel:    int
    spike_time: float
    strength:   float


class SpikeEncoder:
    """
    Converts a float vector into a list of Spike events.

    Parameters
    ----------
    T_max     : time window in ms (default 20ms)
    threshold : minimum signal to generate a spike (default 0.05)
                below this → silent → zero cost

    Example
    -------
    >>> enc = SpikeEncoder(T_max=20.0, threshold=0.05)
    >>> spikes = enc.encode(np.array([0.9, 0.1, 0.0, 0.5]))
    >>> len(spikes)   # 0.0 is silent
    3
    """

    def __init__(self, T_max: float = 20.0, threshold: float = 0.05):
        self.T_max     = T_max
        self.threshold = threshold

    def encode(self, x: np.ndarray) -> List[Spike]:
        """
        Convert float vector to spike events.

        Parameters
        ----------
        x : np.ndarray, values in [0, 1]

        Returns
        -------
        List[Spike] sorted by spike_time (early spikes first)
        """
        spikes = []
        for i, val in enumerate(x):
            if val > self.threshold:
                # Strong signal → early spike
                t = self.T_max * (1.0 - float(val))
                spikes.append(Spike(
                    channel    = i,
                    spike_time = t,
                    strength   = float(val),
                ))

        # Sort by spike time — early (strong) spikes first
        spikes.sort(key=lambda s: s.spike_time)
        return spikes

    def decode(self, spikes: List[Spike], n_channels: int) -> np.ndarray:
        """
        Reconstruct float vector from spike events.

        Parameters
        ----------
        spikes     : list of Spike events
        n_channels : total number of input dimensions

        Returns
        -------
        np.ndarray of shape (n_channels,)
        """
        x = np.zeros(n_channels, dtype=np.float32)
        for spike in spikes:
            x[spike.channel] = spike.strength
        return x

    def spike_overlap(
        self,
        spikes_a: List[Spike],
        spikes_b: List[Spike],
        tau: float = 5.0,
    ) -> float:
        """
        Measure temporal correlation between two spike trains.

        Spikes that fire close together in time are correlated.
        This replaces dot-product similarity for spike signals.

        Parameters
        ----------
        spikes_a : first spike train
        spikes_b : second spike train
        tau      : time constant for correlation window (ms)

        Returns
        -------
        float : correlation score [0, 1]
        """
        if not spikes_a or not spikes_b:
            return 0.0

        score = 0.0
        for sa in spikes_a:
            for sb in spikes_b:
                if sa.channel == sb.channel:
                    dt     = abs(sa.spike_time - sb.spike_time)
                    score += np.exp(-dt / tau)

        # Normalise
        max_possible = min(len(spikes_a), len(spikes_b))
        return float(score / max_possible) if max_possible > 0 else 0.0


class SpikeDecoder:
    """
    Converts spike timing back to float signal.

    Used at the output layer to produce a final prediction.
    """

    def __init__(self, T_max: float = 20.0):
        self.T_max = T_max

    def decode_time(self, spike_time: float) -> float:
        """Convert spike time back to signal strength."""
        return 1.0 - (spike_time / self.T_max)

    def first_spike_decode(
        self,
        spikes: List[Spike],
        n_channels: int,
    ) -> np.ndarray:
        """
        Decode using first spike per channel only.
        First spike = strongest signal = most important feature.
        """
        x    = np.zeros(n_channels, dtype=np.float32)
        seen = set()
        for spike in spikes:   # already sorted by time
            if spike.channel not in seen:
                x[spike.channel] = self.decode_time(spike.spike_time)
                seen.add(spike.channel)
        return x
