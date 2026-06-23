"""
zinohk.core.synapse
===================
Synapse — connection between two ZNodes.

Pillar 5: Local learning only.
  - Hebbian rule : fire together, wire together
  - STDP         : timing-based strengthen or weaken
"""

import numpy as np


class Synapse:
    """
    A single synaptic connection between two nodes.

    Holds one weight value.
    Updates itself using only local information — no global gradient.

    Parameters
    ----------
    pre_id  : node_id of the sending node
    post_id : node_id of the receiving node
    weight  : initial weight (default 0.1)
    lr      : learning rate (default 0.01)
    """

    def __init__(
        self,
        pre_id:  str,
        post_id: str,
        weight:  float = 0.1,
        lr:      float = 0.01,
    ):
        self.pre_id  = pre_id
        self.post_id = post_id
        self.weight  = float(weight)
        self.lr      = lr

    def hebbian_update(
        self,
        pre_fired:  float,
        post_fired: float,
    ) -> float:
        """
        Hebbian learning rule.

        Neurons that fire together wire together.
        Delta_w = lr * pre * post

        Parameters
        ----------
        pre_fired  : output of pre-synaptic node (0.0 if silent)
        post_fired : output of post-synaptic node (0.0 if silent)

        Returns
        -------
        float : weight change applied
        """
        delta = self.lr * pre_fired * post_fired
        self.weight = float(np.clip(self.weight + delta, -5.0, 5.0))
        return delta

    def stdp_update(
        self,
        pre_time:  float,
        post_time: float,
        a_plus:    float = 0.01,
        a_minus:   float = 0.012,
        tau:       float = 20.0,
    ) -> float:
        """
        Spike-Timing Dependent Plasticity (STDP).

        Pre fires BEFORE post → strengthen (potentiation)
          Delta_w = +a_plus  * exp(-delta_t / tau)

        Post fires BEFORE pre → weaken (depression)
          Delta_w = -a_minus * exp(-delta_t / tau)

        Parameters
        ----------
        pre_time  : spike time of pre node (ms)
        post_time : spike time of post node (ms)

        Returns
        -------
        float : weight change applied
        """
        delta_t = post_time - pre_time

        if delta_t > 0:
            # Pre fired first → potentiation
            delta = a_plus * np.exp(-delta_t / tau)
        else:
            # Post fired first → depression
            delta = -a_minus * np.exp(delta_t / tau)

        self.weight = float(np.clip(self.weight + delta, -5.0, 5.0))
        return float(delta)

    def __repr__(self) -> str:
        return (
            f"Synapse({self.pre_id!r} → {self.post_id!r}, "
            f"w={self.weight:.4f})"
        )
