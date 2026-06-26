"""
zinohk.learning.contrastive
============================
Contrastive Hebbian Learning.

Fixes representational collapse by:
  - Strengthening synapses that led to the winner
  - Weakening synapses that led to losers

This forces output nodes to specialise — each learns
to respond to ONE pattern and suppress others.

Rule:
  winner synapse  : w += lr * pre * post  (potentiate)
  loser  synapses : w -= lr * pre * decay (depress)
"""

import numpy as np
from zinohk.core.synapse import Synapse


def contrastive_update(
    synapses:   dict,
    h_out:      list,
    o_out:      list,
    n_hidden:   int,
    n_output:   int,
    lr:         float = 0.01,
    decay:      float = 0.005,
) -> None:
    """
    Contrastive Hebbian update on hidden->output synapses.

    Parameters
    ----------
    synapses : dict (i,j) -> Synapse
    h_out    : hidden layer outputs  [n_hidden]
    o_out    : output layer outputs  [n_output]
    n_hidden : number of hidden nodes
    n_output : number of output nodes
    lr       : learning rate for winner
    decay    : depression rate for losers
    """
    # Find winner output node
    winner = int(np.argmax(o_out))

    for i in range(n_hidden):
        for j in range(n_output):
            syn = synapses[(i, j)]
            if j == winner:
                # Potentiate — fire together wire together
                delta = lr * h_out[i] * o_out[j]
                syn.weight = float(np.clip(
                    syn.weight + delta, 0.0, 2.0))
            else:
                # Depress — loser connections weaken
                delta = decay * h_out[i]
                syn.weight = float(np.clip(
                    syn.weight - delta, 0.0, 2.0))
