"""
zinohk.core.inhibition
======================
InhibitoryNode — suppresses output when too many inputs fire.

This is how the brain solves XOR:
  - When both inputs fire together → inhibitor fires → output suppressed
  - When only one input fires → inhibitor silent → output passes through

Pillar 1: Sparse activation via active suppression.
"""

import numpy as np
from zinohk.core.node import ZNode


class InhibitoryNode:
    """
    An inhibitory interneuron.

    Fires when its inputs are BOTH strong (above threshold).
    When it fires, it sends a negative signal to suppress downstream nodes.

    Parameters
    ----------
    node_id   : unique name
    n_inputs  : number of incoming connections
    threshold : fire only when combined input exceeds this
    strength  : how strongly it suppresses downstream (negative weight)
    """

    def __init__(
        self,
        node_id:   str,
        n_inputs:  int,
        threshold: float = 0.4,
        strength:  float = 2.0,
    ):
        self.node_id   = node_id
        self.n_inputs  = n_inputs
        self.threshold = threshold
        self.strength  = strength

        self.fire_count = 0
        self.step_count = 0

    def forward(self, inputs: np.ndarray) -> float:
        """
        Fire a negative suppression signal if inputs are all strong.

        Returns
        -------
        float : negative suppression value if fired, 0.0 if silent
        """
        assert inputs.shape == (self.n_inputs,), \
            f"Expected {self.n_inputs} inputs, got {inputs.shape}"

        self.step_count += 1

        # Fire only when ALL inputs are above threshold (both active)
        if np.all(inputs > self.threshold):
            self.fire_count += 1
            return -self.strength   # negative = suppression

        return 0.0

    @property
    def activity_rate(self) -> float:
        if self.step_count == 0:
            return 0.0
        return self.fire_count / self.step_count

    def stats(self) -> dict:
        return {
            "node_id"      : self.node_id,
            "type"         : "inhibitory",
            "threshold"    : round(self.threshold, 4),
            "strength"     : self.strength,
            "activity_rate": round(self.activity_rate, 4),
            "fires"        : self.fire_count,
            "steps"        : self.step_count,
        }

    def __repr__(self) -> str:
        return (
            f"InhibitoryNode(id={self.node_id!r}, "
            f"threshold={self.threshold:.3f}, "
            f"strength={self.strength:.3f})"
        )
