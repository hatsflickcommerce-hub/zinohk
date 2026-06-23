"""
zinohk.core.node
================
ZNode — the fundamental compute unit of ZINOHK.

Pillar 1: Sparse threshold gate (fire only if input > threshold)
Pillar 5: Local Hebbian weight update (no backprop)
"""

import numpy as np


class ZNode:
    """
    A single ZINOHK neuron node.

    Fires only when weighted input exceeds threshold theta.
    Updates its own weights locally after every spike.
    Self-regulates threshold to maintain target activity rate.

    Parameters
    ----------
    node_id  : unique name for this node
    n_inputs : number of incoming connections
    threshold: firing threshold theta (default 0.5)
    lr       : Hebbian learning rate (default 0.01)
    target_rate : target fraction of steps to fire (default 0.01 = 1%)
    """

    def __init__(
        self,
        node_id: str,
        n_inputs: int,
        threshold: float = 0.1,
        lr: float = 0.01,
        target_rate: float = 0.01,
    ):
        self.node_id     = node_id
        self.n_inputs    = n_inputs
        self.threshold   = threshold
        self.lr          = lr
        self.target_rate = target_rate

        # Weights: small random init
        self.weights = np.random.uniform(0.0, 0.5, n_inputs)

        # History for homeostasis
        self.fire_count = 0
        self.step_count = 0

    def forward(self, inputs: np.ndarray) -> float:
        """
        Sparse threshold forward pass.

        Returns activation if input > threshold, else 0.0
        Updates weights locally if fired (Hebbian rule).

        Parameters
        ----------
        inputs : np.ndarray of shape (n_inputs,)

        Returns
        -------
        float : activation value if fired, 0.0 if silent
        """
        assert inputs.shape == (self.n_inputs,), \
            f"Expected {self.n_inputs} inputs, got {inputs.shape}"

        self.step_count += 1

        # Weighted sum
        activation = float(np.dot(self.weights, inputs))

        # Sparse gate: fire only if above threshold
        if activation > self.threshold:
            self.fire_count += 1

            # Hebbian update: strengthen connections that caused firing
            self.weights += self.lr * inputs
            self.weights  = np.clip(self.weights, -5.0, 5.0)

            # Homeostasis: raise threshold slightly (fired too much)
            self.threshold += 0.01

            return activation

        # Silent: lower threshold slightly (not firing enough)
        self.threshold -= 0.001
        self.threshold  = max(0.01, self.threshold)

        return 0.0

    @property
    def activity_rate(self) -> float:
        """Fraction of steps this node has fired."""
        if self.step_count == 0:
            return 0.0
        return self.fire_count / self.step_count

    def stats(self) -> dict:
        """Return current node state."""
        return {
            "node_id"      : self.node_id,
            "threshold"    : round(self.threshold, 4),
            "activity_rate": round(self.activity_rate, 4),
            "weight_norm"  : round(float(np.linalg.norm(self.weights)), 4),
            "steps"        : self.step_count,
            "fires"        : self.fire_count,
        }

    def __repr__(self) -> str:
        return (
            f"ZNode(id={self.node_id!r}, "
            f"theta={self.threshold:.3f}, "
            f"activity={self.activity_rate:.3f})"
        )
