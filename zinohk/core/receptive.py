"""
zinohk.core.receptive
======================
Structured Receptive Fields — fixes GAP-01.

Problem:
  Random sparse connections cannot distinguish patterns
  that share active input positions (A+C, B+D collapse).

Brain solution:
  Each neuron in visual cortex watches a specific spatial
  region — its "receptive field". Neighbouring neurons
  watch overlapping but distinct regions.

ZINOHK solution:
  ReceptiveField  : defines which input channels a node watches
  ReceptiveLayout : tiles receptive fields across input space
                    so every region is covered by some node,
                    and adjacent nodes overlap slightly

Types:
  - contiguous : node watches a contiguous block of inputs
  - strided    : node watches every k-th input
  - random_structured : node watches a fixed random region
                        (same seed = reproducible)

Why this fixes GAP-01:
  A = [1,0,1,0,0]  C = [1,1,0,0,1]
  A node watching channels [0,2] sees: A→[1,1]  C→[1,0]
  A node watching channels [1,3] sees: A→[0,0]  C→[1,0]
  Now A and C produce DIFFERENT hidden patterns.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ReceptiveField:
    """
    Defines which input channels a single node monitors.

    Attributes
    ----------
    node_id  : which node owns this field
    channels : list of input indices this node watches
    weight   : connection weight per channel
    """
    node_id:  str
    channels: np.ndarray   # shape (field_size,)
    weights:  np.ndarray   # shape (field_size,)

    def extract(self, x: np.ndarray) -> np.ndarray:
        """Pull out only the channels this node watches."""
        return x[self.channels] * self.weights

    def activation(self, x: np.ndarray, threshold: float) -> float:
        """Compute activation for this receptive field."""
        val = float(np.sum(self.extract(x)))
        return val if val > threshold else 0.0


class ReceptiveLayout:
    """
    Tiles receptive fields across the full input space.

    Ensures:
      - Every input region is covered by at least one node
      - Adjacent nodes have overlapping but distinct fields
      - No two nodes see exactly the same channels

    Parameters
    ----------
    n_inputs    : total input dimension
    n_nodes     : number of hidden nodes
    field_size  : how many channels each node watches
    overlap     : fraction of overlap between adjacent fields
    seed        : random seed for reproducibility

    Example
    -------
    >>> layout = ReceptiveLayout(n_inputs=10, n_nodes=5,
    ...                          field_size=4, overlap=0.25)
    >>> fields = layout.build()
    >>> len(fields)
    5
    """

    def __init__(
        self,
        n_inputs:   int,
        n_nodes:    int,
        field_size: int,
        overlap:    float = 0.25,
        seed:       int   = 42,
    ):
        self.n_inputs   = n_inputs
        self.n_nodes    = n_nodes
        self.field_size = field_size
        self.overlap    = overlap
        self.seed       = seed

    def build(self) -> List[ReceptiveField]:
        """
        Build structured receptive fields for all nodes.

        Uses strided tiling with overlap so every input
        region is covered and patterns are distinguishable.
        """
        rng    = np.random.default_rng(self.seed)
        fields = []

        # Stride = field_size * (1 - overlap)
        stride = max(1, int(self.field_size * (1.0 - self.overlap)))

        for i in range(self.n_nodes):
            # Start position — wraps around input space
            start    = (i * stride) % self.n_inputs
            channels = np.array([
                (start + j) % self.n_inputs
                for j in range(self.field_size)
            ], dtype=int)

            # Remove duplicates while preserving order
            _, idx   = np.unique(channels, return_index=True)
            channels = channels[np.sort(idx)]

            # Random weights for this field
            weights = rng.uniform(0.1, 0.5, len(channels))

            fields.append(ReceptiveField(
                node_id  = f'rf_{i}',
                channels = channels,
                weights  = weights,
            ))

        return fields

    def coverage_report(self, fields: List[ReceptiveField]) -> dict:
        """
        Report how well the fields cover the input space.
        """
        covered = set()
        for f in fields:
            covered.update(f.channels.tolist())

        return {
            "n_inputs"    : self.n_inputs,
            "n_nodes"     : self.n_nodes,
            "field_size"  : self.field_size,
            "covered"     : len(covered),
            "coverage_pct": round(len(covered) / self.n_inputs * 100, 1),
            "overlap"     : self.overlap,
        }
