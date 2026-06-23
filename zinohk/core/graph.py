"""
zinohk.core.graph
=================
ZGraph — async node graph engine.

Pillar 2: No fixed layers.
Each node fires when its inputs arrive.
Nodes connected via Synapses.
"""

import numpy as np
from zinohk.core.node    import ZNode
from zinohk.core.synapse import Synapse


class ZGraph:
    """
    Async computation graph of ZNodes connected by Synapses.

    Usage
    -----
    g = ZGraph()
    g.add_node('n0', n_inputs=2)
    g.add_node('n1', n_inputs=1)
    g.add_edge('n0', 'n1')
    outputs = g.forward({'n0': np.array([0.9, 0.8])})
    """

    def __init__(self):
        self.nodes    = {}   # node_id -> ZNode
        self.synapses = {}   # (pre_id, post_id) -> Synapse
        self.edges    = {}   # node_id -> list of downstream node_ids

    # ----------------------------------------------------------
    # Graph construction
    # ----------------------------------------------------------

    def add_node(self, node_id: str, n_inputs: int, **kwargs) -> ZNode:
        """Add a ZNode to the graph."""
        node = ZNode(node_id=node_id, n_inputs=n_inputs, **kwargs)
        self.nodes[node_id] = node
        self.edges[node_id] = []
        return node

    def add_edge(self, pre_id: str, post_id: str, weight: float = 0.1):
        """Connect two nodes with a Synapse."""
        assert pre_id  in self.nodes, f"Node '{pre_id}' not found"
        assert post_id in self.nodes, f"Node '{post_id}' not found"

        synapse = Synapse(pre_id=pre_id, post_id=post_id, weight=weight)
        self.synapses[(pre_id, post_id)] = synapse
        self.edges[pre_id].append(post_id)

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------

    def forward(self, inputs: dict) -> dict:
        """
        Run one forward pass across the graph.

        Parameters
        ----------
        inputs : dict of {node_id: np.ndarray}
                 Only provide inputs for entry-point nodes.

        Returns
        -------
        dict of {node_id: float}
        """
        outputs = {}

        # Process nodes in insertion order (simple async simulation)
        for node_id, node in self.nodes.items():

            # Get input for this node
            if node_id in inputs:
                node_input = inputs[node_id]
            else:
                # Collect outputs from upstream nodes via synapses
                upstream = [
                    pre for (pre, post) in self.synapses
                    if post == node_id
                ]
                if not upstream:
                    outputs[node_id] = 0.0
                    continue

                # Sum weighted upstream outputs
                total = np.zeros(node.n_inputs)
                for i, pre_id in enumerate(upstream[:node.n_inputs]):
                    if pre_id in outputs:
                        syn = self.synapses.get((pre_id, node_id))
                        w   = syn.weight if syn else 1.0
                        total[i] = outputs[pre_id] * w

                node_input = total

            # Fire the node
            out = node.forward(node_input)
            outputs[node_id] = out

            # Hebbian update on all outgoing synapses
            for post_id in self.edges[node_id]:
                key = (node_id, post_id)
                if key in self.synapses and post_id in outputs:
                    self.synapses[key].hebbian_update(
                        pre_fired=out,
                        post_fired=outputs[post_id],
                    )

        return outputs

    # ----------------------------------------------------------
    # Stats
    # ----------------------------------------------------------

    def sparsity(self, outputs: dict) -> float:
        """Fraction of nodes that fired in the last forward pass."""
        if not outputs:
            return 0.0
        fired = sum(1 for v in outputs.values() if v > 0)
        return fired / len(outputs)

    def summary(self):
        """Print graph summary."""
        print(f"ZGraph | nodes={len(self.nodes)} | "
              f"synapses={len(self.synapses)}")
        for nid, node in self.nodes.items():
            print(f"  {node}")
