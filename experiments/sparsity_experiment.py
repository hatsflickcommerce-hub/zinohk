"""
experiments/sparsity_experiment.py
===================================
Phase 4 — Prove ZINOHK sparsity on a 100-node network.

Goal: show that only 1-5% of nodes fire per input.
Fixed: stronger homeostasis + higher initial thresholds.
"""

import numpy as np
from zinohk.core.node import ZNode


# ------------------------------------------------------------------ #
# Minimal sparse network — bypasses ZGraph routing issues
# and tests sparsity directly on ZNode homeostasis
# ------------------------------------------------------------------ #

N_NODES     = 100
N_INPUTS    = 10
N_STEPS     = 500
TARGET_RATE = 0.05   # target: <=5% active


def run():
    np.random.seed(42)

    print("=" * 56)
    print("ZINOHK Phase 4 — Sparsity Experiment (100 nodes)")
    print("=" * 56)

    # Build 100 independent nodes with strong homeostasis
    nodes = [
        ZNode(
            node_id       = f'n{i}',
            n_inputs      = N_INPUTS,
            threshold     = 0.5,        # start high
            lr            = 0.001,      # slow Hebbian
            target_rate   = 0.02,       # each node targets 2% activity
        )
        for i in range(N_NODES)
    ]

    # Give each node a random weight matrix
    for node in nodes:
        node.weights = np.random.uniform(0.0, 0.3, N_INPUTS)

    sparsity_history = []
    active_history   = []

    for step in range(N_STEPS):
        # Sparse random input — only 20% of inputs active each step
        raw_input = np.zeros(N_INPUTS)
        active_inputs = np.random.choice(N_INPUTS,
                                         size=2,
                                         replace=False)
        raw_input[active_inputs] = np.random.uniform(0.5, 1.0, 2)

        # Forward pass on all nodes
        n_fired = 0
        for node in nodes:
            out = node.forward(raw_input)
            if out > 0:
                n_fired += 1
            # Strong homeostasis — adapt threshold every step
            node.adapt_threshold()

        sparsity = n_fired / N_NODES
        sparsity_history.append(sparsity)
        active_history.append(n_fired)

        if (step + 1) % 100 == 0:
            avg_sparse = np.mean(sparsity_history[-100:])
            avg_active = np.mean(active_history[-100:])
            # Sample one node threshold
            sample_theta = nodes[0].threshold
            print(f"Step {step+1:4d} | "
                  f"fired: {avg_active:4.1f}/{N_NODES} | "
                  f"active: {avg_sparse*100:5.1f}% | "
                  f"theta[0]: {sample_theta:.3f}")

    # Final report
    mean_active   = float(np.mean(active_history))
    mean_sparsity = float(np.mean(sparsity_history))
    min_active    = int(np.min(active_history))
    max_active    = int(np.max(active_history))

    print()
    print("=" * 56)
    print("SPARSITY REPORT")
    print("=" * 56)
    print(f"Total nodes       : {N_NODES}")
    print(f"Total steps       : {N_STEPS}")
    print(f"Mean active/step  : {mean_active:.1f} nodes")
    print(f"Mean sparsity     : {mean_sparsity*100:.1f}% active")
    print(f"Min active        : {min_active} nodes")
    print(f"Max active        : {max_active} nodes")
    print()

    dense_ops  = N_NODES * N_INPUTS
    sparse_ops = mean_active * N_INPUTS
    saving_pct = (1 - sparse_ops / dense_ops) * 100

    print("COMPUTE COMPARISON (vs dense baseline)")
    print(f"Dense  ops/step   : {int(dense_ops):,}")
    print(f"Sparse ops/step   : {int(sparse_ops):,}")
    print(f"Compute saving    : {saving_pct:.1f}%")
    print()

    target_hit = mean_sparsity <= TARGET_RATE
    status = "✅ TARGET HIT" if target_hit else "⚠️  ABOVE TARGET"
    print(f"Target : <={TARGET_RATE*100:.0f}% active → {status}")
    print(f"Result : mean={mean_sparsity*100:.1f}% active")


if __name__ == "__main__":
    run()
