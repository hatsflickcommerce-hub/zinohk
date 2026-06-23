"""
experiments/xor_experiment.py
==============================
Phase 2 — Can ZINOHK learn XOR without backpropagation?

XOR truth table:
  [0, 0] -> 0
  [0, 1] -> 1
  [1, 0] -> 1
  [1, 1] -> 0

We use a 2-layer graph:
  2 input nodes -> 1 hidden node -> 1 output node
All learning is local Hebbian only. No gradients.
"""

import numpy as np
from zinohk.core.node    import ZNode
from zinohk.core.synapse import Synapse


# ------------------------------------------------------------------ #
# XOR data
# ------------------------------------------------------------------ #

XOR_INPUTS = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

XOR_LABELS = np.array([0.0, 1.0, 1.0, 0.0])


# ------------------------------------------------------------------ #
# Build a minimal 2-layer ZINOHK network by hand
#
#   n_in0 \
#           --> n_hidden --> n_out
#   n_in1 /
#
# ------------------------------------------------------------------ #

def build_network():
    hidden = ZNode(node_id='hidden', n_inputs=2, threshold=0.1, lr=0.05)
    output = ZNode(node_id='output', n_inputs=1, threshold=0.1, lr=0.05)

    syn_in0_hid = Synapse('in0', 'hidden', weight=0.3, lr=0.05)
    syn_in1_hid = Synapse('in1', 'hidden', weight=0.3, lr=0.05)
    syn_hid_out = Synapse('hidden', 'output', weight=0.3, lr=0.05)

    return hidden, output, syn_in0_hid, syn_in1_hid, syn_hid_out


def forward(inputs, hidden, output, s0, s1, s_ho):
    """One forward pass through the 2-layer network."""

    # Hidden layer input: each input weighted separately
    h_input = np.array([
        inputs[0] * s0.weight,
        inputs[1] * s1.weight,
    ])
    h_out = hidden.forward(h_input)

    # Output layer input: hidden output * synapse weight
    o_input = np.array([h_out * s_ho.weight])
    o_out   = output.forward(o_input)

    return h_out, o_out


def predict(o_out: float) -> int:
    """Threshold output to binary prediction."""
    return 1 if o_out > 0.1 else 0


# ------------------------------------------------------------------ #
# Training loop
# ------------------------------------------------------------------ #

def train(epochs: int = 200, seed: int = 42):
    np.random.seed(seed)

    hidden, output, s0, s1, s_ho = build_network()

    print("=" * 50)
    print("ZINOHK XOR Experiment — Phase 2")
    print("Learning rule: Hebbian only (no backprop)")
    print("=" * 50)

    for epoch in range(epochs):
        # Shuffle data each epoch
        idx = np.random.permutation(4)

        correct = 0
        for i in idx:
            inputs = XOR_INPUTS[i]
            label  = XOR_LABELS[i]

            h_out, o_out = forward(inputs, hidden, output, s0, s1, s_ho)
            pred = predict(o_out)

            if pred == int(label):
                correct += 1

            # Hebbian update on all synapses
            s0.hebbian_update(pre_fired=inputs[0], post_fired=h_out)
            s1.hebbian_update(pre_fired=inputs[1], post_fired=h_out)
            s_ho.hebbian_update(pre_fired=h_out,   post_fired=o_out)

        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            acc = correct / 4 * 100
            print(f"Epoch {epoch+1:3d} | accuracy: {acc:.1f}% | "
                  f"hidden_theta: {hidden.threshold:.3f} | "
                  f"output_theta: {output.threshold:.3f}")

    # ------------------------------------------------------------------ #
    # Final evaluation
    # ------------------------------------------------------------------ #
    print()
    print("Final predictions:")
    print(f"{'Input':<12} {'Label':<8} {'Output':<10} {'Pred':<6} {'OK'}")
    print("-" * 46)

    correct = 0
    for i in range(4):
        inputs = XOR_INPUTS[i]
        label  = XOR_LABELS[i]
        _, o_out = forward(inputs, hidden, output, s0, s1, s_ho)
        pred = predict(o_out)
        ok   = "✅" if pred == int(label) else "❌"
        if pred == int(label):
            correct += 1
        print(f"{str(inputs):<12} {int(label):<8} {o_out:<10.4f} {pred:<6} {ok}")

    print("-" * 46)
    print(f"Final accuracy: {correct}/4 = {correct/4*100:.1f}%")
    print()
    print("Node stats:")
    print(" ", hidden.stats())
    print(" ", output.stats())


if __name__ == "__main__":
    train(epochs=200)
