"""
experiments/xor_inhibitory.py
==============================
Phase 3 — XOR with InhibitoryNode.

Architecture:
  in0 -> hidden -> output
  in1 ->        ->
              inhibitor (fires when both active, suppresses output)

Target: 4/4 XOR correct without backprop.
"""

import numpy as np
from zinohk.core.node       import ZNode
from zinohk.core.inhibition import InhibitoryNode

XOR_INPUTS = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
XOR_LABELS = np.array([0.0, 1.0, 1.0, 0.0])


def forward(inputs, hidden, inhibitor, output):
    # Hidden node sees both inputs
    h_input = np.array([inputs[0] * 0.5, inputs[1] * 0.5])
    h_out   = hidden.forward(h_input)

    # Inhibitor fires only when both inputs strong
    inh_out = inhibitor.forward(inputs)   # -strength or 0.0

    # Output = hidden signal + inhibitor suppression
    o_input = np.array([h_out + inh_out])
    o_out   = output.forward(o_input)

    return h_out, inh_out, o_out


def predict(o_out: float) -> int:
    return 1 if o_out > 0.05 else 0


def run(epochs: int = 500, seed: int = 42):
    np.random.seed(seed)

    # Key fix: very low lr so threshold doesn't explode
    # disable homeostasis by setting it manually each step
    hidden    = ZNode(node_id='hidden',
                      n_inputs=2,
                      threshold=0.2,
                      lr=0.001)
    inhibitor = InhibitoryNode(node_id='inhibitor',
                               n_inputs=2,
                               threshold=0.4,
                               strength=1.5)
    output    = ZNode(node_id='output',
                      n_inputs=1,
                      threshold=0.05,
                      lr=0.001)

    # Pin thresholds — let inhibition do the work, not homeostasis
    HIDDEN_THETA = 0.2
    OUTPUT_THETA = 0.05

    print("=" * 52)
    print("ZINOHK Phase 3 — XOR with Inhibitory Node")
    print("No backprop. Inhibition does the heavy lifting.")
    print("=" * 52)

    for epoch in range(epochs):
        idx = np.random.permutation(4)
        correct = 0

        for i in idx:
            inputs = XOR_INPUTS[i]
            label  = XOR_LABELS[i]

            h_out, inh_out, o_out = forward(
                inputs, hidden, inhibitor, output)
            pred = predict(o_out)
            if pred == int(label):
                correct += 1

            # Pin thresholds to prevent runaway homeostasis
            hidden.threshold = HIDDEN_THETA
            output.threshold = OUTPUT_THETA

        if (epoch + 1) % 100 == 0:
            acc = correct / 4 * 100
            print(f"Epoch {epoch+1:3d} | accuracy: {acc:.1f}%")

    # Final evaluation
    print()
    print("Final predictions:")
    print(f"{'Input':<12} {'Label':<8} {'Hidden':<8} "
          f"{'Inhib':<8} {'Output':<10} {'Pred':<6} {'OK'}")
    print("-" * 58)

    correct = 0
    for i in range(4):
        inputs = XOR_INPUTS[i]
        label  = XOR_LABELS[i]
        h_out, inh_out, o_out = forward(
            inputs, hidden, inhibitor, output)
        pred = predict(o_out)
        ok   = "✅" if pred == int(label) else "❌"
        if pred == int(label):
            correct += 1
        print(f"{str(inputs):<12} {int(label):<8} {h_out:<8.3f} "
              f"{inh_out:<8.3f} {o_out:<10.4f} {pred:<6} {ok}")

    print("-" * 58)
    print(f"Final accuracy: {correct}/4 = {correct/4*100:.1f}%")
    print()
    print("Node stats:")
    print(" ", hidden.stats())
    print(" ", inhibitor.stats())
    print(" ", output.stats())


if __name__ == "__main__":
    run(epochs=500)
