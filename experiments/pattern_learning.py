"""
experiments/pattern_learning.py
================================
Phase 5 — Async graph learns a repeating pattern.

Goal: feed a 4-pattern sequence repeatedly into a 
20-node async graph. Prove the graph learns to 
anticipate the next pattern via local Hebbian rules.

Patterns (binary, 5-bit):
  A = [1, 0, 1, 0, 0]
  B = [0, 1, 0, 1, 0]
  C = [1, 1, 0, 0, 1]
  D = [0, 0, 1, 1, 0]

Sequence: A -> B -> C -> D -> A -> B -> ...

We measure: does the output layer activate differently
for each pattern? (pattern separation = learning)
"""

import numpy as np
from zinohk.core.node      import ZNode
from zinohk.core.synapse   import Synapse

# ------------------------------------------------------------------ #
# Patterns
# ------------------------------------------------------------------ #
PATTERNS = {
    'A': np.array([1.0, 0.0, 1.0, 0.0, 0.0]),
    'B': np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
    'C': np.array([1.0, 1.0, 0.0, 0.0, 1.0]),
    'D': np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
}
SEQUENCE = ['A', 'B', 'C', 'D']
N_INPUTS  = 5
N_HIDDEN  = 10
N_OUTPUT  = 4   # one output node per pattern


# ------------------------------------------------------------------ #
# Build network
# ------------------------------------------------------------------ #
def build_network():
    # Hidden layer: 10 nodes, each takes 5 inputs
    hidden = [
        ZNode(f'h{i}', n_inputs=N_INPUTS,
              threshold=0.15, lr=0.02, target_rate=0.03)
        for i in range(N_HIDDEN)
    ]

    # Output layer: 4 nodes, each takes 10 inputs
    output = [
        ZNode(f'o{i}', n_inputs=N_HIDDEN,
              threshold=0.1, lr=0.02, target_rate=0.1)
        for i in range(N_OUTPUT)
    ]

    # Synapses: hidden -> output (full connection)
    synapses = {}
    for i, h in enumerate(hidden):
        for j, o in enumerate(output):
            synapses[(i, j)] = Synapse(
                pre_id  = f'h{i}',
                post_id = f'o{j}',
                weight  = np.random.uniform(0.1, 0.3),
                lr      = 0.02,
            )

    return hidden, output, synapses


def forward(pattern, hidden, output, synapses):
    """One async forward pass."""
    # Hidden layer
    h_outputs = []
    for node in hidden:
        out = node.forward(pattern)
        node.adapt_threshold()
        h_outputs.append(out)

    h_vec = np.array(h_outputs)

    # Output layer: each output node gets all hidden outputs
    # weighted by its synapses
    o_outputs = []
    for j, o_node in enumerate(output):
        # Weighted input from hidden layer
        o_input = np.array([
            h_outputs[i] * synapses[(i, j)].weight
            for i in range(N_HIDDEN)
        ])
        out = o_node.forward(o_input)
        o_node.adapt_threshold()
        o_outputs.append(out)

    # Hebbian update on synapses
    for i in range(N_HIDDEN):
        for j in range(N_OUTPUT):
            synapses[(i, j)].hebbian_update(
                pre_fired  = h_outputs[i],
                post_fired = o_outputs[j],
            )

    return h_vec, np.array(o_outputs)


# ------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------ #
def run(epochs=300):
    np.random.seed(42)

    hidden, output, synapses = build_network()

    print("=" * 58)
    print("ZINOHK Phase 5 — Async Pattern Learning")
    print("Sequence: A->B->C->D->A->B->... (300 epochs)")
    print("=" * 58)

    # Track output activations per pattern over time
    history = {p: [] for p in SEQUENCE}

    for epoch in range(epochs):
        for label in SEQUENCE:
            pattern = PATTERNS[label]
            _, o_out = forward(pattern, hidden, output, synapses)
            history[label].append(o_out.copy())

    # ------------------------------------------------------------------ #
    # Report: mean output activation per pattern
    # in first 50 vs last 50 epochs
    # ------------------------------------------------------------------ #
    print()
    print("Output activation per pattern (mean over last 50 epochs):")
    print(f"{'Pattern':<10} {'o0':>8} {'o1':>8} {'o2':>8} {'o3':>8}  {'Fires'}")
    print("-" * 58)

    final_activations = {}
    for label in SEQUENCE:
        recent = np.array(history[label][-50:])
        means  = recent.mean(axis=0)
        fires  = (recent > 0).mean(axis=0)
        final_activations[label] = means
        fire_str = " ".join(["✅" if f > 0.1 else "⬜" for f in fires])
        print(f"{label:<10} "
              f"{means[0]:>8.3f} {means[1]:>8.3f} "
              f"{means[2]:>8.3f} {means[3]:>8.3f}  {fire_str}")

    # ------------------------------------------------------------------ #
    # Pattern separation score
    # Are different patterns activating differently?
    # ------------------------------------------------------------------ #
    print()
    print("Pattern separation (are patterns distinguishable?):")
    print("-" * 58)

    labels = list(final_activations.keys())
    separated = 0
    pairs      = 0
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            a = final_activations[labels[i]]
            b = final_activations[labels[j]]
            diff = float(np.linalg.norm(a - b))
            pairs += 1
            ok = "✅" if diff > 0.05 else "❌"
            if diff > 0.05:
                separated += 1
            print(f"  {labels[i]} vs {labels[j]}: "
                  f"distance={diff:.4f} {ok}")

    print()
    print(f"Separated: {separated}/{pairs} pattern pairs")

    # ------------------------------------------------------------------ #
    # Sparsity report
    # ------------------------------------------------------------------ #
    print()
    h_rates = [round(n.activity_rate, 3) for n in hidden]
    o_rates = [round(n.activity_rate, 3) for n in output]
    h_mean  = round(float(np.mean(h_rates)), 3)
    o_mean  = round(float(np.mean(o_rates)), 3)

    print("Sparsity report:")
    print(f"  Hidden layer mean activity : {h_mean*100:.1f}%")
    print(f"  Output layer mean activity : {o_mean*100:.1f}%")

    all_rates = h_rates + o_rates
    overall   = float(np.mean(all_rates))
    print(f"  Overall mean activity      : {overall*100:.1f}%")

    target_ok = overall <= 0.10
    print(f"  Target <=10%               : "
          f"{'✅' if target_ok else '⚠️ '} ({overall*100:.1f}%)")


if __name__ == "__main__":
    run(epochs=300)


# ------------------------------------------------------------------ #
# Fixed version — clipped outputs + stronger homeostasis
# ------------------------------------------------------------------ #

def run_fixed(epochs=300):
    np.random.seed(42)

    hidden, output, synapses = build_network()

    # Stronger homeostasis on output nodes
    for o in output:
        o.target_rate = 0.05
        o.threshold   = 0.3

    print()
    print("=" * 58)
    print("ZINOHK Phase 5 — Fixed (clipped + stronger homeostasis)")
    print("=" * 58)

    history = {p: [] for p in SEQUENCE}

    for epoch in range(epochs):
        for label in SEQUENCE:
            pattern = PATTERNS[label]

            # Hidden layer
            h_outputs = []
            for node in hidden:
                out = node.forward(pattern)
                out = float(np.clip(out, 0.0, 2.0))  # clip
                node.adapt_threshold()
                h_outputs.append(out)

            # Output layer
            o_outputs = []
            for j, o_node in enumerate(output):
                o_input = np.array([
                    h_outputs[i] * float(np.clip(
                        synapses[(i,j)].weight, 0.0, 1.0))
                    for i in range(N_HIDDEN)
                ])
                out = o_node.forward(o_input)
                out = float(np.clip(out, 0.0, 2.0))  # clip
                o_node.adapt_threshold()
                o_outputs.append(out)

            # Hebbian update
            for i in range(N_HIDDEN):
                for j in range(N_OUTPUT):
                    synapses[(i,j)].hebbian_update(
                        pre_fired  = h_outputs[i],
                        post_fired = o_outputs[j],
                    )

            history[label].append(np.array(o_outputs))

    # Report
    print()
    print("Output activation per pattern (last 50 epochs):")
    print(f"{'Pattern':<10} {'o0':>6} {'o1':>6} "
          f"{'o2':>6} {'o3':>6}  {'Fires'}")
    print("-" * 52)

    final_activations = {}
    for label in SEQUENCE:
        recent = np.array(history[label][-50:])
        means  = recent.mean(axis=0)
        fires  = (recent > 0).mean(axis=0)
        final_activations[label] = means
        fire_str = " ".join(["✅" if f > 0.1 else "⬜" for f in fires])
        print(f"{label:<10} "
              f"{means[0]:>6.3f} {means[1]:>6.3f} "
              f"{means[2]:>6.3f} {means[3]:>6.3f}  {fire_str}")

    print()
    print("Pattern separation:")
    print("-" * 52)
    separated = 0
    pairs     = 0
    for i in range(len(SEQUENCE)):
        for j in range(i+1, len(SEQUENCE)):
            a    = final_activations[SEQUENCE[i]]
            b    = final_activations[SEQUENCE[j]]
            diff = float(np.linalg.norm(a - b))
            pairs += 1
            ok = "✅" if diff > 0.01 else "❌"
            if diff > 0.01:
                separated += 1
            print(f"  {SEQUENCE[i]} vs {SEQUENCE[j]}: "
                  f"distance={diff:.4f} {ok}")

    print(f"\nSeparated: {separated}/{pairs} pairs")

    h_mean = float(np.mean([n.activity_rate for n in hidden]))
    o_mean = float(np.mean([n.activity_rate for n in output]))
    overall = (h_mean + o_mean) / 2
    print(f"\nSparsity:")
    print(f"  Hidden : {h_mean*100:.1f}%")
    print(f"  Output : {o_mean*100:.1f}%")
    print(f"  Overall: {overall*100:.1f}%")
    target_ok = overall <= 0.10
    print(f"  Target <=10%: {'✅' if target_ok else '⚠️ '} "
          f"({overall*100:.1f}%)")


if __name__ == "__main__":
    run(epochs=300)
    run_fixed(epochs=300)


# ------------------------------------------------------------------ #
# Fixed version — clipped outputs + stronger homeostasis
# ------------------------------------------------------------------ #

def run_fixed(epochs=300):
    np.random.seed(42)

    hidden, output, synapses = build_network()

    # Stronger homeostasis on output nodes
    for o in output:
        o.target_rate = 0.05
        o.threshold   = 0.3

    print()
    print("=" * 58)
    print("ZINOHK Phase 5 — Fixed (clipped + stronger homeostasis)")
    print("=" * 58)

    history = {p: [] for p in SEQUENCE}

    for epoch in range(epochs):
        for label in SEQUENCE:
            pattern = PATTERNS[label]

            # Hidden layer
            h_outputs = []
            for node in hidden:
                out = node.forward(pattern)
                out = float(np.clip(out, 0.0, 2.0))  # clip
                node.adapt_threshold()
                h_outputs.append(out)

            # Output layer
            o_outputs = []
            for j, o_node in enumerate(output):
                o_input = np.array([
                    h_outputs[i] * float(np.clip(
                        synapses[(i,j)].weight, 0.0, 1.0))
                    for i in range(N_HIDDEN)
                ])
                out = o_node.forward(o_input)
                out = float(np.clip(out, 0.0, 2.0))  # clip
                o_node.adapt_threshold()
                o_outputs.append(out)

            # Hebbian update
            for i in range(N_HIDDEN):
                for j in range(N_OUTPUT):
                    synapses[(i,j)].hebbian_update(
                        pre_fired  = h_outputs[i],
                        post_fired = o_outputs[j],
                    )

            history[label].append(np.array(o_outputs))

    # Report
    print()
    print("Output activation per pattern (last 50 epochs):")
    print(f"{'Pattern':<10} {'o0':>6} {'o1':>6} "
          f"{'o2':>6} {'o3':>6}  {'Fires'}")
    print("-" * 52)

    final_activations = {}
    for label in SEQUENCE:
        recent = np.array(history[label][-50:])
        means  = recent.mean(axis=0)
        fires  = (recent > 0).mean(axis=0)
        final_activations[label] = means
        fire_str = " ".join(["✅" if f > 0.1 else "⬜" for f in fires])
        print(f"{label:<10} "
              f"{means[0]:>6.3f} {means[1]:>6.3f} "
              f"{means[2]:>6.3f} {means[3]:>6.3f}  {fire_str}")

    print()
    print("Pattern separation:")
    print("-" * 52)
    separated = 0
    pairs     = 0
    for i in range(len(SEQUENCE)):
        for j in range(i+1, len(SEQUENCE)):
            a    = final_activations[SEQUENCE[i]]
            b    = final_activations[SEQUENCE[j]]
            diff = float(np.linalg.norm(a - b))
            pairs += 1
            ok = "✅" if diff > 0.01 else "❌"
            if diff > 0.01:
                separated += 1
            print(f"  {SEQUENCE[i]} vs {SEQUENCE[j]}: "
                  f"distance={diff:.4f} {ok}")

    print(f"\nSeparated: {separated}/{pairs} pairs")

    h_mean = float(np.mean([n.activity_rate for n in hidden]))
    o_mean = float(np.mean([n.activity_rate for n in output]))
    overall = (h_mean + o_mean) / 2
    print(f"\nSparsity:")
    print(f"  Hidden : {h_mean*100:.1f}%")
    print(f"  Output : {o_mean*100:.1f}%")
    print(f"  Overall: {overall*100:.1f}%")
    target_ok = overall <= 0.10
    print(f"  Target <=10%: {'✅' if target_ok else '⚠️ '} "
          f"({overall*100:.1f}%)")


if __name__ == "__main__":
    run(epochs=300)
    run_fixed(epochs=300)
