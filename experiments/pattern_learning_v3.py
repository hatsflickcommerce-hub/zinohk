"""
experiments/pattern_learning_v3.py
====================================
Phase 5c — Pattern learning with contrastive Hebbian.

Fixes GAP-01: representational collapse of C and D.

Changes from v2:
  - Replace plain Hebbian with contrastive update
  - Winner synapses strengthen, loser synapses weaken
  - Forces each output node to specialise on one pattern

Target: 4/4 unique winners + 6/6 pattern separation.
"""

import numpy as np
from zinohk.core.node              import ZNode
from zinohk.core.synapse           import Synapse
from zinohk.learning.contrastive   import contrastive_update

PATTERNS = {
    'A': np.array([1.0, 0.0, 1.0, 0.0, 0.0]),
    'B': np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
    'C': np.array([1.0, 1.0, 0.0, 0.0, 1.0]),
    'D': np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
}
SEQUENCE  = ['A', 'B', 'C', 'D']
N_INPUTS  = 5
N_HIDDEN  = 10
N_OUTPUT  = 4


def build_network(rng):
    hidden = [
        ZNode(f'h{i}', n_inputs=N_INPUTS,
              threshold=0.2, lr=0.005, target_rate=0.02)
        for i in range(N_HIDDEN)
    ]
    for node in hidden:
        node.weights = rng.uniform(0.0, 0.5, N_INPUTS)
        mask = np.zeros(N_INPUTS)
        idx  = rng.choice(N_INPUTS, size=3, replace=False)
        mask[idx] = 1.0
        node.weights *= mask

    output = [
        ZNode(f'o{i}', n_inputs=N_HIDDEN,
              threshold=0.15, lr=0.005, target_rate=0.05)
        for i in range(N_OUTPUT)
    ]
    for node in output:
        node.weights = rng.uniform(0.0, 0.3, N_HIDDEN)

    synapses = {}
    for i in range(N_HIDDEN):
        for j in range(N_OUTPUT):
            synapses[(i, j)] = Synapse(
                pre_id  = f'h{i}',
                post_id = f'o{j}',
                weight  = float(rng.uniform(0.05, 0.4)),
                lr      = 0.005,
            )

    return hidden, output, synapses


def forward(pattern, hidden, output, synapses):
    # Hidden layer
    h_out = []
    for node in hidden:
        val = float(np.clip(node.forward(pattern), 0.0, 1.0))
        node.adapt_threshold()
        h_out.append(val)

    # Output layer
    o_raw = []
    for j, o_node in enumerate(output):
        o_input = np.array([
            h_out[i] * float(np.clip(synapses[(i,j)].weight, 0.0, 1.0))
            for i in range(N_HIDDEN)
        ])
        val = float(np.clip(o_node.forward(o_input), 0.0, 1.0))
        o_node.adapt_threshold()
        o_raw.append(val)

    # Winner-take-all
    if max(o_raw) > 0:
        winner  = int(np.argmax(o_raw))
        o_out   = [0.0] * N_OUTPUT
        o_out[winner] = o_raw[winner]
    else:
        o_out = o_raw

    # Contrastive Hebbian update
    contrastive_update(
        synapses = synapses,
        h_out    = h_out,
        o_out    = o_out,
        n_hidden = N_HIDDEN,
        n_output = N_OUTPUT,
        lr       = 0.02,
        decay    = 0.008,
    )

    return np.array(h_out), np.array(o_out)


def run(epochs=500):
    rng = np.random.default_rng(7)
    hidden, output, synapses = build_network(rng)

    print("=" * 58)
    print("ZINOHK Phase 5c — Contrastive Hebbian")
    print("Target: 4/4 unique winners + 6/6 separation")
    print("=" * 58)

    history = {p: [] for p in SEQUENCE}

    for epoch in range(epochs):
        for label in SEQUENCE:
            _, o_out = forward(
                PATTERNS[label], hidden, output, synapses)
            history[label].append(o_out.copy())

        if (epoch + 1) % 100 == 0:
            winners = {}
            for label in SEQUENCE:
                recent   = np.array(history[label][-20:])
                mean_out = recent.mean(axis=0)
                winners[label] = int(np.argmax(mean_out))
            unique = len(set(winners.values()))
            print(f"Epoch {epoch+1:4d} | "
                  f"winners: {winners} | "
                  f"unique: {unique}/4")

    # Final report
    print()
    print("Final output activations (mean last 50 epochs):")
    print(f"{'Pattern':<10} {'o0':>6} {'o1':>6} "
          f"{'o2':>6} {'o3':>6}  Winner")
    print("-" * 52)

    final = {}
    for label in SEQUENCE:
        recent = np.array(history[label][-50:])
        means  = recent.mean(axis=0)
        final[label] = means
        winner = int(np.argmax(means))
        print(f"{label:<10} "
              f"{means[0]:>6.3f} {means[1]:>6.3f} "
              f"{means[2]:>6.3f} {means[3]:>6.3f}  o{winner}")

    # Unique winners
    winners = {lb: int(np.argmax(v)) for lb, v in final.items()}
    unique  = len(set(winners.values()))
    ok      = "✅" if unique == 4 else "⚠️ "
    print()
    print(f"Unique winners : {unique}/4 {ok}")
    print(f"Pattern->Winner: {winners}")

    # Separation
    print()
    print("Pattern separation:")
    separated = 0
    pairs     = 0
    for i in range(len(SEQUENCE)):
        for j in range(i+1, len(SEQUENCE)):
            a    = final[SEQUENCE[i]]
            b    = final[SEQUENCE[j]]
            diff = float(np.linalg.norm(a - b))
            pairs += 1
            ok2  = "✅" if diff > 0.01 else "❌"
            if diff > 0.01:
                separated += 1
            print(f"  {SEQUENCE[i]} vs {SEQUENCE[j]}: "
                  f"distance={diff:.4f} {ok2}")
    sep_ok = "✅" if separated == pairs else "⚠️ "
    print(f"\nSeparated: {separated}/{pairs} {sep_ok}")

    # Sparsity
    h_mean  = float(np.mean([n.activity_rate for n in hidden]))
    o_mean  = float(np.mean([n.activity_rate for n in output]))
    overall = (h_mean + o_mean) / 2
    print()
    print("Sparsity:")
    print(f"  Hidden : {h_mean*100:.1f}%")
    print(f"  Output : {o_mean*100:.1f}%")
    print(f"  Overall: {overall*100:.1f}%")
    print(f"  Target <=10%: "
          f"{'✅' if overall<=0.10 else '⚠️ '} "
          f"({overall*100:.1f}%)")


if __name__ == "__main__":
    run(epochs=500)


def run_aggressive(epochs=1000):
    """Stronger decay, more epochs — force 4/4 separation."""
    rng = np.random.default_rng(13)
    hidden, output, synapses = build_network(rng)

    print()
    print("=" * 58)
    print("ZINOHK Phase 5c — Aggressive contrastive (1000 epochs)")
    print("=" * 58)

    history = {p: [] for p in SEQUENCE}

    for epoch in range(epochs):
        for label in SEQUENCE:
            _, o_out = forward(
                PATTERNS[label], hidden, output, synapses)
            history[label].append(o_out.copy())

        if (epoch + 1) % 200 == 0:
            winners = {}
            for label in SEQUENCE:
                recent   = np.array(history[label][-20:])
                mean_out = recent.mean(axis=0)
                winners[label] = int(np.argmax(mean_out))
            unique = len(set(winners.values()))
            print(f"Epoch {epoch+1:4d} | "
                  f"winners: {winners} | "
                  f"unique: {unique}/4")

    # Final
    print()
    final = {}
    for label in SEQUENCE:
        recent = np.array(history[label][-50:])
        final[label] = recent.mean(axis=0)

    winners = {lb: int(np.argmax(v)) for lb, v in final.items()}
    unique  = len(set(winners.values()))
    ok      = "✅" if unique == 4 else "⚠️ "
    print(f"Unique winners : {unique}/4 {ok}")
    print(f"Pattern->Winner: {winners}")

    separated = 0
    pairs     = 0
    print("\nPattern separation:")
    for i in range(len(SEQUENCE)):
        for j in range(i+1, len(SEQUENCE)):
            a    = final[SEQUENCE[i]]
            b    = final[SEQUENCE[j]]
            diff = float(np.linalg.norm(a - b))
            pairs += 1
            ok2  = "✅" if diff > 0.01 else "❌"
            if diff > 0.01:
                separated += 1
            print(f"  {SEQUENCE[i]} vs {SEQUENCE[j]}: "
                  f"distance={diff:.4f} {ok2}")
    print(f"\nSeparated: {separated}/{pairs} "
          f"{'✅' if separated==pairs else '⚠️ '}")

    h_mean  = float(np.mean([n.activity_rate for n in hidden]))
    o_mean  = float(np.mean([n.activity_rate for n in output]))
    overall = (h_mean + o_mean) / 2
    print(f"\nSparsity: {overall*100:.1f}% "
          f"{'✅' if overall<=0.10 else '⚠️ '}")


if __name__ == "__main__":
    run(epochs=500)
    run_aggressive(epochs=1000)
