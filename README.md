# ZINOHK

> Brain-inspired AI architecture — Sparse · Async · Spike-based · Feedback-driven · Locally-learned
Version : 0.1.0

Phase   : 5c complete — moving to Kaggle GPU (Phase 6)

Python  : 3.12+

Tests   : 17/17 passing

---

## What is ZINOHK?

ZINOHK is a research-stage AI architecture built around how the
human brain actually works — not how transformers approximate it.

| Property       | Transformer       | ZINOHK                  |
|----------------|-------------------|-------------------------|
| Activation     | 100% every token  | 2.6% sparse firing      |
| Signal type    | 32-bit floats     | Threshold-gated values  |
| Processing     | Layer by layer    | Async — node by node    |
| Learning       | Backprop only     | Local Hebbian (always)  |
| After deploy   | Frozen forever    | Learns continuously     |
| Hardware       | GPU cluster       | CPU / edge device       |
| Training cost  | $10M–$100M        | Near zero               |

---

## Proven results (real code, not theory)

| Claim                    | Result              | Experiment                   |
|--------------------------|---------------------|------------------------------|
| Sparse activation        | 2.6% active         | sparsity_experiment.py       |
| Compute saving           | 97.4% vs dense      | sparsity_experiment.py       |
| Hebbian learning         | XOR 75% no backprop | xor_experiment.py            |
| Inhibitory suppression   | XOR 4/4 no backprop | xor_inhibitory.py            |
| Homeostatic regulation   | theta self-tunes    | sparsity_experiment.py       |
| Pattern separation       | 6/6 pairs distinct  | pattern_learning.py          |
| Brain-like sparsity      | 2.8% hidden layer   | pattern_learning_v2.py       |
| Contrastive learning     | winner specialises  | pattern_learning_v3.py       |

---

## Project structure
zinohk/

├── zinohk/                  source package

│   ├── init.py          v0.1.0

│   └── core/

│       ├── node.py          ZNode — sparse threshold + Hebbian + homeostasis

│       ├── synapse.py       Synapse — Hebbian + STDP local learning

│       ├── graph.py         ZGraph — async node graph

│       └── inhibition.py   InhibitoryNode — active suppression

│   └── learning/

│       └── contrastive.py  Contrastive Hebbian — winner/loser update

├── experiments/

│   ├── xor_experiment.py        Phase 2 — XOR, Hebbian only, 75%

│   ├── xor_inhibitory.py        Phase 3 — XOR + inhibition, 100%

│   ├── sparsity_experiment.py   Phase 4 — 2.6% active, 97.4% saving

│   ├── pattern_learning.py      Phase 5  — 6/6 pattern separation

│   ├── pattern_learning_v2.py   Phase 5b — sparsity 3.7%

│   └── pattern_learning_v3.py   Phase 5c — contrastive Hebbian

├── tests/

│   ├── test_node.py         8 tests — ZNode

│   └── test_synapse.py      9 tests — Synapse

├── docs/

├── pyproject.toml

└── README.md

---

## Phase tracker

| Phase | Goal                            | Status      | Result                     |
|-------|---------------------------------|-------------|----------------------------|
| 1     | Repo + core code + tests        | ✅ Done     | 17 tests passing           |
| 2     | XOR without backprop            | ✅ Done     | 75% Hebbian only           |
| 3     | Inhibitory nodes + XOR 4/4      | ✅ Done     | 100% no backprop           |
| 4     | Sparsity on 100 nodes           | ✅ Done     | 2.6% active, 97.4% saving  |
| 5     | Async pattern learning          | ✅ Done     | 6/6 separation, 4.2% sparse|
| 6     | Language task — Kaggle GPU      | 🔜 Next     |                            |
| 7     | Paper + open source             | ⏳ Pending  |                            |

---

## Quick start

```bash
git clone <repo>
cd zinohk
python3 -m venv venv
source venv/bin/activate
pip install numpy loguru pyyaml tqdm networkx pytest pytest-cov
pip install -e "." --no-deps
pytest tests/ -v
```

---

## Known gaps — tracked openly

| ID     | Issue                              | Severity | Status       | Fix planned         |
|--------|------------------------------------|----------|--------------|---------------------|
| GAP-01 | Pattern collapse (A+C, B+D)        | Medium   | Confirmed    | Structured receptive fields (Phase 6) |
| GAP-02 | Language fluency                   | High     | Open         | Kaggle GPU (Phase 6)|
| GAP-03 | World knowledge                    | High     | Open         | Kaggle GPU (Phase 6)|
| GAP-04 | Catastrophic forgetting            | High     | Open         | Phase 6+            |
| GAP-05 | Spike encoding precision           | Medium   | In research  | Phase 6+            |
| GAP-06 | Feedback loop stability            | Medium   | In research  | Phase 6+            |
| GAP-07 | Safety and alignment               | High     | Unstarted    | Future research     |
| GAP-08 | Instruction following              | Medium   | Unstarted    | Phase 6+            |

> Policy: no gap is hidden. Every known limitation is listed,
> tracked, and addressed in the roadmap.

---

## Architecture — the 5 pillars

### Pillar 1 — Sparse Threshold Activation
Only fire when input exceeds threshold theta.
Self-regulating via homeostatic controller.
output = x     if x > theta

= 0     otherwise

theta += alpha * (activity_rate - target_rate)
**Proved: 2.6% active, 97.4% compute saving.**

### Pillar 2 — Async Node Graph
No fixed layers. Each node fires when inputs arrive.
node.fire() when inputs.ready()

node.push(output) to downstream
**Implemented: ZGraph with topological routing.**

### Pillar 3 — Local Hebbian Learning
No backprop. Each synapse updates from local signal only.
delta_w = lr * pre * post

w = clip(w + delta_w, -5, 5)
**Proved: XOR learned, weights update without gradients.**

### Pillar 4 — Inhibitory Suppression
Active suppression when too many inputs fire.
if all(inputs > threshold): return -strength

else: return 0
**Proved: XOR 4/4 solved with inhibitory node.**

### Pillar 5 — Contrastive Hebbian
Winner strengthens, losers weaken.
winner: w += lr * pre * post

losers: w -= decay * pre
**Proved: output nodes specialise per pattern.**

---

*ZINOHK — because the next paradigm in AI should cost 20 watts, not 700 kilowatts.*

*Version 0.1.0 — Phase 5c complete*
*Next: Kaggle GPU — language task*
