# ZINOHK

> Brain-inspired AI architecture — Sparse · Async · Spike-based · Feedback-driven · Locally-learned

---

## What is ZINOHK?

ZINOHK is a research-stage AI architecture built around how the
human brain actually works — not how transformers approximate it.

| Property        | Transformer      | ZINOHK                  |
|-----------------|------------------|-------------------------|
| Activation      | 100% every token | ~1% sparse firing       |
| Signal type     | 32-bit floats    | Binary spikes           |
| Processing      | Layer by layer   | Async — node by node    |
| Learning        | Backprop only    | Local Hebbian (always)  |
| After deploy    | Frozen forever   | Learns continuously     |

---

## Status

| Phase | Goal                        | Status      |
|-------|-----------------------------|-------------|
| 1     | Repo setup                  | 🔄 In progress |
| 2     | Single neuron prototype     | ⏳ Pending  |
| 3     | 100-node async graph        | ⏳ Pending  |
| 4     | Feedback loops              | ⏳ Pending  |
| 5     | Language task + Kaggle GPU  | ⏳ Pending  |
| 6     | Paper + open source         | ⏳ Pending  |

---

## Quick start

```bash
git clone <repo>
cd zinohk
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

---

*Full architecture docs → coming in Phase 1 completion*
