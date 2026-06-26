# ZINOHK — Known Gaps Tracker

> Every gap is tracked openly. None hidden. Each has a fix plan.
> Updated after Phase 6 Kaggle results.

---

| ID     | Issue                       | Severity | Status      | Phase | Fix                          |
|--------|-----------------------------|----------|-------------|-------|------------------------------|
| GAP-01 | Pattern collapse (A+C, B+D) | Medium   | Confirmed   | 7     | Structured receptive fields  |
| GAP-02 | Language fluency            | High     | Partial ✅  | 7     | Spike encoding + more data   |
| GAP-03 | World knowledge             | High     | Partial ✅  | 7     | Larger dataset (full IMDb)   |
| GAP-04 | Catastrophic forgetting     | High     | Open        | 7     | Complementary learning (CLS) |
| GAP-05 | Spike encoding precision    | Medium   | Open        | 7     | Temporal spike encoding      |
| GAP-06 | Feedback loop stability     | Medium   | Open        | 7     | Predictive coding layer      |
| GAP-07 | Safety and alignment        | High     | Unstarted   | 8     | Value-protected weights      |
| GAP-08 | Instruction following       | Medium   | Unstarted   | 8     | Supervised signal shaping    |

---

## Phase 7 attack order

### GAP-05 first — Spike encoding
Why first: fixes GAP-01 and GAP-02 simultaneously.
Word order is lost in bag-of-words. Spike timing encodes
order and strength — more information per signal.

### GAP-06 second — Feedback loops
Why second: builds on spike encoding.
High-level nodes predict. Low-level nodes only fire
on prediction error. Cuts compute further.

### GAP-04 third — Catastrophic forgetting
Why third: needed before any real deployment.
New learning must not overwrite old knowledge.
Fix: fast memory (hippocampus) + slow consolidation (cortex).

### GAP-01 fourth — Pattern collapse
Why fourth: structured receptive fields need spike encoding first.
Each hidden node watches a specific input region, not random.

### GAP-02 + GAP-03 — Language + knowledge
Addressed continuously as architecture improves.
Full IMDb (25K) on Kaggle after spike encoding is in.

### GAP-07 + GAP-08 — Safety + instruction following
Phase 8 — after core architecture is stable.

---

## Progress log

| Phase | Gaps addressed | Result |
|-------|---------------|--------|
| 1–5   | Architecture  | Core built, tested |
| 6     | GAP-02, GAP-03 partial | 56.2% IMDb, 95% MLP accuracy at 4% compute |
| 7     | GAP-05, GAP-06, GAP-04, GAP-01 | Planned |
| 8     | GAP-07, GAP-08 | Future |
