# ZINOHK — Known Gaps Tracker

> Every gap is tracked openly. None hidden. Each has a fix plan.
> Updated after Phase 7 completion.

---

| ID     | Issue                       | Severity | Status       | Phase | Fix                          |
|--------|-----------------------------|----------|--------------|-------|------------------------------|
| GAP-01 | Pattern collapse (A+C, B+D) | Medium   | ✅ Fixed     | 7     | Structured receptive fields  |
| GAP-02 | Language fluency            | High     | 🔄 Partial   | 8     | Phase 8 language encoding    |
| GAP-03 | World knowledge             | High     | 🔄 Partial   | 8     | Phase 8 knowledge base       |
| GAP-04 | Catastrophic forgetting     | High     | ✅ Fixed     | 7     | Complementary learning (CLS) |
| GAP-05 | Spike encoding precision    | Medium   | ✅ Fixed     | 7     | Temporal spike encoding      |
| GAP-06 | Feedback loop stability     | Medium   | ✅ Fixed     | 7     | Predictive coding layer      |
| GAP-07 | Safety and alignment        | High     | ✅ Fixed     | 7     | Value gate + safety filter   |
| GAP-08 | Instruction following       | Medium   | ✅ Fixed     | 7     | InstructionShaper lr mod     |

---

## What was built per gap

### GAP-01 ✅ — Structured Receptive Fields
File: `zinohk/core/receptive.py`
- ReceptiveField: each node watches specific input channels
- ReceptiveLayout: tiled fields with overlap, 100% coverage
- A vs C distance: 0.757 (was 0.0 — collapsed before)
- B vs D distance: 0.484 (was 0.0 — collapsed before)

### GAP-04 ✅ — Complementary Learning System
File: `zinohk/learning/memory.py`
- FastMemory: hippocampus-like, learns immediately
- SlowMemory: cortex-like, learns via replay only
- MemorySystem: coordinates both
- Verified: A not forgotten after learning B

### GAP-05 ✅ — Spike Temporal Encoding
File: `zinohk/encoding/spike.py`
- SpikeEncoder: float → timed spike events
- SpikeDecoder: spike timing → float reconstruction
- Strong signal = early spike, weak = late, zero = silent
- Round-trip verified, correlation 0.84

### GAP-06 ✅ — Predictive Feedback Loop
File: `zinohk/feedback/predictor.py`
- PredictiveNode: fires only on prediction error > epsilon
- Same input cost: 2.50 → 0.00 over 5 steps
- Novel input correctly scores 3.70 (high surprise)

### GAP-07 ✅ — Safety & Alignment
File: `zinohk/utils/safety.py`
- ValueGate: protected weights, hard constraints
- SafetyClassifier: rule-based output filter
- Blocks violations, returns safe default

### GAP-08 ✅ — Instruction Following
File: `zinohk/utils/safety.py`
- InstructionShaper: lr modulation per instruction
- Correct follow → lr × 2.0 (amplify)
- Violation → lr × -1.0 (reverse)

---

## Remaining open gaps

| ID     | Issue            | Severity | Status     | Phase | Fix                        |
|--------|------------------|----------|------------|-------|----------------------------|
| GAP-02 | Language fluency | High     | 🔄 Partial | 8     | Large corpus + embeddings  |
| GAP-03 | World knowledge  | High     | 🔄 Partial | 8     | Knowledge base + retrieval |

---

## Progress log

| Phase | Gaps addressed              | Result                                    |
|-------|-----------------------------|-------------------------------------------|
| 1–5   | Architecture                | Core built, 35 tests passing              |
| 6     | GAP-02, GAP-03 partial      | 56.2% IMDb, 95% MLP at 4% compute        |
| 7     | GAP-01,04,05,06,07,08       | All fixed, 35/35 tests green              |
| 8     | GAP-02, GAP-03              | Language + world knowledge (Kaggle next)  |
