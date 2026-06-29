"""
ZINOHK REST API
================
Full REST API for the ZINOHK brain-inspired AI architecture.

Endpoints:
  GET  /                    health check + version
  POST /ask                 Q&A — answer any question
  POST /classify            text classification
  POST /learn               teach ZINOHK a new fact
  POST /generate            generate text from a seed word
  GET  /stats               system statistics
  GET  /knowledge           list all stored facts
  DELETE /knowledge/{id}    remove a fact

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Usage:
  curl http://localhost:8000/
  curl -X POST http://localhost:8000/ask \
       -H 'Content-Type: application/json' \
       -d '{"question": "What is the capital of France?"}'
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import time
import sys
import os

# Add zinohk to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from zinohk.knowledge.learner  import DynamicKnowledgeBase
from zinohk.encoding.vocab     import Vocabulary
from zinohk.encoding.decoder   import NGramDecoder
from zinohk.utils.safety       import SafetyClassifier

# ------------------------------------------------------------------ #
# App setup
# ------------------------------------------------------------------ #

app = FastAPI(
    title       = "ZINOHK API",
    description = "Brain-inspired AI — sparse, async, locally-learned",
    version     = "0.1.0",
)

# Allow all origins for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ------------------------------------------------------------------ #
# Initialise ZINOHK components on startup
# ------------------------------------------------------------------ #

print("Initialising ZINOHK...")

# Knowledge base — starts empty, grows from every /learn call
kb = DynamicKnowledgeBase(vocab_size=2000)

# Seed with basic facts so API is useful out of the box
SEED_FACTS = [
    ("What is ZINOHK? ZINOHK is a brain-inspired AI architecture built in 2026",
     "ZINOHK is a brain-inspired AI architecture", "system"),
    ("What is ZINOHK made of? ZINOHK uses sparse neurons async graph spike encoding and Hebbian learning",
     "sparse neurons, async graph, spike encoding, Hebbian learning", "system"),
    ("Who built ZINOHK? ZINOHK was built as a research project in 2026",
     "Built as a research project in 2026", "system"),
]
for text, answer, cat in SEED_FACTS:
    kb.learn(text, answer, cat)

# Text generation decoder
vocab   = Vocabulary(max_size=500, min_freq=1)
decoder = None   # built lazily when /generate is first called

# Safety classifier
safety = SafetyClassifier(safe_default=0)
safety.add_rule(
    'no_empty',
    'block empty inputs',
    lambda pred, scores: True,
)

# Stats tracking
start_time  = time.time()
n_requests  = 0
n_questions = 0
n_learns    = 0

print(f"✅ ZINOHK API ready | facts={kb.n_added}")


# ------------------------------------------------------------------ #
# Request / Response models
# ------------------------------------------------------------------ #

class AskRequest(BaseModel):
    question:            str
    use_wikipedia:       bool = True
    confidence_threshold: float = 0.35

class AskResponse(BaseModel):
    question:   str
    answer:     str
    confidence: float
    source:     str
    from_web:   bool
    latency_ms: float

class LearnRequest(BaseModel):
    text:     str
    answer:   str
    category: str = "general"

class LearnResponse(BaseModel):
    status:      str
    facts_total: int
    message:     str

class ClassifyRequest(BaseModel):
    text:   str
    labels: List[str] = ["positive", "negative"]

class ClassifyResponse(BaseModel):
    text:       str
    label:      str
    confidence: float
    scores:     dict
    latency_ms: float

class GenerateRequest(BaseModel):
    seed:        str
    max_length:  int   = 15
    temperature: float = 0.8
    train_on:    Optional[List[str]] = None

class GenerateResponse(BaseModel):
    seed:       str
    generated:  str
    latency_ms: float

class StatsResponse(BaseModel):
    version:       str
    uptime_s:      float
    n_requests:    int
    n_questions:   int
    n_learns:      int
    facts_total:   int
    vocab_size:    int


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.get("/", tags=["System"])
def root():
    """Health check and version info."""
    return {
        "name"       : "ZINOHK API",
        "version"    : "0.1.0",
        "status"     : "running",
        "description": "Brain-inspired AI — sparse, async, locally-learned",
        "endpoints"  : ["/ask", "/classify", "/learn",
                        "/generate", "/stats", "/knowledge"],
        "facts"      : kb.n_added,
        "uptime_s"   : round(time.time() - start_time, 1),
    }


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask(req: AskRequest):
    """
    Answer a question using ZINOHK knowledge base.

    - Searches local knowledge first (fast)
    - Falls back to Wikipedia if not confident
    - Caches Wikipedia answers permanently
    """
    global n_requests, n_questions
    n_requests  += 1
    n_questions += 1

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    t0 = time.time()
    r  = kb.ask(
        req.question,
        confidence_threshold = req.confidence_threshold,
        use_wikipedia        = req.use_wikipedia,
    )
    latency = (time.time() - t0) * 1000

    return AskResponse(
        question   = req.question,
        answer     = r.get("answer", "I don't know."),
        confidence = r.get("confidence", 0.0),
        source     = r.get("source", "unknown") or "unknown",
        from_web   = r.get("from_web", False),
        latency_ms = round(latency, 2),
    )


@app.post("/learn", response_model=LearnResponse, tags=["Knowledge"])
def learn(req: LearnRequest):
    """
    Teach ZINOHK a new fact.

    Facts are stored permanently and immediately searchable.
    No retraining required — continuous learning.
    """
    global n_requests, n_learns
    n_requests += 1
    n_learns   += 1

    if not req.text.strip() or not req.answer.strip():
        raise HTTPException(
            status_code=400,
            detail="Both text and answer are required")

    # Store as Q+A anchor pattern
    store_text = req.text + " " + req.text + " " + req.answer
    kb.learn(store_text, req.answer, req.category)

    return LearnResponse(
        status      = "learned",
        facts_total = kb.n_added,
        message     = f"Fact stored. Total facts: {kb.n_added}",
    )


@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
def classify(req: ClassifyRequest):
    """
    Classify text into provided labels using ZINOHK sparse encoding.

    Uses semantic similarity between the text and each label.
    No training required — zero-shot classification.
    """
    global n_requests
    n_requests += 1

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(req.labels) < 2:
        raise HTTPException(
            status_code=400, detail="At least 2 labels required")

    import numpy as np
    from zinohk.encoding.text import TFIDFEncoder

    t0 = time.time()

    # Score text against each label using kb retrieval
    scores = {}
    for label in req.labels:
        # Store label as a temporary query
        query   = f"{req.text} {label}"
        results = kb.retrieve(query, top_k=1)
        if results:
            scores[label] = round(float(results[0][0]), 4)
        else:
            scores[label] = 0.0

    # Fallback — use direct text similarity to labels
    if max(scores.values()) == 0.0:
        # Simple keyword overlap
        text_words = set(req.text.lower().split())
        for label in req.labels:
            label_words     = set(label.lower().split())
            overlap         = len(text_words & label_words)
            scores[label]   = float(overlap) / (len(text_words) + 1)

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    latency    = (time.time() - t0) * 1000

    return ClassifyResponse(
        text       = req.text,
        label      = best_label,
        confidence = round(best_score, 4),
        scores     = scores,
        latency_ms = round(latency, 2),
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
def generate(req: GenerateRequest):
    """
    Generate text from a seed word using ZINOHK NGramDecoder.

    Optionally train on new sentences first via train_on parameter.
    """
    global n_requests, decoder
    n_requests += 1

    if not req.seed.strip():
        raise HTTPException(status_code=400, detail="Seed cannot be empty")

    t0 = time.time()

    # Build decoder lazily on first call
    if decoder is None:
        _init_decoder()

    # Train on new sentences if provided
    if req.train_on:
        decoder.train(req.train_on, epochs=100)

    # Generate
    decoder.temperature = req.temperature
    generated = decoder.generate(req.seed.lower(), max_len=req.max_length)
    latency   = (time.time() - t0) * 1000

    return GenerateResponse(
        seed       = req.seed,
        generated  = generated if generated else f"{req.seed}...",
        latency_ms = round(latency, 2),
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
def stats():
    """Return ZINOHK system statistics."""
    return StatsResponse(
        version     = "0.1.0",
        uptime_s    = round(time.time() - start_time, 1),
        n_requests  = n_requests,
        n_questions = n_questions,
        n_learns    = n_learns,
        facts_total = kb.n_added,
        vocab_size  = kb.text_enc.vocab_size,
    )


@app.get("/knowledge", tags=["Knowledge"])
def list_knowledge(limit: int = 20, category: Optional[str] = None):
    """List stored facts."""
    facts = []
    for i, (text, answer, cat) in enumerate(
        zip(kb._raw_texts, kb._raw_answers, kb._raw_cats)
    ):
        if category and cat != category:
            continue
        facts.append({
            "id"      : i,
            "text"    : text[:100],
            "answer"  : answer[:100],
            "category": cat,
        })
        if len(facts) >= limit:
            break

    return {
        "total"   : kb.n_added,
        "returned": len(facts),
        "facts"   : facts,
    }


@app.delete("/knowledge/{fact_id}", tags=["Knowledge"])
def delete_fact(fact_id: int):
    """Remove a fact by index."""
    if fact_id < 0 or fact_id >= len(kb._raw_texts):
        raise HTTPException(status_code=404, detail="Fact not found")

    text = kb._raw_texts.pop(fact_id)
    kb._raw_answers.pop(fact_id)
    kb._raw_cats.pop(fact_id)
    kb._dirty   = True
    kb.n_added -= 1

    return {"status": "deleted", "text": text[:80]}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _init_decoder():
    """Initialise text generation decoder with seed corpus."""
    global decoder, vocab

    seed_corpus = [
        "the sky is blue and clear today",
        "knowledge grows from every interaction",
        "the brain learns from every experience",
        "neurons fire and wire together",
        "sparse activation saves compute power",
        "the model learns without backpropagation",
        "intelligence emerges from simple local rules",
        "data flows through the network like water",
        "every question makes the system smarter",
        "the future of AI is brain inspired",
    ]

    vocab.build(seed_corpus)
    decoder = NGramDecoder(vocab=vocab, lr=0.15, temperature=0.8)
    decoder.train(seed_corpus, epochs=300)
    print(f"✅ Decoder ready | vocab={len(vocab)}")
