"""
ZINOHK REST API v0.2.0
========================
Full multi-domain API with thinking layer.

Endpoints:
  GET  /                    health check + version
  POST /ask                 Q&A with thinking + domain routing
  POST /learn               teach new facts to any domain
  POST /train               bulk train a domain from list of facts
  POST /classify            text classification
  POST /generate            text generation
  GET  /stats               system statistics
  GET  /knowledge           list stored facts
  GET  /domains             list registered domains
  DELETE /knowledge/{id}    remove a fact

New in v0.2:
  - DomainRouter: routes questions to correct KB
  - Thinker: 5-step reasoning before answering
  - /train endpoint: universal training API
  - /domains endpoint: list all active domains
  - Full reasoning trace in /ask response
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from zinohk.knowledge.learner  import DynamicKnowledgeBase
from zinohk.encoding.vocab     import Vocabulary
from zinohk.encoding.decoder   import NGramDecoder
from zinohk.utils.safety       import SafetyClassifier
from zinohk.router.domain      import DomainRouter
from zinohk.router.thinker     import Thinker

# ------------------------------------------------------------------ #
# App setup
# ------------------------------------------------------------------ #

app = FastAPI(
    title       = "ZINOHK API",
    description = "Brain-inspired AI — multi-domain, thinking, locally-learned",
    version     = "0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ------------------------------------------------------------------ #
# Initialise ZINOHK components
# ------------------------------------------------------------------ #

print("Initialising ZINOHK v0.2.0...")

# Domain knowledge bases — one per domain
domain_kbs: Dict[str, DynamicKnowledgeBase] = {}

def get_or_create_kb(domain: str) -> DynamicKnowledgeBase:
    """Get existing KB or create new one for domain."""
    if domain not in domain_kbs:
        domain_kbs[domain] = DynamicKnowledgeBase(vocab_size=2000)
        print(f"✅ Created KB for domain: '{domain}'")
    return domain_kbs[domain]

# Create default domains
general_kb = get_or_create_kb('general')
finance_kb = get_or_create_kb('finance')

# Seed general KB
for text, answer, cat in [
    ("What is ZINOHK? ZINOHK is a brain-inspired AI architecture built in 2026",
     "ZINOHK is a brain-inspired AI architecture", "system"),
    ("What is ZINOHK made of? ZINOHK uses sparse neurons and Hebbian learning",
     "Sparse neurons, async graph, spike encoding, Hebbian learning", "system"),
]:
    general_kb.learn(text, answer, cat)

# Seed finance KB
for text, answer, cat in [
    ("What is a call option? A call option gives the right to buy shares at the strike price",
     "Right to buy shares at strike price before expiry", "options_trading"),
    ("What is swing trading? Swing trading holds positions for days to weeks",
     "Holding trades for days to weeks to capture price moves", "swing_trading"),
    ("What is RSI? RSI measures momentum on a 0 to 100 scale above 70 overbought below 30 oversold",
     "Momentum oscillator measuring overbought or oversold conditions", "swing_trading"),
]:
    finance_kb.learn(text, answer, cat)

# Router + Thinker
router  = DomainRouter()
thinker = Thinker(top_k=5, min_confidence=0.3, chain_threshold=0.6)

# Register KBs with router
router.register('general', general_kb)
router.register('finance', finance_kb)

# Text generation
vocab   = Vocabulary(max_size=500, min_freq=1)
decoder = None

# Stats
start_time  = time.time()
n_requests  = 0
n_questions = 0
n_learns    = 0
n_trains    = 0

print(f"✅ ZINOHK API v0.2.0 ready")
print(f"   Domains: {list(domain_kbs.keys())}")


# ------------------------------------------------------------------ #
# Request / Response models
# ------------------------------------------------------------------ #

class AskRequest(BaseModel):
    question:      str
    domain:        Optional[str] = None   # force a domain, or auto-route
    show_thinking: bool = False           # include reasoning trace

class AskResponse(BaseModel):
    question:   str
    answer:     str
    confidence: float
    domain:     str
    method:     str
    latency_ms: float
    thinking:   Optional[List[dict]] = None

class LearnRequest(BaseModel):
    text:     str
    answer:   str
    category: str  = "general"
    domain:   str  = "general"

class TrainRequest(BaseModel):
    domain: str
    facts:  List[Dict[str, str]]   # [{"text":..., "answer":..., "category":...}]

class TrainResponse(BaseModel):
    domain:      str
    facts_added: int
    facts_total: int
    latency_ms:  float

class GenerateRequest(BaseModel):
    seed:        str
    max_length:  int   = 15
    temperature: float = 0.8

class ClassifyRequest(BaseModel):
    text:   str
    labels: List[str] = ["positive", "negative"]

class StatsResponse(BaseModel):
    version:      str
    uptime_s:     float
    n_requests:   int
    n_questions:  int
    n_learns:     int
    n_trains:     int
    domains:      dict


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #

def make_retrieve_fn(kb: DynamicKnowledgeBase):
    """Create retrieve function for thinker from a KB."""
    def retrieve(query: str, top_k: int):
        results = kb.retrieve(query, top_k=top_k)
        return [
            {
                'title'     : fact.get('text', '')[:60],
                'answer'    : fact.get('answer', ''),
                'confidence': score,
                'category'  : fact.get('category', ''),
            }
            for score, fact in results
        ]
    return retrieve


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.get("/", tags=["System"])
def root():
    return {
        "name"       : "ZINOHK API",
        "version"    : "0.2.0",
        "status"     : "running",
        "domains"    : {d: len(kb._raw_texts)
                        for d, kb in domain_kbs.items()},
        "uptime_s"   : round(time.time() - start_time, 1),
        "endpoints"  : ["/ask", "/learn", "/train", "/classify",
                        "/generate", "/stats", "/domains",
                        "/knowledge"],
    }


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask(req: AskRequest):
    """
    Answer a question using ZINOHK's thinking layer.

    Auto-routes to the correct domain KB.
    Returns full reasoning trace if show_thinking=true.
    """
    global n_requests, n_questions
    n_requests  += 1
    n_questions += 1

    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    # Route to correct domain
    if req.domain and req.domain in domain_kbs:
        domain = req.domain
        kb     = domain_kbs[domain]
    else:
        classification = router.classify(req.question)
        domain_guess   = classification[0][0] if classification else 'general'
        domain         = domain_guess if domain_guess in domain_kbs else 'general'
        kb             = domain_kbs[domain]

    # Think before answering
    result = thinker.think(
        question    = req.question,
        retrieve_fn = make_retrieve_fn(kb),
        domain      = domain,
    )

    # Format thinking trace if requested
    thinking = None
    if req.show_thinking:
        thinking = [
            {
                'step'      : s.step,
                'action'    : s.action,
                'content'   : s.content,
                'confidence': s.confidence,
                'latency_ms': s.latency_ms,
            }
            for s in result.reasoning
        ]

    return AskResponse(
        question   = req.question,
        answer     = result.answer,
        confidence = result.confidence,
        domain     = result.domain,
        method     = result.method,
        latency_ms = result.latency_ms,
        thinking   = thinking,
    )


@app.post("/learn", tags=["Knowledge"])
def learn(req: LearnRequest):
    """Teach ZINOHK one new fact in any domain."""
    global n_requests, n_learns
    n_requests += 1
    n_learns   += 1

    kb = get_or_create_kb(req.domain)
    router.register(req.domain, kb)

    store_text = req.text + " " + req.text + " " + req.answer
    kb.learn(store_text, req.answer, req.category)

    return {
        "status"     : "learned",
        "domain"     : req.domain,
        "facts_total": len(kb._raw_texts),
        "message"    : f"Stored in '{req.domain}'. Total: {len(kb._raw_texts):,}",
    }


@app.post("/train", response_model=TrainResponse, tags=["Training"])
def train(req: TrainRequest):
    """
    Bulk train a domain from a list of facts.

    Universal training endpoint — works for any domain.
    Facts format: [{"text": "...", "answer": "...", "category": "..."}]

    Example:
      domain: "finance"
      facts: [
        {"text": "What is a put option? ...",
         "answer": "Right to sell shares",
         "category": "options_trading"},
        ...
      ]
    """
    global n_requests, n_trains
    n_requests += 1
    n_trains   += 1

    if not req.facts:
        raise HTTPException(400, "facts list cannot be empty")

    t0 = time.time()
    kb = get_or_create_kb(req.domain)
    router.register(req.domain, kb)

    added = 0
    for fact in req.facts:
        text     = fact.get('text', '').strip()
        answer   = fact.get('answer', '').strip()
        category = fact.get('category', req.domain)
        if text and answer:
            store_text = text + " " + text + " " + answer
            kb.learn(store_text, answer, category)
            added += 1

    return TrainResponse(
        domain      = req.domain,
        facts_added = added,
        facts_total = len(kb._raw_texts),
        latency_ms  = round((time.time()-t0)*1000, 2),
    )


@app.post("/classify", tags=["Classification"])
def classify(req: ClassifyRequest):
    """Zero-shot text classification."""
    global n_requests
    n_requests += 1

    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    import numpy as np
    t0     = time.time()
    scores = {}

    for label in req.labels:
        results = domain_kbs['general'].retrieve(
            f"{req.text} {label}", top_k=1)
        scores[label] = round(results[0][0], 4) if results else 0.0

    best = max(scores, key=scores.get)
    return {
        "text"      : req.text,
        "label"     : best,
        "confidence": scores[best],
        "scores"    : scores,
        "latency_ms": round((time.time()-t0)*1000, 2),
    }


@app.post("/generate", tags=["Generation"])
def generate(req: GenerateRequest):
    """Generate text from a seed word."""
    global n_requests, decoder
    n_requests += 1

    if not req.seed.strip():
        raise HTTPException(400, "Seed cannot be empty")

    if decoder is None:
        _init_decoder()

    decoder.temperature = req.temperature
    t0  = time.time()
    gen = decoder.generate(req.seed.lower(), max_len=req.max_length)
    return {
        "seed"      : req.seed,
        "generated" : gen or f"{req.seed}...",
        "latency_ms": round((time.time()-t0)*1000, 2),
    }


@app.get("/domains", tags=["System"])
def list_domains():
    """List all registered domains and their fact counts."""
    return {
        "domains": {
            d: {
                "facts"     : len(kb._raw_texts),
                "registered": d in router.registered,
            }
            for d, kb in domain_kbs.items()
        },
        "router_keywords": {
            d: len(kws)
            for d, kws in router.keyword_map.items()
            if d != 'general'
        },
    }


@app.get("/stats", response_model=StatsResponse, tags=["System"])
def stats():
    return StatsResponse(
        version     = "0.2.0",
        uptime_s    = round(time.time() - start_time, 1),
        n_requests  = n_requests,
        n_questions = n_questions,
        n_learns    = n_learns,
        n_trains    = n_trains,
        domains     = {d: len(kb._raw_texts)
                       for d, kb in domain_kbs.items()},
    )


@app.get("/knowledge", tags=["Knowledge"])
def list_knowledge(domain: str = "general", limit: int = 20):
    """List stored facts for a domain."""
    if domain not in domain_kbs:
        raise HTTPException(404, f"Domain '{domain}' not found")
    kb    = domain_kbs[domain]
    facts = []
    for i, (text, answer, cat) in enumerate(
        zip(kb._raw_texts, kb._raw_answers, kb._raw_cats)
    ):
        facts.append({
            "id"      : i,
            "text"    : text[:100],
            "answer"  : answer[:100],
            "category": cat,
        })
        if len(facts) >= limit:
            break
    return {"domain": domain, "total": len(kb._raw_texts),
            "facts": facts}


@app.delete("/knowledge/{fact_id}", tags=["Knowledge"])
def delete_fact(fact_id: int, domain: str = "general"):
    """Remove a fact by index from a domain."""
    if domain not in domain_kbs:
        raise HTTPException(404, f"Domain '{domain}' not found")
    kb = domain_kbs[domain]
    if fact_id < 0 or fact_id >= len(kb._raw_texts):
        raise HTTPException(404, "Fact not found")
    text = kb._raw_texts.pop(fact_id)
    kb._raw_answers.pop(fact_id)
    kb._raw_cats.pop(fact_id)
    kb._dirty   = True
    kb.n_added -= 1
    return {"status": "deleted", "domain": domain,
            "text": text[:80]}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _init_decoder():
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
