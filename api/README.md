# ZINOHK API — Complete Reference

Brain-inspired AI REST API. No backprop. Continuous learning.
Multi-domain knowledge: Wikipedia (6.4M facts) + Finance (96 facts).

## Start the server

```bash
HF_TOKEN="hf_..." uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Base URL
Local     : http://localhost:8000
Production: https://<tunnel>.trycloudflare.com  (Kaggle session)

---

## Endpoints

### GET /
Chat UI — opens interactive browser interface.
Returns HTML page for chatting with ZINOHK directly.

```bash
curl http://localhost:8000/
# Opens chat UI in browser
```

---

### POST /ask
Main Q&A endpoint. Auto-routes to correct domain.
Uses Wikipedia FAISS (6.4M facts) or Finance FAISS (96 facts).

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Who was Albert Einstein?"}'

# Force a domain
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is RSI?", "domain": "finance"}'

# Show thinking trace
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is delta?", "show_thinking": true}'
```

Request:
```json
{
  "question"      : "Who was Albert Einstein?",
  "domain"        : null,
  "show_thinking" : false
}
```

Response:
```json
{
  "question"   : "Who was Albert Einstein?",
  "answer"     : "Albert Einstein was a German-born theoretical physicist...",
  "confidence" : 1.0,
  "domain"     : "general",
  "method"     : "exact",
  "latency_ms" : 0.75,
  "thinking"   : null
}
```

Domain routing (automatic):
question contains stock/option/trade/rsi/macd... → finance
everything else                                   → general (Wikipedia)

---

### POST /learn
Teach ZINOHK one new fact instantly. No retraining needed.
Fact is immediately searchable.

```bash
curl -X POST http://localhost:8000/learn \
  -H 'Content-Type: application/json' \
  -d '{
    "text"    : "What is ZINOHK? ZINOHK is a brain-inspired AI",
    "answer"  : "Brain-inspired AI built from scratch in 2026",
    "category": "system",
    "domain"  : "general"
  }'
```

Request:
```json
{
  "text"    : "What is X? X is ...",
  "answer"  : "Short answer",
  "category": "general",
  "domain"  : "general"
}
```

Response:
```json
{
  "status"     : "learned",
  "domain"     : "general",
  "facts_total": 4,
  "message"    : "Stored in 'general'. Total: 4"
}
```

Domain options: `"general"`, `"finance"`, `"medicine"`,
`"science"`, `"tech"` or any custom string.

---

### POST /train
Bulk train a domain from a list of facts.
Use this to add an entire niche knowledge base at once.

```bash
curl -X POST http://localhost:8000/train \
  -H 'Content-Type: application/json' \
  -d '{
    "domain": "medicine",
    "facts": [
      {
        "text"    : "What is hypertension? High blood pressure above 140/90",
        "answer"  : "High blood pressure above 140/90 mmHg",
        "category": "cardiology"
      },
      {
        "text"    : "What is diabetes? Condition where blood sugar is too high",
        "answer"  : "Condition of chronically high blood sugar",
        "category": "endocrinology"
      }
    ]
  }'
```

Request:
```json
{
  "domain": "finance",
  "facts" : [
    {"text": "...", "answer": "...", "category": "..."},
    {"text": "...", "answer": "...", "category": "..."}
  ]
}
```

Response:
```json
{
  "domain"     : "finance",
  "facts_added": 2,
  "facts_total": 98,
  "latency_ms" : 0.07
}
```

This is the universal training endpoint. Same call works for
any domain — finance, medicine, legal, science, custom.

---

### POST /classify
Zero-shot text classification. No training needed.
Works with any labels you provide.

```bash
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{
    "text"  : "The stock market fell 3% today",
    "labels": ["business", "sports", "technology", "politics"]
  }'
```

Request:
```json
{
  "text"  : "Text to classify",
  "labels": ["label1", "label2", "label3"]
}
```

Response:
```json
{
  "text"      : "The stock market fell 3% today",
  "label"     : "business",
  "confidence": 0.219,
  "scores"    : {"business": 0.219, "sports": 0.1, "technology": 0.08},
  "latency_ms": 64.46
}
```

---

### POST /generate
Generate text from a seed phrase using the trigram decoder.
Trained on 800 Gutenberg books (113K sentences, 30K vocab).

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"seed": "the market", "max_length": 15, "temperature": 0.7}'
```

Request:
```json
{
  "seed"       : "the future of",
  "max_length" : 15,
  "temperature": 0.7
}
```

Response:
```json
{
  "seed"      : "the future of",
  "generated" : "the future of ai is brain inspired",
  "latency_ms": 0.5
}
```

Temperature: lower = more focused, higher = more creative.
Range: 0.3 (deterministic) → 1.5 (random)

---

### POST /session/clear
Clear conversation memory for a session.

```bash
curl -X POST http://localhost:8000/session/clear \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "user_123"}'
```

---

### GET /session/{session_id}
Get conversation history for a session.

```bash
curl http://localhost:8000/session/user_123
```

Response:
```json
{
  "session_id": "user_123",
  "turns"     : 5,
  "entities"  : ["Albert Einstein", "Amazon River"],
  "context"   : "Q: Who was Einstein?\nA: German physicist..."
}
```

---

### GET /domains
List all registered knowledge domains and their fact counts.

```bash
curl http://localhost:8000/domains
```

Response:
```json
{
  "domains": {
    "general": {"facts": 2, "registered": true},
    "finance": {"facts": 7, "registered": true}
  },
  "router_keywords": {
    "finance" : 50,
    "medicine": 20,
    "science" : 20,
    "tech"    : 21
  }
}
```

---

### GET /stats
System statistics and request counters.

```bash
curl http://localhost:8000/stats
```

Response:
```json
{
  "version"    : "0.2.0",
  "uptime_s"   : 128.2,
  "n_requests" : 42,
  "n_questions": 30,
  "n_learns"   : 5,
  "n_trains"   : 2,
  "domains"    : {"general": 2, "finance": 7}
}
```

---

### GET /health
Quick health check. Returns immediately.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status"    : "ready",
  "version"   : "0.2.0",
  "uptime_s"  : 14.7
}
```

---

## Interactive docs

Open in browser:
http://localhost:8000/docs    ← Swagger UI (test all endpoints)
http://localhost:8000/redoc   ← ReDoc (clean documentation)

---

## Training a new domain — step by step

### Step 1: Small domain on local CPU (< 1000 facts)

```bash
# Add via /train endpoint
curl -X POST http://localhost:8000/train \
  -H 'Content-Type: application/json' \
  -d '{
    "domain": "medicine",
    "facts": [
      {"text": "What is hypertension? ...", "answer": "...", "category": "cardiology"},
      {"text": "What is diabetes? ...",     "answer": "...", "category": "endocrinology"}
    ]
  }'
```

### Step 2: Large domain on Kaggle GPU (> 10K facts)

```python
# On Kaggle — encode dataset + build FAISS + upload to HuggingFace
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import HfApi

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Load your domain data
texts   = ["fact 1 text", "fact 2 text", ...]
answers = ["answer 1",    "answer 2",    ...]

# Encode on GPU
vecs = model.encode(texts, batch_size=512,
                     normalize_embeddings=True)

# Build FAISS
index = faiss.IndexFlatIP(384)
index.add(vecs)
faiss.write_index(index, "domain.index")

# Upload to HuggingFace
api = HfApi()
api.upload_file(path_or_fileobj="domain.index",
                path_in_repo="trained/domain.index",
                repo_id="orbitaven/zinohk",
                repo_type="model", token=HF_TOKEN)
```

### Step 3: Load in API

```python
# In api/loader.py — add loader function
def load_domain_kb(domain_name: str):
    index = faiss.read_index(_download(f"{domain_name}.index"))
    with open(_download(f"{domain_name}_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return index, meta
```

---

## Endpoint summary table

| Method | Endpoint           | Purpose                        | Speed    |
|--------|--------------------|--------------------------------|----------|
| GET    | /                  | Chat UI                        | instant  |
| POST   | /ask               | Q&A with domain routing        | 0-600ms  |
| POST   | /learn             | Add one fact                   | <1ms     |
| POST   | /train             | Bulk add facts                 | 0.07ms   |
| POST   | /classify          | Zero-shot classification       | 50-100ms |
| POST   | /generate          | Text generation                | <1ms     |
| POST   | /session/clear     | Clear conversation memory      | instant  |
| GET    | /session/{id}      | Get session history            | instant  |
| GET    | /domains           | List all domains               | instant  |
| GET    | /stats             | System statistics              | instant  |
| GET    | /health            | Health check                   | instant  |
| GET    | /docs              | Swagger UI                     | instant  |
