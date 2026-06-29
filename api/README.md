# ZINOHK API

Brain-inspired AI available as a REST API.
No GPU needed. Runs on any machine.

## Start the server

```bash
pip install fastapi uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Endpoints

### Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the capital of France?"}'
```

### Teach a new fact
```bash
curl -X POST http://localhost:8000/learn \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "What is Python? Python is a programming language",
    "answer": "Python is a programming language",
    "category": "technology"
  }'
```

### Classify text
```bash
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "The stock market fell 3% today",
    "labels": ["business", "sports", "technology"]
  }'
```

### Generate text
```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"seed": "the brain", "max_length": 15}'
```

### System stats
```bash
curl http://localhost:8000/stats
```

## Interactive docs
Open http://localhost:8000/docs in your browser.
Full Swagger UI — test every endpoint interactively.

## Key properties
- No GPU required
- Learns from every /learn call — no retraining
- Wikipedia fallback for unknown questions
- 0.5–2ms response time on known facts
- Continuous learning — gets smarter over time
