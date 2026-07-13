"""
ZINOHK REST API v0.3.0
========================
Multi-domain AI with conversation memory + chat UI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import time, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from zinohk.knowledge.learner   import DynamicKnowledgeBase
from zinohk.encoding.vocab      import Vocabulary
from zinohk.encoding.decoder    import NGramDecoder
from zinohk.router.domain       import DomainRouter
from zinohk.router.thinker      import Thinker
from zinohk.memory.conversation import SessionStore

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="ZINOHK API", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Init ──────────────────────────────────────────────────────
print("Initialising ZINOHK v0.3.0...")

domain_kbs: Dict[str, DynamicKnowledgeBase] = {}

def get_or_create_kb(domain):
    if domain not in domain_kbs:
        domain_kbs[domain] = DynamicKnowledgeBase(vocab_size=2000)
    return domain_kbs[domain]

general_kb = get_or_create_kb('general')
finance_kb = get_or_create_kb('finance')

# Seed facts
for text, answer, cat in [
    ("What is ZINOHK? ZINOHK is a brain-inspired AI built in 2026",
     "ZINOHK is a brain-inspired AI architecture", "system"),
    ("What makes ZINOHK different? ZINOHK uses sparse neurons Hebbian learning no backpropagation",
     "Sparse neurons, Hebbian learning, zero backpropagation", "system"),
]:
    general_kb.learn(text, answer, cat)

for text, answer, cat in [
    ("What is a call option? A call option gives right to buy shares at strike price",
     "Right to buy shares at the strike price before expiry", "options"),
    ("What is swing trading? Swing trading holds positions days to weeks",
     "Holding trades for days to weeks to capture price moves", "trading"),
    ("What is RSI? RSI relative strength index measures momentum 0 to 100",
     "Momentum oscillator: above 70 overbought, below 30 oversold", "trading"),
    ("What is a covered call? Covered call sells call on owned shares for income",
     "Selling a call option on shares you own to earn premium income", "options"),
]:
    finance_kb.learn(text, answer, cat)

router       = DomainRouter()
thinker      = Thinker(top_k=5, min_confidence=0.3, chain_threshold=0.6)
sessions     = SessionStore(max_turns=20, inactivity_timeout=7200)
vocab        = Vocabulary(max_size=500, min_freq=1)
decoder      = None
start_time   = time.time()
n_requests   = 0

router.register('general', general_kb)
router.register('finance', finance_kb)

print(f"✅ ZINOHK API v0.3.0 ready | domains={list(domain_kbs.keys())}")

# ── Models ────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question:      str
    session_id:    str   = "default"
    domain:        Optional[str] = None
    show_thinking: bool  = False

class LearnRequest(BaseModel):
    text:      str
    answer:    str
    category:  str = "general"
    domain:    str = "general"

class TrainRequest(BaseModel):
    domain: str
    facts:  List[Dict[str, str]]

class GenerateRequest(BaseModel):
    seed:        str
    max_length:  int   = 15
    temperature: float = 0.8

class ClassifyRequest(BaseModel):
    text:   str
    labels: List[str] = ["positive", "negative"]

class ClearSessionRequest(BaseModel):
    session_id: str

# ── Helpers ───────────────────────────────────────────────────
def make_retrieve_fn(kb):
    def retrieve(query, top_k):
        results = kb.retrieve(query, top_k=top_k)
        return [{'title': f['text'][:60], 'answer': f['answer'],
                 'confidence': s, 'category': f['category']}
                for s, f in results]
    return retrieve

def _init_decoder():
    global decoder
    corpus = [
        "the sky is blue and clear today",
        "knowledge grows from every interaction",
        "the brain learns from every experience",
        "neurons fire and wire together",
        "sparse activation saves compute power",
        "intelligence emerges from simple local rules",
        "every question makes the system smarter",
        "the future of AI is brain inspired",
        "data flows through the network like water",
        "learning happens at the edge of the network",
    ]
    vocab.build(corpus)
    decoder = NGramDecoder(vocab=vocab, lr=0.15, temperature=0.8)
    decoder.train(corpus, epochs=300)

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def chat_ui():
    """ZINOHK Chat UI — works like ChatGPT/Claude."""
    return HTMLResponse(content=CHAT_HTML)

@app.post("/ask", tags=["Q&A"])
def ask(req: AskRequest):
    global n_requests
    n_requests += 1

    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    # Get session conversation buffer
    buf = sessions.get_or_create(req.session_id)

    # Enrich question with conversation context
    enriched = buf.enrich_question(req.question)
    is_followup = buf.is_followup(req.question)

    # Route to correct domain
    if req.domain and req.domain in domain_kbs:
        domain = req.domain
        kb     = domain_kbs[domain]
    else:
        ranked = router.classify(enriched)
        domain = ranked[0][0] if ranked else 'general'
        if domain not in domain_kbs:
            domain = 'general'
        kb = domain_kbs[domain]

    # Think before answering
    result = thinker.think(
        question    = enriched,
        retrieve_fn = make_retrieve_fn(kb),
        domain      = domain,
    )

    # Store turn in session memory
    buf.add(
        question   = req.question,
        answer     = result.answer,
        domain     = domain,
        confidence = result.confidence,
    )

    thinking = None
    if req.show_thinking:
        thinking = [
            {'step': s.step, 'action': s.action,
             'content': s.content, 'ms': s.latency_ms}
            for s in result.reasoning
        ]

    return {
        "question"       : req.question,
        "enriched"       : enriched if enriched != req.question else None,
        "answer"         : result.answer,
        "confidence"     : result.confidence,
        "domain"         : domain,
        "method"         : result.method,
        "latency_ms"     : result.latency_ms,
        "session_id"     : req.session_id,
        "turn"           : buf.turn_counter,
        "is_followup"    : is_followup,
        "thinking"       : thinking,
    }

@app.post("/learn", tags=["Knowledge"])
def learn(req: LearnRequest):
    global n_requests
    n_requests += 1
    kb = get_or_create_kb(req.domain)
    router.register(req.domain, kb)
    store = req.text + " " + req.text + " " + req.answer
    kb.learn(store, req.answer, req.category)
    return {"status": "learned", "domain": req.domain,
            "facts_total": len(kb._raw_texts)}

@app.post("/train", tags=["Training"])
def train(req: TrainRequest):
    global n_requests
    n_requests += 1
    t0 = time.time()
    kb = get_or_create_kb(req.domain)
    router.register(req.domain, kb)
    added = 0
    for fact in req.facts:
        text = fact.get('text','').strip()
        ans  = fact.get('answer','').strip()
        cat  = fact.get('category', req.domain)
        if text and ans:
            kb.learn(text+" "+text+" "+ans, ans, cat)
            added += 1
    return {"domain": req.domain, "added": added,
            "total": len(kb._raw_texts),
            "ms": round((time.time()-t0)*1000, 2)}

@app.post("/classify", tags=["Classification"])
def classify(req: ClassifyRequest):
    global n_requests
    n_requests += 1
    t0 = time.time()
    scores = {}
    for label in req.labels:
        results = general_kb.retrieve(f"{req.text} {label}", top_k=1)
        scores[label] = round(results[0][0], 4) if results else 0.0
    best = max(scores, key=scores.get)
    return {"text": req.text, "label": best,
            "confidence": scores[best], "scores": scores,
            "ms": round((time.time()-t0)*1000, 2)}

@app.post("/generate", tags=["Generation"])
def generate(req: GenerateRequest):
    global n_requests, decoder
    n_requests += 1
    if not req.seed.strip():
        raise HTTPException(400, "Seed cannot be empty")
    if decoder is None:
        _init_decoder()
    decoder.temperature = req.temperature
    t0  = time.time()
    gen = decoder.generate(req.seed.lower(), max_len=req.max_length)
    return {"seed": req.seed, "generated": gen or f"{req.seed}...",
            "ms": round((time.time()-t0)*1000, 2)}

@app.post("/session/clear", tags=["Session"])
def clear_session(req: ClearSessionRequest):
    cleared = sessions.clear(req.session_id)
    return {"cleared": cleared, "session_id": req.session_id}

@app.get("/session/{session_id}", tags=["Session"])
def get_session(session_id: str):
    buf = sessions.get_or_create(session_id)
    return {
        "session_id": session_id,
        "turns"     : buf.turn_counter,
        "context"   : buf.get_context(n_turns=3),
        "topic"     : buf.current_topic(),
        "entities"  : buf.get_recent_entities(),
    }

@app.get("/domains", tags=["System"])
def list_domains():
    return {"domains": {d: {"facts": len(kb._raw_texts),
                            "registered": d in router.registered}
                        for d, kb in domain_kbs.items()}}

@app.get("/stats", tags=["System"])
def stats():
    return {
        "version"   : "0.3.0",
        "uptime_s"  : round(time.time()-start_time, 1),
        "requests"  : n_requests,
        "domains"   : {d: len(kb._raw_texts) for d, kb in domain_kbs.items()},
        "sessions"  : sessions.stats(),
    }

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "0.3.0"}

# ── Chat UI HTML ──────────────────────────────────────────────
CHAT_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZINOHK — Brain-inspired AI</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f0f0f; --surface: #1a1a1a; --surface2: #242424;
    --border: #2a2a2a; --text: #e8e8e8; --muted: #888;
    --accent: #7c6af7; --accent-dim: #4a3fa0;
    --success: #1d9e75; --user-bg: #1e1b4b;
    --ai-bg: #1a1a1a; --radius: 12px;
  }
  body { background: var(--bg); color: var(--text);
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         height: 100vh; display: flex; flex-direction: column; }

  /* Header */
  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 14px 20px; display: flex; align-items: center;
           justify-content: space-between; flex-shrink: 0; }
  .logo { display: flex; align-items: center; gap: 10px; }
  .logo-icon { width: 32px; height: 32px; background: var(--accent);
               border-radius: 8px; display: flex; align-items: center;
               justify-content: center; font-size: 16px; }
  .logo-name { font-size: 17px; font-weight: 600; }
  .logo-tag  { font-size: 11px; color: var(--muted); margin-top: 1px; }
  .header-right { display: flex; align-items: center; gap: 12px; }
  .domain-badge { background: var(--surface2); border: 1px solid var(--border);
                  border-radius: 20px; padding: 4px 12px; font-size: 12px;
                  color: var(--muted); }
  .domain-badge span { color: var(--accent); font-weight: 500; }
  .clear-btn { background: none; border: 1px solid var(--border);
               color: var(--muted); padding: 6px 14px; border-radius: 8px;
               cursor: pointer; font-size: 12px; transition: all .2s; }
  .clear-btn:hover { border-color: var(--accent); color: var(--text); }

  /* Chat area */
  #chat { flex: 1; overflow-y: auto; padding: 24px 0; }
  #chat::-webkit-scrollbar { width: 6px; }
  #chat::-webkit-scrollbar-track { background: transparent; }
  #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .msg-wrap { max-width: 760px; margin: 0 auto; padding: 0 20px 4px; }

  /* Messages */
  .msg { display: flex; gap: 12px; margin-bottom: 20px; animation: fadeIn .3s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; } }
  .msg.user { flex-direction: row-reverse; }

  .avatar { width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 14px; font-weight: 600; margin-top: 2px; }
  .avatar.ai   { background: var(--accent); color: #fff; }
  .avatar.user { background: var(--user-bg); color: var(--accent);
                 border: 1px solid var(--accent-dim); }

  .bubble { max-width: calc(100% - 50px); }
  .bubble-name { font-size: 11px; color: var(--muted); margin-bottom: 5px;
                 font-weight: 500; }
  .msg.user .bubble-name { text-align: right; }

  .bubble-content { background: var(--ai-bg); border: 1px solid var(--border);
                    border-radius: var(--radius); padding: 12px 16px;
                    font-size: 14px; line-height: 1.65; }
  .msg.user .bubble-content { background: var(--user-bg);
                               border-color: var(--accent-dim); }

  .bubble-meta { margin-top: 5px; display: flex; gap: 10px;
                 font-size: 11px; color: var(--muted); align-items: center; }
  .msg.user .bubble-meta { justify-content: flex-end; }

  .domain-tag { background: var(--surface2); border: 1px solid var(--border);
                border-radius: 10px; padding: 1px 7px; font-size: 11px; }
  .conf-bar { width: 40px; height: 3px; background: var(--border);
              border-radius: 2px; overflow: hidden; }
  .conf-fill { height: 100%; background: var(--success); border-radius: 2px; }

  /* Thinking trace */
  .thinking { margin-top: 8px; background: var(--surface2);
              border: 1px solid var(--border); border-radius: 8px;
              overflow: hidden; }
  .thinking-header { padding: 8px 12px; font-size: 11px; color: var(--muted);
                     cursor: pointer; display: flex; align-items: center;
                     gap: 6px; user-select: none; }
  .thinking-header:hover { color: var(--text); }
  .thinking-steps { padding: 0 12px 10px; display: none; }
  .thinking-steps.open { display: block; }
  .step { padding: 4px 0; font-size: 11px; color: var(--muted);
          display: flex; gap: 8px; align-items: baseline; }
  .step-tag { background: var(--border); border-radius: 4px;
              padding: 1px 6px; font-size: 10px; color: var(--accent);
              flex-shrink: 0; }

  /* Welcome */
  .welcome { text-align: center; padding: 60px 20px 40px; }
  .welcome h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
  .welcome p  { color: var(--muted); font-size: 14px; max-width: 420px;
                margin: 0 auto 32px; line-height: 1.6; }
  .examples { display: flex; flex-wrap: wrap; gap: 8px;
              justify-content: center; max-width: 600px; margin: 0 auto; }
  .example { background: var(--surface); border: 1px solid var(--border);
             border-radius: 20px; padding: 8px 16px; font-size: 13px;
             cursor: pointer; transition: all .2s; color: var(--text); }
  .example:hover { border-color: var(--accent); background: var(--surface2); }

  /* Typing indicator */
  .typing { display: flex; gap: 4px; align-items: center; padding: 4px 0; }
  .dot { width: 6px; height: 6px; background: var(--muted);
         border-radius: 50%; animation: bounce .8s infinite; }
  .dot:nth-child(2) { animation-delay: .15s; }
  .dot:nth-child(3) { animation-delay: .3s; }
  @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-6px)} }

  /* Input */
  .input-area { background: var(--surface); border-top: 1px solid var(--border);
                padding: 16px 20px; flex-shrink: 0; }
  .input-wrap { max-width: 760px; margin: 0 auto; position: relative; }
  #input { width: 100%; background: var(--surface2); border: 1px solid var(--border);
           border-radius: var(--radius); padding: 12px 52px 12px 16px;
           color: var(--text); font-size: 14px; resize: none;
           font-family: inherit; line-height: 1.5; max-height: 140px;
           outline: none; transition: border-color .2s; }
  #input:focus { border-color: var(--accent); }
  #input::placeholder { color: var(--muted); }
  #send { position: absolute; right: 10px; bottom: 10px;
          width: 34px; height: 34px; background: var(--accent);
          border: none; border-radius: 8px; cursor: pointer;
          display: flex; align-items: center; justify-content: center;
          transition: all .2s; color: white; }
  #send:hover { background: var(--accent-dim); }
  #send:disabled { background: var(--border); cursor: not-allowed; }
  #send svg { width: 16px; height: 16px; }

  .input-footer { max-width: 760px; margin: 8px auto 0;
                  font-size: 11px; color: var(--muted);
                  display: flex; justify-content: space-between; }

  /* Capabilities strip */
  .caps { display: flex; gap: 16px; flex-wrap: wrap; }
  .cap { display: flex; align-items: center; gap: 4px; }
  .cap-dot { width: 6px; height: 6px; border-radius: 50%; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">Z</div>
    <div>
      <div class="logo-name">ZINOHK</div>
      <div class="logo-tag">Brain-inspired AI</div>
    </div>
  </div>
  <div class="header-right">
    <div class="domain-badge">domain: <span id="current-domain">auto</span></div>
    <button class="clear-btn" onclick="clearChat()">New chat</button>
  </div>
</header>

<div id="chat">
  <div class="msg-wrap">
    <div class="welcome" id="welcome">
      <h1>ZINOHK</h1>
      <p>Brain-inspired AI with sparse neurons, Hebbian learning, and real Wikipedia knowledge. No backpropagation.</p>
      <div class="examples">
        <button class="example" onclick="sendExample(this)">Who was Albert Einstein?</button>
        <button class="example" onclick="sendExample(this)">What is a call option?</button>
        <button class="example" onclick="sendExample(this)">What is the Amazon River?</button>
        <button class="example" onclick="sendExample(this)">What is swing trading?</button>
        <button class="example" onclick="sendExample(this)">What is machine learning?</button>
        <button class="example" onclick="sendExample(this)">What is ZINOHK?</button>
      </div>
    </div>
  </div>
</div>

<div class="input-area">
  <div class="input-wrap">
    <textarea id="input" rows="1" placeholder="Ask anything..."
              onkeydown="handleKey(event)" oninput="autoResize()"></textarea>
    <button id="send" onclick="sendMessage()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M22 2L11 13M22 2L15 22l-4-9-9-4 20-7z"/>
      </svg>
    </button>
  </div>
  <div class="input-footer">
    <div class="caps">
      <div class="cap"><div class="cap-dot" style="background:#7c6af7"></div>6.4M Wikipedia facts</div>
      <div class="cap"><div class="cap-dot" style="background:#1d9e75"></div>Finance domain</div>
      <div class="cap"><div class="cap-dot" style="background:#ef9f27"></div>Conversation memory</div>
      <div class="cap"><div class="cap-dot" style="background:#e24b4a"></div>No backprop</div>
    </div>
    <div id="turn-count" style="color:var(--muted)">turn 0</div>
  </div>
</div>

<script>
const SESSION = 'user_' + Math.random().toString(36).slice(2, 9);
let turn = 0;
let waiting = false;

function autoResize() {
  const el = document.getElementById('input');
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function sendExample(btn) {
  document.getElementById('input').value = btn.textContent;
  sendMessage();
}

function sendMessage() {
  const input = document.getElementById('input');
  const q = input.value.trim();
  if (!q || waiting) return;

  hideWelcome();
  appendUser(q);
  input.value = '';
  input.style.height = 'auto';

  const typingId = appendTyping();
  waiting = true;
  document.getElementById('send').disabled = true;

  fetch('/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      question: q,
      session_id: SESSION,
      show_thinking: true,
    })
  })
  .then(r => r.json())
  .then(data => {
    removeTyping(typingId);
    appendAI(data);
    turn = data.turn || turn + 1;
    document.getElementById('turn-count').textContent = 'turn ' + turn;
    document.getElementById('current-domain').textContent = data.domain || 'general';
    waiting = false;
    document.getElementById('send').disabled = false;
  })
  .catch(err => {
    removeTyping(typingId);
    appendError("Connection error — is the server running?");
    waiting = false;
    document.getElementById('send').disabled = false;
  });
}

function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.style.display = 'none';
}

function appendUser(text) {
  const chat = document.getElementById('chat');
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap';
  wrap.innerHTML = `
    <div class="msg user">
      <div class="avatar user">U</div>
      <div class="bubble">
        <div class="bubble-name">You</div>
        <div class="bubble-content">${escHtml(text)}</div>
      </div>
    </div>`;
  chat.appendChild(wrap);
  scrollBottom();
}

function appendTyping() {
  const id = 'typing-' + Date.now();
  const chat = document.getElementById('chat');
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap';
  wrap.id = id;
  wrap.innerHTML = `
    <div class="msg">
      <div class="avatar ai">Z</div>
      <div class="bubble">
        <div class="bubble-name">ZINOHK</div>
        <div class="bubble-content">
          <div class="typing">
            <div class="dot"></div><div class="dot"></div><div class="dot"></div>
          </div>
        </div>
      </div>
    </div>`;
  chat.appendChild(wrap);
  scrollBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendAI(data) {
  const chat  = document.getElementById('chat');
  const wrap  = document.createElement('div');
  wrap.className = 'msg-wrap';
  const conf  = Math.round((data.confidence || 0) * 100);
  const confColor = conf > 70 ? '#1d9e75' : conf > 40 ? '#ef9f27' : '#e24b4a';
  const enriched  = data.enriched
    ? `<div style="font-size:11px;color:var(--muted);margin-bottom:6px;">
         interpreted as: <em>${escHtml(data.enriched)}</em></div>` : '';

  let thinkingHtml = '';
  if (data.thinking && data.thinking.length) {
    const steps = data.thinking.map(s =>
      `<div class="step">
         <span class="step-tag">${s.action}</span>
         <span>${escHtml(s.content)}</span>
         <span style="margin-left:auto;opacity:.5">${s.ms.toFixed(1)}ms</span>
       </div>`
    ).join('');
    thinkingHtml = `
      <div class="thinking">
        <div class="thinking-header" onclick="toggleThinking(this)">
          🤔 Thinking trace (${data.thinking.length} steps · ${data.latency_ms}ms)
          <span style="margin-left:auto">▶</span>
        </div>
        <div class="thinking-steps">${steps}</div>
      </div>`;
  }

  wrap.innerHTML = `
    <div class="msg">
      <div class="avatar ai">Z</div>
      <div class="bubble">
        <div class="bubble-name">ZINOHK</div>
        <div class="bubble-content">
          ${enriched}
          ${escHtml(data.answer)}
          ${thinkingHtml}
        </div>
        <div class="bubble-meta">
          <div class="conf-bar"><div class="conf-fill" style="width:${conf}%;background:${confColor}"></div></div>
          <span>${conf}% confidence</span>
          <span class="domain-tag">${data.domain}</span>
          <span>${data.latency_ms}ms</span>
          ${data.is_followup ? '<span style="color:var(--accent)">↩ follow-up</span>' : ''}
        </div>
      </div>
    </div>`;
  chat.appendChild(wrap);
  scrollBottom();
}

function appendError(msg) {
  const chat = document.getElementById('chat');
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap';
  wrap.innerHTML = `
    <div class="msg">
      <div class="avatar ai" style="background:#e24b4a">Z</div>
      <div class="bubble">
        <div class="bubble-content" style="border-color:#e24b4a;color:#f09595">
          ${escHtml(msg)}
        </div>
      </div>
    </div>`;
  chat.appendChild(wrap);
  scrollBottom();
}

function toggleThinking(header) {
  const steps = header.nextElementSibling;
  const arrow = header.querySelector('span:last-child');
  steps.classList.toggle('open');
  arrow.textContent = steps.classList.contains('open') ? '▼' : '▶';
}

function clearChat() {
  fetch('/session/clear', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({session_id: SESSION})
  });
  const chat = document.getElementById('chat');
  chat.innerHTML = `<div class="msg-wrap"><div class="welcome" id="welcome">
    <h1>ZINOHK</h1>
    <p>Brain-inspired AI with sparse neurons, Hebbian learning, and real Wikipedia knowledge.</p>
    <div class="examples">
      <button class="example" onclick="sendExample(this)">Who was Albert Einstein?</button>
      <button class="example" onclick="sendExample(this)">What is a call option?</button>
      <button class="example" onclick="sendExample(this)">What is the Amazon River?</button>
      <button class="example" onclick="sendExample(this)">What is swing trading?</button>
    </div>
  </div></div>`;
  turn = 0;
  document.getElementById('turn-count').textContent = 'turn 0';
  document.getElementById('current-domain').textContent = 'auto';
}

function scrollBottom() {
  const chat = document.getElementById('chat');
  chat.scrollTop = chat.scrollHeight;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>'''


# ------------------------------------------------------------------ #
# Load trained data from HuggingFace on startup
# ------------------------------------------------------------------ #

import os
import numpy as np
import faiss
from api.loader import load_finance_kb, load_trigram, load_wiki_index

# HF token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Load finance KB
print("Loading trained data from HuggingFace...")
try:
    _fin_index, _fin_meta = load_finance_kb()
    fin_titles    = _fin_meta['titles']
    fin_answers   = _fin_meta['answers']
    fin_summaries = _fin_meta['summaries']
    fin_cats      = _fin_meta['categories']
    print(f"✅ Finance KB: {_fin_index.ntotal} facts")
except Exception as e:
    print(f"⚠️  Finance KB load failed: {e}")
    _fin_index = None

# Load trigram model
try:
    _tri_data = load_trigram()
    print(f"✅ Trigram: {len(_tri_data['vocab_word2idx'])} words")
except Exception as e:
    print(f"⚠️  Trigram load failed: {e}")
    _tri_data = None

# Wire finance KB into router
if _fin_index is not None:
    # Add finance facts to finance domain KB
    for i, (text, answer, cat) in enumerate(
        zip(fin_titles, fin_answers, fin_cats)
    ):
        finance_kb.learn(
            f"{text} {text} {answer}",
            answer, cat
        )
    print(f"✅ Finance KB wired into router")
