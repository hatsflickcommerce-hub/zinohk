"""
zinohk.reasoning.retrieval
===========================
Improved retrieval — Fix 2, 3, 4.

Fix 2: Better entity extraction
  - Keep full proper nouns intact
  - Smarter question parsing
  - Handle possessives correctly

Fix 3: Answer extraction from article text
  - Find the specific sentence that answers the question
  - Don't return the full article dump

Fix 4: Re-rank by keyword overlap
  - Boost results where query words appear in title
  - Reduces irrelevant article returns
"""

import re
from typing import List, Dict, Optional, Tuple


# ── Fix 2: Better entity extraction ──────────────────────────

def extract_entity_v2(question: str) -> str:
    """
    Extract search entity from question.

    Key improvements over v1:
    - Keeps full name intact (Albert Einstein not Einstein)
    - Handles possessives (Einstein's wife → Albert Einstein)
    - Keeps context words for disambiguation
    """
    q = question.strip().rstrip('?')

    # Possessive: "Einstein's wife" → search "Einstein"
    # But keep full name if present: "Albert Einstein's wife"
    poss = re.search(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s", q)
    if poss:
        return poss.group(1)

    q_lower = q.lower()

    # Specific patterns — ordered from most to least specific
    patterns = [
        # "capital of X" → search "X capital"
        (r"(?:what is )?the capital (?:city )?of (.+)",
         lambda m: f"{m.group(1).strip()} capital"),

        # "Who was/is X" → search X directly
        (r"^who (?:was|is) (.+)",
         lambda m: m.group(1).strip()),

        # "Who invented/discovered/founded X" → search X
        (r"^who (?:invented|discovered|founded|created|built) (.+)",
         lambda m: m.group(1).strip()),

        # "What is X?" where X is a proper concept
        (r"^what (?:is|are|was|were) (.+)",
         lambda m: m.group(1).strip()),

        # "Where is X?"
        (r"^where (?:is|was|are) (.+)",
         lambda m: m.group(1).strip()),

        # "When was X?"
        (r"^when (?:was|is|did) (.+)",
         lambda m: m.group(1).strip()),

        # "How tall/long/old is X?"
        (r"^how (?:tall|long|old|big|far|fast) (?:is|was) (?:the )?(.+)",
         lambda m: m.group(1).strip()),
    ]

    for pattern, extractor in patterns:
        m = re.match(pattern, q_lower)
        if m:
            entity = extractor(m)
            # Remove leading AND trailing stopwords
            entity = re.sub(r'^(the|a|an)\s+', '', entity,
                             flags=re.IGNORECASE).strip()
            entity = re.sub(r'\s+(the|a|an|in|of|for|'
                             r'trading|market|finance|'
                             r'today|now|currently)$',
                             '', entity,
                             flags=re.IGNORECASE).strip()
            # Preserve known acronyms (DNA, RSI, etc.)
            ACRONYMS = {'dna','rna','rsi','macd','etf','ai','ml',
                        'cpu','gpu','gps','dna','iq','uk','usa',
                        'ussr','un','nasa','atp','atp'}
            words = []
            for w in entity.split():
                if w.lower() in ACRONYMS:
                    words.append(w.upper())
                else:
                    words.append(w.capitalize())
            return ' '.join(words)

    # Fallback — return cleaned question
    return q


# ── Fix 3: Answer extraction ──────────────────────────────────

def extract_answer_from_text(
    text:     str,
    question: str,
    entity:   str,
) -> str:
    """
    Extract specific answer sentence from article text.

    Instead of returning the full article dump,
    find the sentence most likely to answer the question.

    Strategy:
    1. Find sentences containing question keywords
    2. Prefer shorter, more specific sentences
    3. Fall back to first sentence if nothing matches
    """
    if not text:
        return text

    q_words = set(re.findall(r'\b\w+\b', question.lower())) - {
        'what', 'is', 'the', 'a', 'an', 'who', 'how', 'when',
        'where', 'was', 'did', 'does', 'do', 'of', 'in', 'on',
        'are', 'were', 'which', 'that', 'this',
    }

    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 20]

    if not sents:
        return text[:300]

    # Score each sentence
    scored = []
    for sent in sents:
        s_words = set(re.findall(r'\b\w+\b', sent.lower()))
        overlap = len(q_words & s_words)
        # Bonus for entity mention
        entity_bonus = 2 if entity.lower() in sent.lower() else 0
        # Penalty for very long sentences (less specific)
        length_penalty = len(sent) / 500
        score = overlap + entity_bonus - length_penalty
        scored.append((score, sent))

    scored.sort(reverse=True)
    best = scored[0][1]

    # If best sentence is too long, truncate
    if len(best) > 300:
        best = best[:297] + '...'

    return best


# ── Fix 4: Re-rank by keyword overlap ────────────────────────

def rerank_results(
    results:  List[Dict],
    question: str,
    entity:   str,
    top_k:    int = 3,
) -> List[Dict]:
    """
    Re-rank FAISS results by keyword overlap with query.

    FAISS returns semantic similarity — good but not perfect.
    Re-ranking boosts results where query words appear in title.

    Parameters
    ----------
    results  : raw FAISS results (title, answer, confidence)
    question : original question
    entity   : extracted entity
    top_k    : final number of results to return
    """
    if not results:
        return results

    q_words = set(re.findall(r'\b\w+\b',
                              (question + ' ' + entity).lower())) - {
        'what', 'is', 'the', 'a', 'an', 'who', 'how', 'when',
        'where', 'was', 'did', 'does', 'do', 'of', 'in',
    }

    reranked = []
    for r in results:
        title_words = set(re.findall(r'\b\w+\b',
                                      r.get('title','').lower()))
        overlap     = len(q_words & title_words)

        # Boost: exact entity match in title
        entity_match = 1.0 if entity.lower() in r.get(
            'title','').lower() else 0.0

        # Final score: FAISS score + overlap boost
        final_score = (
            r.get('confidence', 0.0) +
            overlap * 0.05 +
            entity_match * 0.2
        )
        reranked.append({**r, 'reranked_score': final_score})

    reranked.sort(key=lambda x: x['reranked_score'], reverse=True)
    return reranked[:top_k]


# ── Combined improved retrieval ───────────────────────────────

def improved_retrieve(
    question:     str,
    faiss_search_fn,   # function(query_vec, k) → (scores, indices)
    encode_fn,         # function(text) → vector
    titles:       list,
    summaries:    list,
    title_lower:  dict,
    top_k:        int = 5,
) -> List[Dict]:
    """
    Full improved retrieval pipeline.

    1. Extract entity (Fix 2)
    2. Try exact/prefix title match first
    3. FAISS semantic search
    4. Re-rank results (Fix 4)
    5. Extract specific answer (Fix 3)
    """
    import numpy as np

    entity = extract_entity_v2(question)
    el     = entity.lower()

    # Exact title match — instant and reliable
    if el in title_lower:
        idx     = title_lower[el]
        raw_ans = summaries[idx]
        answer  = extract_answer_from_text(raw_ans, question, entity)
        return [{
            'title'     : titles[idx],
            'answer'    : answer,
            'confidence': 1.0,
            'source'    : 'exact_title',
            'entity'    : entity,
        }]

    # Prefix match — reliable for "Amazon River" etc.
    for tl, idx in title_lower.items():
        if tl.startswith(el) and len(el) > 5:
            raw_ans = summaries[idx]
            answer  = extract_answer_from_text(
                raw_ans, question, entity)
            return [{
                'title'     : titles[idx],
                'answer'    : answer,
                'confidence': 0.95,
                'source'    : 'prefix_title',
                'entity'    : entity,
            }]

    # FAISS semantic search
    q_vec = encode_fn(entity)
    scores, indices = faiss_search_fn(
        q_vec.astype(np.float32), top_k * 2)

    raw_results = []
    for s, i in zip(scores[0], indices[0]):
        if i < 0: continue
        raw_results.append({
            'title'     : titles[i],
            'answer'    : summaries[i],
            'confidence': round(float(s), 4),
            'source'    : 'semantic',
            'entity'    : entity,
        })

    # Re-rank (Fix 4)
    reranked = rerank_results(raw_results, question, entity, top_k)

    # Extract specific answers (Fix 3)
    final = []
    for r in reranked:
        answer = extract_answer_from_text(
            r['answer'], question, entity)
        final.append({**r, 'answer': answer})

    return final
