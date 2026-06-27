"""
zinohk.knowledge.retriever
===========================
Hybrid retriever — local first, internet fallback.

Option C architecture:
  1. Try local KnowledgeBase first (fast, offline)
  2. If confidence < threshold → fetch from Wikipedia API
  3. Cache fetched facts into local base permanently
  4. Next time same question → answered locally (free)
"""

import re
import json
import urllib.request
import urllib.parse
from typing import Optional, List, Tuple
from zinohk.knowledge.base import KnowledgeBase, Fact


# ------------------------------------------------------------------ #
# Abbreviation expansion
# ------------------------------------------------------------------ #
ABBREVIATIONS = {
    'ww2' : 'World War Two', 'ww1' : 'World War One',
    'wwii': 'World War Two', 'wwi' : 'World War One',
    'usa' : 'United States of America',
    'uk'  : 'United Kingdom',
    'ussr': 'Soviet Union',
    'dna' : 'deoxyribonucleic acid DNA',
    'rna' : 'ribonucleic acid RNA',
    'ai'  : 'artificial intelligence',
    'ml'  : 'machine learning',
    'cpu' : 'central processing unit',
    'gpu' : 'graphics processing unit',
    'h2o' : 'water H2O hydrogen oxygen',
    'co2' : 'carbon dioxide CO2',
    'gps' : 'global positioning system',
    'nasa': 'National Aeronautics Space Administration',
}


def expand_abbreviations(text: str) -> str:
    words  = text.split()
    result = []
    for w in words:
        clean = w.lower().strip('?.,!:;')
        result.append(ABBREVIATIONS.get(clean, w))
    return ' '.join(result)


def normalise_query(query: str) -> str:
    return re.sub(r'\s+', ' ',
                  expand_abbreviations(query)).strip()


def optimise_for_wikipedia(query: str) -> str:
    """Extract key search terms from a natural language question."""
    q = query.lower().strip().rstrip('?')

    patterns = [
        (r'^who invented (.+)',           r'\1 inventor'),
        (r'^who created (.+)',            r'\1 inventor creator'),
        (r'^who discovered (.+)',         r'\1 discovery discoverer'),
        (r'^who wrote (.+)',              r'\1 author writer'),
        (r'^who painted (.+)',            r'\1 painter artist'),
        (r'^who built (.+)',              r'\1 builder architect'),
        (r'^who (was|is) (.+)',           r'\2'),
        (r'^what is the capital of (.+)', r'\1 capital'),
        (r'^what is the (.+)',            r'\1'),
        (r'^what (is|are|was) (.+)',      r'\2'),
        (r'^when did (.+)',               r'\1 year'),
        (r'^when was (.+)',               r'\1 year'),
        (r'^where is (.+)',               r'\1 location'),
        (r'^how tall (is|was) (.+)',      r'\2 height'),
        (r'^how fast (is|does) (.+)',     r'\2 speed'),
        (r'^how many (.+)',               r'\1'),
        (r'^what is largest (.+)',        r'largest \1'),
        (r'^largest (.+)',                r'largest \1'),
    ]

    for pattern, replacement in patterns:
        m = re.match(pattern, q)
        if m:
            q = re.sub(pattern, replacement, q)
            break

    # Remove leftover stopwords
    q = re.sub(r'\b(the|a|an|is|are|was|were)\b', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()
    return q


# ------------------------------------------------------------------ #
# Wikipedia API fetcher
# ------------------------------------------------------------------ #

def fetch_wikipedia(
    query:     str,
    sentences: int = 2,
) -> Optional[str]:
    """Fetch summary from Wikipedia REST API."""
    try:
        # Search for best article
        search_url = (
            "https://en.wikipedia.org/w/api.php?"
            "action=query&list=search&format=json"
            f"&srsearch={urllib.parse.quote(query)}&srlimit=1"
        )
        req = urllib.request.Request(
            search_url,
            headers={'User-Agent': 'ZINOHK/0.1 (educational)'}
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data    = json.loads(r.read())
            results = data.get('query', {}).get('search', [])

        if not results:
            return None

        title  = results[0]['title']
        safe_t = urllib.parse.quote(title.replace(' ', '_'))
        s_url  = (
            f"https://en.wikipedia.org/api/rest_v1/"
            f"page/summary/{safe_t}"
        )
        req2 = urllib.request.Request(
            s_url,
            headers={'User-Agent': 'ZINOHK/0.1 (educational)'}
        )
        with urllib.request.urlopen(req2, timeout=5) as r2:
            summary = json.loads(r2.read())

        extract = summary.get('extract', '')
        if not extract:
            return None

        sents = re.split(r'(?<=[.!?])\s+', extract)
        return ' '.join(sents[:sentences])

    except Exception:
        return None


def extract_answer(wiki_text: str, query: str) -> str:
    """
    Extract the most relevant sentence from Wikipedia text.

    Scores each sentence by keyword overlap with query.
    Returns the highest scoring sentence.
    """
    if not wiki_text:
        return wiki_text

    q_words = set(re.findall(r'\b\w+\b', query.lower()))
    q_words -= {'what','is','the','a','an','who','when',
                'where','how','why','did','was','were',
                'does','do','has','have'}

    sents  = re.split(r'(?<=[.!?])\s+', wiki_text)
    best   = wiki_text
    best_s = -1

    for sent in sents:
        s_words = set(re.findall(r'\b\w+\b', sent.lower()))
        score   = len(q_words & s_words)
        if score > best_s:
            best_s = score
            best   = sent

    return best


# ------------------------------------------------------------------ #
# Hybrid retriever
# ------------------------------------------------------------------ #

class HybridRetriever:
    """
    Hybrid local + internet retriever.

    Local first — web fallback — auto-cache.
    """

    def __init__(
        self,
        kb:                   KnowledgeBase,
        confidence_threshold: float = 0.4,
        cache_fetched:        bool  = True,
    ):
        self.kb                   = kb
        self.confidence_threshold = confidence_threshold
        self.cache_fetched        = cache_fetched
        self.n_local              = 0
        self.n_fetched            = 0
        self.n_failed             = 0
        self.fetch_log:           List[dict] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> Tuple[List[tuple], str]:
        """Local first, Wikipedia fallback, auto-cache."""

        # Expand abbreviations
        expanded = normalise_query(query)

        # Local search
        results = self.kb.retrieve(expanded, top_k=top_k)
        if not results:
            results = self.kb.retrieve(query, top_k=top_k)

        if results and results[0][0] >= self.confidence_threshold:
            self.n_local += 1
            return results, 'local'

        # Wikipedia fallback
        opt        = optimise_for_wikipedia(expanded)
        wiki_text  = fetch_wikipedia(opt, sentences=3)

        if wiki_text:
            self.n_fetched += 1
            best = extract_answer(wiki_text, query)

            if self.cache_fetched:
                self.kb.add_fact(
                    text     = wiki_text,
                    answer   = best,
                    category = 'wikipedia_cached',
                )
                self.fetch_log.append({
                    'query'  : query,
                    'fetched': best[:100],
                })

            results = self.kb.retrieve(expanded, top_k=top_k)
            return results, 'wikipedia'

        self.n_failed += 1
        return results, 'not_found'

    def stats(self) -> dict:
        return {
            "total"     : self.n_local+self.n_fetched+self.n_failed,
            "local"     : self.n_local,
            "web"       : self.n_fetched,
            "failed"    : self.n_failed,
            "cached"    : len(self.fetch_log),
            "kb_facts"  : len(self.kb.facts),
        }

    def __repr__(self) -> str:
        return (f"HybridRetriever("
                f"local={self.n_local}, "
                f"web={self.n_fetched})")
