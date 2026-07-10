"""
zinohk.router.domain
=====================
Domain router — decides which knowledge base to query.

When a question comes in ZINOHK:
  1. Classifies the domain (finance, general, science, etc.)
  2. Routes to the correct knowledge base
  3. Retrieves from the right FAISS index
  4. Combines answers if multi-domain

Domains:
  general   → Wikipedia (6.4M facts)
  finance   → options, swing trading, market concepts
  science   → physics, biology, chemistry
  tech      → programming, AI, software
  medicine  → health, drugs, anatomy
  custom    → user-defined domains
"""

import re
from typing import Dict, List, Optional, Tuple


# ── Domain keyword signatures ──────────────────────────────────

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    'finance': [
        'stock', 'option', 'call', 'put', 'trade', 'trading',
        'market', 'price', 'invest', 'portfolio', 'dividend',
        'swing', 'bull', 'bear', 'rsi', 'macd', 'delta', 'theta',
        'gamma', 'vega', 'volatility', 'hedge', 'short', 'long',
        'equity', 'bond', 'etf', 'index', 'pe ratio', 'earnings',
        'revenue', 'profit', 'loss', 'candlestick', 'chart',
        'support', 'resistance', 'breakout', 'momentum', 'spread',
        'strike', 'expiry', 'premium', 'covered call', 'iron condor',
        'sharpe', 'alpha', 'beta', 'drawdown', 'backtest',
    ],
    'medicine': [
        'disease', 'drug', 'medicine', 'symptom', 'treatment',
        'cancer', 'virus', 'bacteria', 'health', 'medical',
        'diagnosis', 'therapy', 'surgery', 'hospital', 'doctor',
        'patient', 'clinical', 'dose', 'vaccine', 'immune',
    ],
    'science': [
        'physics', 'chemistry', 'biology', 'quantum', 'relativity',
        'atom', 'molecule', 'energy', 'force', 'gravity', 'light',
        'electron', 'proton', 'neutron', 'dna', 'gene', 'cell',
        'evolution', 'photosynthesis', 'thermodynamics',
    ],
    'tech': [
        'python', 'javascript', 'code', 'program', 'algorithm',
        'database', 'api', 'machine learning', 'neural network',
        'software', 'hardware', 'cpu', 'gpu', 'cloud', 'docker',
        'kubernetes', 'linux', 'git', 'ai', 'llm', 'transformer',
    ],
    'general': []  # fallback
}


class DomainRouter:
    """
    Routes questions to the correct knowledge base.

    Supports keyword-based routing and semantic routing.
    Multiple domains can match — returns ranked list.

    Parameters
    ----------
    registered_domains : dict of domain_name → any (KB handle)
    """

    def __init__(self):
        self.registered: Dict[str, object] = {}
        self.keyword_map = DOMAIN_KEYWORDS.copy()

    def register(self, domain: str, kb, keywords: Optional[List[str]] = None):
        """
        Register a knowledge base for a domain.

        Parameters
        ----------
        domain   : domain name ('finance', 'general', etc.)
        kb       : knowledge base object (must have .ask() method)
        keywords : additional keywords for this domain
        """
        self.registered[domain] = kb
        if keywords:
            existing = self.keyword_map.get(domain, [])
            self.keyword_map[domain] = existing + keywords
        print(f"✅ Registered domain: '{domain}'")

    def classify(self, question: str) -> List[Tuple[str, float]]:
        """
        Classify question into domains with confidence scores.

        Returns
        -------
        List of (domain, score) sorted by confidence descending.
        """
        q_lower  = question.lower()
        q_words  = set(re.findall(r'\b\w+\b', q_lower))
        scores   = {}

        for domain, keywords in self.keyword_map.items():
            if domain == 'general':
                continue
            kw_set = set(keywords)
            # Count keyword matches
            matches = len(q_words & kw_set)
            # Also check multi-word phrases
            for kw in keywords:
                if ' ' in kw and kw in q_lower:
                    matches += 2   # phrase match is stronger
            if matches > 0:
                scores[domain] = matches

        if not scores:
            return [('general', 1.0)]

        # Normalise scores
        total = sum(scores.values())
        ranked = sorted(
            [(d, s/total) for d, s in scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    def route(self, question: str) -> Tuple[str, object]:
        """
        Route question to the best registered knowledge base.

        Returns
        -------
        (domain_name, knowledge_base)
        """
        ranked = self.classify(question)

        # Find first registered domain in ranked list
        for domain, score in ranked:
            if domain in self.registered:
                return domain, self.registered[domain]

        # Fallback to general
        if 'general' in self.registered:
            return 'general', self.registered['general']

        # Return domain name only if nothing registered
        return ('general', None)

    def classify_verbose(self, question: str) -> dict:
        """Return full classification report."""
        ranked = self.classify(question)
        domain = ranked[0][0] if ranked else 'general'
        if self.registered:
            domain, _ = self.route(question)
        return {
            'question'      : question,
            'routed_to'     : domain,
            'scores'        : dict(ranked),
            'registered_kbs': list(self.registered.keys()),
        }

    def __repr__(self) -> str:
        return (f"DomainRouter("
                f"domains={list(self.registered.keys())})")
