"""
zinohk.reasoning.multihop
===========================
Multi-hop reasoning — chain facts to answer complex questions.

Single-hop: question → retrieve → answer
Multi-hop:  question → retrieve A → extract entity from A
                     → retrieve B → extract entity from B
                     → combine A + B → answer

Examples:
  "Who was Einstein's wife?"
    hop 1: retrieve "Albert Einstein" → text mentions "Mileva Marić"
    hop 2: retrieve "Mileva Marić"    → full answer

  "What is the capital of the country with the Amazon River?"
    hop 1: retrieve "Amazon River" → "flows through Brazil"
    hop 2: retrieve "Brazil capital" → "Brasília"

  "What did the inventor of the telephone study?"
    hop 1: retrieve "telephone inventor" → "Alexander Graham Bell"
    hop 2: retrieve "Alexander Graham Bell" → "studied acoustics"

No backprop. Pure retrieval chaining with entity extraction.
"""

import re
import time
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass, field


@dataclass
class HopResult:
    """Result of one retrieval hop."""
    hop_num   : int
    query     : str
    answer    : str
    confidence: float
    source    : str
    entities  : List[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class ReasoningResult:
    """Full multi-hop reasoning result."""
    question     : str
    final_answer : str
    confidence   : float
    hops         : List[HopResult]
    n_hops       : int
    total_ms     : float
    method       : str


class EntityExtractor:
    """
    Extracts entities from retrieved text.

    These entities become the next hop's query.
    No hardcoding — uses structural patterns.
    """

    # Relation patterns — maps question words to answer patterns
    RELATION_PATTERNS = {
        'wife'      : [r'\bmarried\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                       r'\bwife[,\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                       r'\bspouse[,\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'],
        'husband'   : [r'\bmarried\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                       r'\bhusband[,\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'],
        'capital'   : [r'\bcapital[,\s]+(?:is\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                       r'\bcapital\s+city[,\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'],
        'country'   : [r'\blocated\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                       r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),',
                       r'\bflows\s+through\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'],
        'inventor'  : [r'\binvented\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                       r'\bcreated\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'],
        'discovered': [r'\bdiscovered\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'],
        'born'      : [r'\bborn\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                       r'\bnative\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'],
        'founded'   : [r'\bfounded\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                       r'\bfounder[,\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'],
    }

    def extract_relation_entity(
        self,
        text:     str,
        relation: str,
    ) -> Optional[str]:
        """
        Extract entity for a specific relation from text.

        Parameters
        ----------
        text     : retrieved article text
        relation : what we're looking for (wife, capital, etc.)

        Returns
        -------
        str : extracted entity, or None
        """
        patterns = self.RELATION_PATTERNS.get(relation, [])
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return m.group(1).strip()
        return None

    def extract_all_entities(self, text: str) -> List[str]:
        """Extract all named entities from text."""
        # Multi-word proper nouns
        multi  = re.findall(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        # Single proper nouns (length > 4)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        starters  = {s.split()[0] for s in sentences if s.split()}
        single    = [w for w in re.findall(r'\b[A-Z][a-z]+\b', text)
                     if w not in starters and len(w) > 4]

        seen, result = set(), []
        for w in multi + single:
            if w not in seen:
                seen.add(w)
                result.append(w)
        return result[:8]

    def detect_relation(self, question: str) -> Optional[str]:
        """
        Detect what relation the question is asking about.

        Examples
        --------
        "Who was Einstein's wife?" → "wife"
        "What is the capital of France?" → "capital"
        "Who invented the telephone?" → "inventor"
        """
        q_lower = question.lower()
        relation_keywords = {
            'wife'      : ['wife', 'married', 'spouse'],
            'husband'   : ['husband', 'married', 'spouse'],
            'capital'   : ['capital', 'capital city'],
            'country'   : ['country', 'nation', 'located in'],
            'inventor'  : ['invent', 'inventor', 'created by', 'who made'],
            'discovered': ['discover', 'who found', 'who identified'],
            'born'      : ['born', 'birthplace', 'native'],
            'founded'   : ['founded', 'founder', 'established by'],
        }
        for relation, keywords in relation_keywords.items():
            if any(kw in q_lower for kw in keywords):
                return relation
        return None

    def needs_multihop(self, question: str) -> bool:
        """
        Detect if question needs multi-hop reasoning.

        Multi-hop signals:
        - Possessive: "Einstein's wife", "Amazon's source"
        - Relational: "capital of the country that..."
        - Chained: "the person who invented X studied at..."
        """
        q_lower = question.lower()

        # Possessive pattern
        if re.search(r"\b\w+'s\b", question):
            return True

        # Relation keywords
        multihop_phrases = [
            'capital of', 'capital city of',
            'country where', 'country that',
            'person who', 'inventor of',
            'founded by', 'discovered by',
            'studied at', 'born in',
            'wife of', 'husband of',
            'who invented', 'who founded',
            'who discovered', 'who created',
            'source of', 'origin of',
        ]
        return any(p in q_lower for p in multihop_phrases)


class MultiHopReasoner:
    """
    Multi-hop reasoning engine for ZINOHK.

    Chains retrieval hops to answer complex questions.
    Each hop retrieves a fact and extracts the next entity.

    Parameters
    ----------
    retrieve_fn : function(query, top_k) → List[dict]
                  Each dict must have: title, answer, confidence
    max_hops    : maximum number of retrieval hops
    min_confidence: minimum score to trust a retrieval

    Example
    -------
    >>> reasoner = MultiHopReasoner(retrieve_fn=wiki_search)
    >>> result = reasoner.reason("Who was Einstein's wife?")
    >>> print(result.final_answer)
    "Mileva Marić was a Serbian physicist and Einstein's first wife"
    """

    def __init__(
        self,
        retrieve_fn: Callable,
        max_hops:    int   = 3,
        min_confidence: float = 0.3,
    ):
        self.retrieve_fn    = retrieve_fn
        self.max_hops       = max_hops
        self.min_confidence = min_confidence
        self.extractor      = EntityExtractor()
        self.n_queries      = 0

    def _retrieve_one(self, query: str, hop_num: int) -> HopResult:
        """Execute one retrieval hop."""
        t0 = time.perf_counter()
        results = self.retrieve_fn(query, top_k=3)
        ms = (time.perf_counter() - t0) * 1000

        if not results:
            return HopResult(
                hop_num=hop_num, query=query,
                answer="", confidence=0.0, source="none",
                latency_ms=ms)

        best = results[0]
        text = best.get('answer', '')
        return HopResult(
            hop_num    = hop_num,
            query      = query,
            answer     = text,
            confidence = best.get('confidence', 0.0),
            source     = best.get('source', ''),
            entities   = self.extractor.extract_all_entities(text),
            latency_ms = round(ms, 2),
        )

    def reason(self, question: str) -> ReasoningResult:
        """
        Answer a question using multi-hop reasoning.

        Parameters
        ----------
        question : natural language question

        Returns
        -------
        ReasoningResult with full hop trace
        """
        self.n_queries += 1
        t_start = time.perf_counter()
        hops    = []

        # Detect if multi-hop needed
        if not self.extractor.needs_multihop(question):
            # Single hop — direct retrieval
            hop = self._retrieve_one(question, hop_num=1)
            hops.append(hop)
            return ReasoningResult(
                question     = question,
                final_answer = hop.answer or "No answer found.",
                confidence   = hop.confidence,
                hops         = hops,
                n_hops       = 1,
                total_ms     = round(
                    (time.perf_counter()-t_start)*1000, 2),
                method       = 'single_hop',
            )

        # Detect relation type
        relation = self.extractor.detect_relation(question)

        # Extract initial entity from question
        initial_entity = self._extract_question_entity(question)

        # Hop 1: retrieve about the initial entity
        hop1 = self._retrieve_one(initial_entity, hop_num=1)
        hops.append(hop1)

        if not hop1.answer or hop1.confidence < self.min_confidence:
            return ReasoningResult(
                question     = question,
                final_answer = "Could not find initial entity.",
                confidence   = 0.0,
                hops         = hops,
                n_hops       = 1,
                total_ms     = round(
                    (time.perf_counter()-t_start)*1000, 2),
                method       = 'failed_hop1',
            )

        # Extract next entity from hop 1 result
        next_entity = None
        if relation:
            next_entity = self.extractor.extract_relation_entity(
                hop1.answer, relation)

        # Fallback: use first named entity from hop 1
        if not next_entity and hop1.entities:
            next_entity = hop1.entities[0]

        if not next_entity:
            # Can't chain — return hop 1 result
            return ReasoningResult(
                question     = question,
                final_answer = hop1.answer,
                confidence   = hop1.confidence,
                hops         = hops,
                n_hops       = 1,
                total_ms     = round(
                    (time.perf_counter()-t_start)*1000, 2),
                method       = 'single_hop_fallback',
            )

        # Hop 2: retrieve about the extracted entity
        hop2 = self._retrieve_one(next_entity, hop_num=2)
        hops.append(hop2)

        # Hop 3: if needed and confidence still low
        if (hop2.confidence < self.min_confidence and
                len(hops) < self.max_hops):
            # Try combining question with extracted entity
            combined_query = f"{next_entity} {question.split()[-1]}"
            hop3 = self._retrieve_one(combined_query, hop_num=3)
            hops.append(hop3)
            final_answer = hop3.answer or hop2.answer
            final_conf   = max(hop3.confidence, hop2.confidence)
            method       = 'three_hop'
        else:
            final_answer = hop2.answer
            final_conf   = hop2.confidence
            method       = 'two_hop'

        # Build combined answer
        if hop2.answer and hop1.answer:
            combined = self._combine_answers(
                question, hop1, hop2, next_entity)
            final_answer = combined
            method      += '_combined'

        return ReasoningResult(
            question     = question,
            final_answer = final_answer or "Could not find answer.",
            confidence   = final_conf,
            hops         = hops,
            n_hops       = len(hops),
            total_ms     = round(
                (time.perf_counter()-t_start)*1000, 2),
            method       = method,
        )

    def _extract_question_entity(self, question: str) -> str:
        """Extract the main subject entity from a question."""
        q = question.lower().strip().rstrip('?')

        # Possessive: "Einstein's wife" → "Albert Einstein"
        poss = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s", question)
        if poss:
            return poss.group(1)

        # Standard patterns
        patterns = [
            (r"^who was (.+?)'s", 1),
            (r"^what is the (.+?) of (.+)", 2),
            (r"^what is (.+)", 1),
            (r"^who was (.+)", 1),
            (r"^who is (.+)", 1),
            (r"^where is (.+)", 1),
        ]
        for pat, grp in patterns:
            m = re.match(pat, q)
            if m:
                e = m.group(grp).strip()
                e = re.sub(r'\b(the|a|an|of)\b', '', e).strip()
                return ' '.join(w.capitalize() for w in e.split())
        return question

    def _combine_answers(
        self,
        question: str,
        hop1:     HopResult,
        hop2:     HopResult,
        bridge:   str,
    ) -> str:
        """Combine two hop answers into a coherent response."""
        # The bridge entity (e.g. "Mileva Marić") is the connection
        if bridge and hop2.answer:
            # Lead with the bridge entity + its description
            return f"{bridge}: {hop2.answer}"
        return hop2.answer or hop1.answer

    def format_trace(self, result: ReasoningResult) -> str:
        """Format reasoning trace for display."""
        lines = [
            f"🔗 Multi-hop Reasoning",
            f"{'─'*50}",
            f"Q: {result.question}",
            f"Method: {result.method} ({result.n_hops} hops)",
            f"{'─'*50}",
        ]
        for hop in result.hops:
            lines.append(
                f"Hop {hop.hop_num}: query={hop.query!r}")
            lines.append(
                f"  → {hop.answer[:80]}...")
            lines.append(
                f"  confidence={hop.confidence:.3f} | "
                f"{hop.latency_ms}ms")
            if hop.entities:
                lines.append(f"  entities={hop.entities[:3]}")
        lines += [
            f"{'─'*50}",
            f"A: {result.final_answer}",
            f"Confidence: {result.confidence:.3f}",
            f"Total: {result.total_ms}ms",
        ]
        return '\n'.join(lines)
