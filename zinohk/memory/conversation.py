"""
zinohk.memory.conversation
===========================
Conversation memory for ZINOHK API.
"""

import time
import re
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    turn_id   : int
    question  : str
    answer    : str
    domain    : str
    confidence: float
    timestamp : float = field(default_factory=time.time)
    entities  : List[str] = field(default_factory=list)


class ConversationBuffer:
    """Sliding window of recent conversation turns."""

    def __init__(self, max_turns: int = 10):
        self.max_turns      = max_turns
        self.turns: List[ConversationTurn] = []
        self.turn_counter   = 0
        self._current_topic = ""

    # ── Entity extraction ─────────────────────────────────────

    def _extract_entities(self, text: str) -> List[str]:
        """
        Dynamic entity extraction from text.
        Multi-word capitalised sequences first (most reliable),
        then single capitalised words longer than 4 chars.
        No hardcoded stop word lists.
        """
        # Multi-word proper nouns: Albert Einstein, Amazon River
        multi = re.findall(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)

        # Single capitalised words from non-sentence-start positions
        # Split into sentences, skip first word of each
        sentences     = re.split(r'(?<=[.!?])\s+', text)
        sent_starters = set()
        for s in sentences:
            w = s.split()
            if w:
                sent_starters.add(w[0])

        single = [
            w for w in re.findall(r'\b[A-Z][a-z]+\b', text)
            if w not in sent_starters and len(w) > 4
        ]

        # Combine, deduplicate, multi-word first
        seen, result = set(), []
        for w in multi + single:
            if w not in seen:
                seen.add(w)
                result.append(w)
        return result[:5]

    def _best_entity(self, entities: List[str]) -> str:
        """Pick best entity: longest multi-word first."""
        multi  = [e for e in entities if len(e.split()) > 1]
        single = [e for e in entities if len(e.split()) == 1]
        pool   = multi + single
        return max(pool, key=len) if pool else ""

    # ── Add turn ──────────────────────────────────────────────

    def add(
        self,
        question:   str,
        answer:     str,
        domain:     str   = 'general',
        confidence: float = 0.0,
    ) -> ConversationTurn:
        """Add a Q+A turn. Updates current topic from answer."""
        # Extract entities from answer (more reliable than question)
        ans_entities = self._extract_entities(answer)
        all_entities = self._extract_entities(
            question + ' ' + answer)

        turn = ConversationTurn(
            turn_id    = self.turn_counter,
            question   = question,
            answer     = answer,
            domain     = domain,
            confidence = confidence,
            entities   = all_entities,
        )
        self.turns.append(turn)
        self.turn_counter += 1
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

        # Update topic from answer entities
        best = self._best_entity(ans_entities)
        if best:
            self._current_topic = best

        return turn

    # ── Context & topic ───────────────────────────────────────

    def current_topic(self) -> str:
        """
        Topic at question time = last answer's subject.
        This is what pronouns like 'it', 'he', 'they' refer to.
        """
        if not self.turns:
            return ""
        # Read entities from last completed answer
        last_ans     = self.turns[-1].answer
        ans_entities = self._extract_entities(last_ans)
        best         = self._best_entity(ans_entities)
        return best if best else self._current_topic

    def get_context(self, n_turns: int = 3) -> str:
        """Build context string from recent turns."""
        recent = self.turns[-n_turns:]
        lines  = []
        for t in recent:
            lines.append(f"Q: {t.question}")
            lines.append(f"A: {t.answer[:100]}")
        return '\n'.join(lines)

    def get_recent_entities(self, n_turns: int = 3) -> List[str]:
        """Get entities from recent turns."""
        entities = []
        for t in self.turns[-n_turns:]:
            entities.extend(t.entities)
        return list(dict.fromkeys(entities))

    # ── Pronoun resolution ────────────────────────────────────

    def resolve_pronouns(self, question: str) -> str:
        """
        Replace pronouns with current topic.
        Topic = most recent answer's subject entity.
        """
        if not self.turns:
            return question

        topic   = self.current_topic()
        q_lower = question.lower()

        if not topic:
            return question

        patterns = [
            r'\b(he|him|his)\b',
            r'\b(she|her|hers)\b',
            r'\b(it|its)\b',
            r'\b(they|them|their)\b',
            r'\bthis\b',
            r'\bthat\b',
            r'\bthere\b',
        ]

        resolved = question
        for p in patterns:
            if re.search(p, q_lower):
                resolved = re.sub(p, topic, resolved,
                                  flags=re.IGNORECASE)
        return resolved

    def is_followup(self, question: str) -> bool:
        """Detect follow-up questions."""
        if not self.turns:
            return False
        q_lower  = question.lower().strip()
        words    = q_lower.split()
        pronouns = {'he','him','his','she','her','it','its',
                    'they','them','their','this','that','there'}
        if any(w in pronouns for w in words):
            return True
        if len(words) <= 3:
            return True
        starts = ['and ','also ','what about','tell me more',
                  'why ','how about','what else']
        return any(q_lower.startswith(s) for s in starts)

    def enrich_question(self, question: str) -> str:
        """Resolve pronouns in question using conversation context."""
        return self.resolve_pronouns(question)

    def summary(self) -> dict:
        return {
            'turns'        : len(self.turns),
            'max_turns'    : self.max_turns,
            'turn_counter' : self.turn_counter,
            'current_topic': self.current_topic(),
            'entities'     : self.get_recent_entities(),
        }

    def __repr__(self) -> str:
        return f"ConversationBuffer(turns={len(self.turns)})"


class SessionStore:
    """Manages conversation buffers for multiple sessions."""

    def __init__(
        self,
        max_turns:          int = 10,
        inactivity_timeout: int = 3600,
    ):
        self.max_turns          = max_turns
        self.inactivity_timeout = inactivity_timeout
        self._sessions: Dict[str, dict] = {}

    def get_or_create(self, session_id: str) -> ConversationBuffer:
        now     = time.time()
        expired = [
            sid for sid, data in self._sessions.items()
            if now - data['last_active'] > self.inactivity_timeout
        ]
        for sid in expired:
            del self._sessions[sid]

        if session_id not in self._sessions:
            self._sessions[session_id] = {
                'buffer'     : ConversationBuffer(self.max_turns),
                'created_at' : now,
                'last_active': now,
            }
        self._sessions[session_id]['last_active'] = now
        return self._sessions[session_id]['buffer']

    def clear(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def stats(self) -> dict:
        return {
            'active_sessions': len(self._sessions),
            'total_turns'    : sum(
                d['buffer'].turn_counter
                for d in self._sessions.values()),
        }

    def __repr__(self) -> str:
        return f"SessionStore(sessions={len(self._sessions)})"
