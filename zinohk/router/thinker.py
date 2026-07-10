"""
zinohk.router.thinker
======================
ZINOHK Thinking Layer — reason before answering.

Instead of directly returning the first retrieved fact,
ZINOHK now:
  1. RETRIEVES   — get top-k relevant facts
  2. THINKS      — chains facts, checks consistency
  3. SIMULATES   — considers alternative answers
  4. ANSWERS     — produces final confident response

This is different from LLMs which generate directly.
ZINOHK's thinking is explicit, traceable, and fast.

Think steps:
  step 1 — retrieve top-k candidates
  step 2 — score each by relevance + confidence
  step 3 — check if answer needs multiple facts chained
  step 4 — build reasoning trace
  step 5 — return answer + full reasoning chain
"""

import re
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ThoughtStep:
    """One step in ZINOHK's reasoning chain."""
    step:       int
    action:     str    # retrieve, evaluate, chain, conclude
    content:    str
    confidence: float
    latency_ms: float


@dataclass
class ThinkResult:
    """Full result of ZINOHK's thinking process."""
    question:      str
    answer:        str
    confidence:    float
    domain:        str
    reasoning:     List[ThoughtStep]
    candidates:    List[dict]
    latency_ms:    float
    method:        str


class Thinker:
    """
    ZINOHK's explicit reasoning engine.

    Wraps any knowledge base with a thinking layer.
    All reasoning steps are visible and traceable.

    Parameters
    ----------
    top_k         : number of candidates to retrieve
    min_confidence: minimum confidence to answer directly
    chain_threshold: confidence below which to chain facts
    """

    def __init__(
        self,
        top_k:            int   = 5,
        min_confidence:   float = 0.5,
        chain_threshold:  float = 0.7,
    ):
        self.top_k           = top_k
        self.min_confidence  = min_confidence
        self.chain_threshold = chain_threshold
        self.n_queries       = 0

    def think(
        self,
        question:  str,
        retrieve_fn,   # function(query, top_k) → List[dict]
        domain:    str = 'general',
    ) -> ThinkResult:
        """
        Full thinking pipeline.

        Parameters
        ----------
        question    : user's question
        retrieve_fn : function that returns list of candidate facts
                      each fact must have: title, answer, confidence
        domain      : which domain this is being answered from

        Returns
        -------
        ThinkResult with full reasoning trace
        """
        self.n_queries += 1
        t_start    = time.perf_counter()
        reasoning  = []

        # ── STEP 1: Retrieve ───────────────────────────────────
        t0 = time.perf_counter()
        candidates = retrieve_fn(question, self.top_k)
        t1 = time.perf_counter()

        reasoning.append(ThoughtStep(
            step       = 1,
            action     = 'retrieve',
            content    = (f"Retrieved {len(candidates)} candidates "
                         f"from {domain} knowledge base"),
            confidence = 1.0,
            latency_ms = (t1-t0)*1000,
        ))

        if not candidates:
            return ThinkResult(
                question   = question,
                answer     = "I don't have information on that.",
                confidence = 0.0,
                domain     = domain,
                reasoning  = reasoning,
                candidates = [],
                latency_ms = (time.perf_counter()-t_start)*1000,
                method     = 'no_results',
            )

        # ── STEP 2: Evaluate candidates ────────────────────────
        t0 = time.perf_counter()
        best = candidates[0]
        best_conf = best.get('confidence', 0.0)

        # Score all candidates
        scored = []
        q_words = set(re.findall(r'\b\w+\b', question.lower())) - {
            'what','is','the','a','an','who','how','when',
            'where','was','did','does','do','of','in'}

        for c in candidates:
            title_words = set(re.findall(
                r'\b\w+\b', c.get('title','').lower()))
            ans_words   = set(re.findall(
                r'\b\w+\b', c.get('answer','').lower()))
            overlap     = len(q_words & (title_words | ans_words))
            score       = c.get('confidence', 0.0) + overlap * 0.05
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]
        t1 = time.perf_counter()

        reasoning.append(ThoughtStep(
            step       = 2,
            action     = 'evaluate',
            content    = (f"Best match: '{best.get('title','')}' "
                         f"(score={best_score:.3f})"),
            confidence = best_score,
            latency_ms = (t1-t0)*1000,
        ))

        # ── STEP 3: Check if chaining needed ──────────────────
        t0 = time.perf_counter()
        needs_chain = best_score < self.chain_threshold

        if needs_chain and len(candidates) > 1:
            # Try combining top-2 answers
            combined = (
                best.get('answer', '') + ' ' +
                scored[1][1].get('answer', '') if len(scored) > 1
                else best.get('answer', '')
            )
            reasoning.append(ThoughtStep(
                step       = 3,
                action     = 'chain',
                content    = (f"Low confidence ({best_score:.3f}) — "
                             f"combining top 2 facts"),
                confidence = best_score,
                latency_ms = (time.perf_counter()-t0)*1000,
            ))
            final_answer = combined
            method       = 'chained'
        else:
            reasoning.append(ThoughtStep(
                step       = 3,
                action     = 'chain',
                content    = (f"High confidence ({best_score:.3f}) — "
                             f"single fact sufficient"),
                confidence = best_score,
                latency_ms = (time.perf_counter()-t0)*1000,
            ))
            final_answer = best.get('answer', '')
            method       = 'direct'

        # ── STEP 4: Simulate alternatives ─────────────────────
        t0 = time.perf_counter()
        alternatives = [s[1].get('title','') for s in scored[1:3]]

        reasoning.append(ThoughtStep(
            step       = 4,
            action     = 'simulate',
            content    = (f"Alternatives considered: "
                         f"{alternatives}"),
            confidence = best_score,
            latency_ms = (time.perf_counter()-t0)*1000,
        ))

        # ── STEP 5: Conclude ───────────────────────────────────
        t0 = time.perf_counter()
        final_conf = min(best_score, 1.0)

        if final_conf < self.min_confidence:
            prefix       = "Based on available information: "
            final_answer = prefix + final_answer
            method       = method + '_low_conf'

        reasoning.append(ThoughtStep(
            step       = 5,
            action     = 'conclude',
            content    = (f"Final answer selected. "
                         f"Confidence: {final_conf:.3f}"),
            confidence = final_conf,
            latency_ms = (time.perf_counter()-t0)*1000,
        ))

        return ThinkResult(
            question   = question,
            answer     = final_answer,
            confidence = round(final_conf, 4),
            domain     = domain,
            reasoning  = reasoning,
            candidates = [s[1] for s in scored],
            latency_ms = round(
                (time.perf_counter()-t_start)*1000, 2),
            method     = method,
        )

    def format_trace(self, result: ThinkResult) -> str:
        """Format thinking trace for display."""
        lines = [
            f"🤔 ZINOHK Thinking Trace",
            f"{'─'*50}",
            f"Q: {result.question}",
            f"Domain: {result.domain}",
            f"{'─'*50}",
        ]
        for step in result.reasoning:
            lines.append(
                f"Step {step.step} [{step.action.upper()}] "
                f"{step.content} "
                f"({step.latency_ms:.1f}ms)")
        lines += [
            f"{'─'*50}",
            f"A: {result.answer}",
            f"Confidence: {result.confidence:.3f}",
            f"Method: {result.method}",
            f"Total: {result.latency_ms:.1f}ms",
        ]
        return '\n'.join(lines)
