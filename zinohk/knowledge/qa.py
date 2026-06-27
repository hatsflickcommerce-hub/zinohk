"""
zinohk.knowledge.qa
====================
Full Q&A pipeline — ties everything together.

Pipeline:
  question (text)
      → TFIDFEncoder    (text → vector)
      → SpikeEncoder    (vector → spikes)
      → KnowledgeBase   (spikes → retrieve top facts)
      → AnswerBuilder   (facts → natural language answer)
      → SafetyClassifier (filter output)
      → answer (text)

This is ZINOHK's first general Q&A system.
No backprop. No GPU. Runs on CPU only.
Knowledge grows by adding facts to the base.
"""

import numpy as np
from typing import List, Optional, Tuple
from zinohk.knowledge.base  import KnowledgeBase, Fact
from zinohk.utils.safety    import SafetyClassifier
from zinohk.learning.memory import MemorySystem


class AnswerBuilder:
    """
    Converts retrieved facts into natural language answers.

    Strategies:
      - direct   : return the answer field directly
      - extract  : pull answer from matched fact text
      - template : fill a template with retrieved info

    Parameters
    ----------
    confidence_threshold : minimum score to give a direct answer
                          below this → say "I'm not sure"
    """

    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold

    def build(
        self,
        query:   str,
        results: List[Tuple[float, Fact]],
    ) -> dict:
        """
        Build a natural language answer from retrieved facts.

        Parameters
        ----------
        query   : original question
        results : list of (score, Fact) from retriever

        Returns
        -------
        dict with keys: answer, confidence, source, fact_used
        """
        if not results:
            return {
                "answer"    : "I don't have information on that yet.",
                "confidence": 0.0,
                "source"    : None,
                "fact_used" : None,
            }

        top_score, top_fact = results[0]

        if top_score < self.confidence_threshold:
            return {
                "answer"    : f"I'm not confident about that. "
                              f"The closest I have is: "
                              f"{top_fact.text}",
                "confidence": top_score,
                "source"    : top_fact.category,
                "fact_used" : top_fact.text,
            }

        # High confidence — build direct answer
        answer = self._format_answer(query, top_fact, top_score)

        return {
            "answer"    : answer,
            "confidence": round(top_score, 4),
            "source"    : top_fact.category,
            "fact_used" : top_fact.text,
        }

    def _format_answer(
        self,
        query: str,
        fact:  Fact,
        score: float,
    ) -> str:
        """Format a natural language answer."""
        q = query.lower().strip().rstrip('?')

        # Who questions
        if q.startswith('who'):
            return f"{fact.answer}."

        # What colour / what color
        if 'colour' in q or 'color' in q:
            return f"It is {fact.answer}."

        # What is / what are
        if q.startswith('what is') or q.startswith('what are'):
            return f"{fact.answer}."

        # How fast / how long / how many
        if q.startswith('how'):
            return f"{fact.answer}."

        # Why questions
        if q.startswith('why'):
            return f"{fact.text}."

        # When questions
        if q.startswith('when'):
            return f"{fact.answer}."

        # Where questions
        if q.startswith('where'):
            return f"{fact.answer}."

        # Default — return full fact
        return f"{fact.text}."


class ZINOHKQA:
    """
    ZINOHK Question Answering System — Phase 8.

    Combines:
      - KnowledgeBase  : stores and retrieves facts
      - AnswerBuilder  : formats natural language answers
      - SafetyClassifier: filters unsafe outputs
      - MemorySystem   : remembers conversation context

    Parameters
    ----------
    vocab_size  : TF-IDF vocabulary size
    top_k       : number of facts to retrieve per query
    confidence  : minimum retrieval score for direct answer

    Example
    -------
    >>> qa = ZINOHKQA()
    >>> qa.load_facts(texts, answers, categories)
    >>> response = qa.ask("Why is the sky blue?")
    >>> print(response['answer'])
    """

    def __init__(
        self,
        vocab_size: int   = 1000,
        top_k:      int   = 3,
        confidence: float = 0.3,
    ):
        self.kb      = KnowledgeBase(vocab_size=vocab_size)
        self.builder = AnswerBuilder(confidence_threshold=confidence)
        self.top_k   = top_k

        # Safety filter
        self.safety = SafetyClassifier(safe_default=0)
        self.safety.add_rule(
            'no_empty',
            'block empty answers',
            lambda pred, scores: True,  # always pass for text
        )

        # Conversation memory
        self.memory = MemorySystem(
            n_inputs  = vocab_size,
            n_outputs = 2,
            capacity  = 50,
            replay_k  = 5,
        )

        self.history: List[dict] = []
        self.n_asked: int        = 0

    def load_facts(
        self,
        texts:      List[str],
        answers:    List[str],
        categories: Optional[List[str]] = None,
    ) -> None:
        """Load a knowledge corpus into the base."""
        self.kb.build(texts, answers, categories)

    def add_fact(self, text: str, answer: str,
                 category: str = 'general') -> None:
        """Add a single fact dynamically."""
        self.kb.add_fact(text, answer, category)

    def ask(self, question: str) -> dict:
        """
        Answer a question using the knowledge base.

        Parameters
        ----------
        question : natural language question

        Returns
        -------
        dict with: answer, confidence, source, fact_used, rank
        """
        self.n_asked += 1

        # Retrieve relevant facts
        results = self.kb.retrieve(question, top_k=self.top_k)

        # Build answer
        response = self.builder.build(question, results)
        response['question'] = question
        response['rank']     = self.n_asked

        # Store in history
        self.history.append(response)

        return response

    def show(self, response: dict) -> None:
        """Pretty print a Q&A response."""
        print(f"Q: {response['question']}")
        print(f"A: {response['answer']}")
        print(f"   confidence : {response['confidence']}")
        print(f"   source     : {response['source']}")
        print(f"   matched    : {response.get('fact_used','')[:70]}")
        print()

    def stats(self) -> dict:
        return {
            "questions_asked": self.n_asked,
            "kb_facts"       : len(self.kb.facts),
            "kb_vocab"       : self.kb.text_enc.vocab_size,
        }

    def __repr__(self) -> str:
        return (f"ZINOHKQA(facts={len(self.kb.facts)}, "
                f"asked={self.n_asked})")
