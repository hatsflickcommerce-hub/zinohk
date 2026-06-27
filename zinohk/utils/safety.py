"""
zinohk.utils.safety
====================
Safety & Alignment layer — GAP-07 + GAP-08.

Problem:
  A continuously learning system that rewires itself
  from user interactions can learn and reinforce harmful
  patterns. Unlike frozen LLMs, ZINOHK learns after deploy.

Solution — three mechanisms:

  1. ValueGate (GAP-07):
       A set of protected weights that CANNOT be updated
       by Hebbian learning. These encode hard constraints
       (e.g. never output class X for input pattern Y).
       Like a constitutional layer — always enforced.

  2. SafetyClassifier (GAP-07):
       A simple rule-based filter on outputs.
       Blocks outputs that violate defined constraints.
       Runs AFTER the network, before output is returned.

  3. InstructionShaper (GAP-08):
       Provides a supervised signal that shapes learning
       toward instruction-following behaviour.
       Uses a reward signal: +1 if instruction followed,
       -1 if violated. Modulates Hebbian update strength.

Philosophy:
  Safety in ZINOHK is architectural — not prompt-based.
  Constraints are baked into the weight structure itself,
  not just into the input/output text.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional


# ------------------------------------------------------------------ #
# 1. VALUE GATE — protected weights
# ------------------------------------------------------------------ #

class ValueGate:
    """
    Protected weight layer that cannot be overwritten by learning.

    ValueGate sits between the network output and the final
    prediction. It applies hard constraints regardless of
    what the network learned.

    Parameters
    ----------
    n_inputs  : output dimension of the main network
    n_outputs : final output dimension
    seed      : random seed

    Usage
    -----
    gate = ValueGate(n_inputs=2, n_outputs=2)
    gate.protect(input_class=1, output_class=0)
    # Now class 1 input will never produce class 0 output
    """

    def __init__(
        self,
        n_inputs:  int,
        n_outputs: int,
        seed:      int = 42,
    ):
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs

        # Protected weights — fixed, never updated
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(0.1, 0.5, (n_outputs, n_inputs))

        # Hard blocks: (input_class, output_class) pairs to block
        self.blocks: List[tuple] = []

    def protect(self, input_class: int, output_class: int) -> None:
        """
        Block a specific (input, output) mapping permanently.
        This constraint survives all future learning.
        """
        self.blocks.append((input_class, output_class))

    def apply(self, network_output: np.ndarray) -> np.ndarray:
        """
        Apply value gate to network output.

        Parameters
        ----------
        network_output : raw scores from main network

        Returns
        -------
        np.ndarray : gated scores with hard constraints applied
        """
        scores = self.weights @ network_output

        # Apply hard blocks
        pred = int(np.argmax(network_output))
        for (in_cls, out_cls) in self.blocks:
            if pred == in_cls:
                scores[out_cls] = -np.inf

        return scores

    def __repr__(self) -> str:
        return (f"ValueGate(inputs={self.n_inputs}, "
                f"outputs={self.n_outputs}, "
                f"blocks={len(self.blocks)})")


# ------------------------------------------------------------------ #
# 2. SAFETY CLASSIFIER — output filter
# ------------------------------------------------------------------ #

@dataclass
class SafetyRule:
    """A single safety constraint."""
    name:        str
    description: str
    check:       Callable[[int, np.ndarray], bool]
    # Returns True if SAFE, False if BLOCKED


class SafetyClassifier:
    """
    Rule-based output safety filter.

    Runs after the network produces a prediction.
    If any rule is violated, output is blocked and
    a safe default is returned instead.

    Parameters
    ----------
    safe_default : output to return when blocked (default 0)

    Example
    -------
    sc = SafetyClassifier(safe_default=0)
    sc.add_rule('no_extreme', 'block extreme outputs',
                lambda pred, scores: scores.max() < 10.0)
    safe_pred = sc.check(pred=1, scores=np.array([0.1, 9.5]))
    """

    def __init__(self, safe_default: int = 0):
        self.safe_default = safe_default
        self.rules:        List[SafetyRule] = []
        self.n_blocked:    int = 0
        self.n_passed:     int = 0

    def add_rule(
        self,
        name:        str,
        description: str,
        check:       Callable[[int, np.ndarray], bool],
    ) -> None:
        """Add a safety rule."""
        self.rules.append(SafetyRule(
            name        = name,
            description = description,
            check       = check,
        ))

    def filter(
        self,
        pred:   int,
        scores: np.ndarray,
    ) -> tuple:
        """
        Filter a prediction through all safety rules.

        Returns
        -------
        (safe_pred, is_safe, violated_rule)
        """
        for rule in self.rules:
            if not rule.check(pred, scores):
                self.n_blocked += 1
                return self.safe_default, False, rule.name

        self.n_passed += 1
        return pred, True, None

    def stats(self) -> dict:
        total = self.n_passed + self.n_blocked
        return {
            "total"    : total,
            "passed"   : self.n_passed,
            "blocked"  : self.n_blocked,
            "block_rate": round(
                self.n_blocked / total if total > 0 else 0.0, 4),
        }

    def __repr__(self) -> str:
        return (f"SafetyClassifier(rules={len(self.rules)}, "
                f"blocked={self.n_blocked})")


# ------------------------------------------------------------------ #
# 3. INSTRUCTION SHAPER — GAP-08
# ------------------------------------------------------------------ #

class InstructionShaper:
    """
    Shapes Hebbian learning toward instruction-following.

    When the network follows the instruction correctly:
      → reward = +1 → Hebbian update is amplified
    When the network violates the instruction:
      → reward = -1 → Hebbian update is reversed

    This modulates the learning signal without backprop.

    Parameters
    ----------
    lr_scale_positive : multiply lr by this on correct follow
    lr_scale_negative : multiply lr by this on violation

    Example
    -------
    shaper = InstructionShaper()
    lr_mod = shaper.modulate(pred=1, instruction=1)
    # lr_mod > 1.0 if correct, < 0 if wrong
    """

    def __init__(
        self,
        lr_scale_positive: float = 2.0,
        lr_scale_negative: float = -1.0,
    ):
        self.lr_scale_positive = lr_scale_positive
        self.lr_scale_negative = lr_scale_negative
        self.n_correct         = 0
        self.n_violated        = 0

    def modulate(
        self,
        pred:        int,
        instruction: int,
    ) -> float:
        """
        Compute learning rate modulator based on instruction.

        Parameters
        ----------
        pred        : network prediction
        instruction : desired output (from instruction)

        Returns
        -------
        float : learning rate multiplier
                > 1.0 → amplify learning (correct)
                < 0.0 → reverse learning (violation)
                  1.0 → no modulation
        """
        if pred == instruction:
            self.n_correct += 1
            return self.lr_scale_positive
        else:
            self.n_violated += 1
            return self.lr_scale_negative

    @property
    def follow_rate(self) -> float:
        total = self.n_correct + self.n_violated
        return self.n_correct / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "correct"    : self.n_correct,
            "violated"   : self.n_violated,
            "follow_rate": round(self.follow_rate, 4),
        }

    def __repr__(self) -> str:
        return (f"InstructionShaper("
                f"follow_rate={self.follow_rate:.2f})")
