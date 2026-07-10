"""
zinohk.world.model
===================
ZINOHK World Model — simulates real-world scenarios.

The world model lets ZINOHK:
  1. PREDICT   — what happens next given current state
  2. SIMULATE  — run multiple future scenarios
  3. EVALUATE  — score each scenario by likelihood
  4. PLAN      — choose best action toward a goal

Architecture:
  State encoder  : current world state → spike pattern
  Transition net : spike(t) → spike(t+1) prediction
  Reward model   : state → how good is this state
  Planner        : simulate N steps, pick best path

This is the foundation for:
  - Voice: predict next audio frame from text spikes
  - Image: predict pixel patches from concept spikes
  - Action: predict consequences of actions

No backprop — all Hebbian updates.
State transitions learned from co-occurrence.
"""

import numpy as np
import time
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class WorldState:
    """
    A snapshot of the world at one moment in time.

    Attributes
    ----------
    state_id   : unique identifier
    spikes     : sparse spike pattern (N,) binary
    context    : text description of this state
    reward     : how desirable this state is (0-1)
    timestamp  : when this state was created
    parent_id  : which state led to this one
    """
    state_id  : int
    spikes    : np.ndarray
    context   : str
    reward    : float        = 0.0
    timestamp : float        = field(default_factory=time.time)
    parent_id : Optional[int] = None


class TransitionMemory:
    """
    Learns state transitions via Hebbian co-occurrence.

    When state A is followed by state B:
      W[A_active_nodes, B_active_nodes] += lr

    Over time, active nodes in A predict active nodes in B.
    This is how the world model learns causality.
    """

    def __init__(self, n_nodes: int = 512, lr: float = 0.01):
        self.n_nodes = n_nodes
        self.lr      = lr
        # Transition weight matrix
        self.W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        self.n_transitions = 0

    def learn_transition(
        self,
        state_a: np.ndarray,
        state_b: np.ndarray,
    ) -> None:
        """
        Learn that state_a leads to state_b.

        Hebbian rule: nodes active in both get stronger.
        """
        # Outer product of active nodes
        active_a = (state_a > 0).astype(np.float32)
        active_b = (state_b > 0).astype(np.float32)
        self.W  += self.lr * np.outer(active_a, active_b)
        self.W   = np.clip(self.W, 0.0, 1.0)
        self.n_transitions += 1

    def predict_next(self, state: np.ndarray) -> np.ndarray:
        """
        Predict the next state given current state.

        Returns predicted spike pattern for next state.
        """
        active   = (state > 0).astype(np.float32)
        raw      = active @ self.W              # (n_nodes,)
        # Sparse threshold — only top 5% fire
        threshold = np.percentile(raw, 95)
        predicted = (raw > threshold).astype(np.float32)
        return predicted


class RewardModel:
    """
    Scores how desirable a state is.

    Learns which spike patterns are associated with
    positive outcomes via Hebbian reinforcement.

    reward > 0.5 → good state
    reward < 0.5 → bad state
    """

    def __init__(self, n_nodes: int = 512):
        self.n_nodes = n_nodes
        self.W       = np.random.uniform(
            0.4, 0.6, n_nodes).astype(np.float32)
        self.n_updates = 0

    def score(self, state: np.ndarray) -> float:
        """Score a state between 0 and 1."""
        active = (state > 0).astype(np.float32)
        if active.sum() == 0:
            return 0.5
        score = float(active @ self.W) / (active.sum() + 1e-8)
        return float(np.clip(score, 0.0, 1.0))

    def update(
        self,
        state:  np.ndarray,
        reward: float,
    ) -> None:
        """Update reward weights from feedback."""
        active = (state > 0).astype(np.float32)
        error  = reward - self.score(state)
        self.W += 0.01 * error * active
        self.W  = np.clip(self.W, 0.0, 1.0)
        self.n_updates += 1


class WorldModel:
    """
    ZINOHK World Model — predict, simulate, plan.

    Parameters
    ----------
    n_nodes    : number of sparse nodes in state space
    n_sim_steps: how many steps to simulate ahead
    n_scenarios: how many parallel scenarios to run

    Example
    -------
    >>> wm = WorldModel(n_nodes=512)
    >>> state = wm.encode_context("stock market is rising")
    >>> next_state = wm.predict(state)
    >>> scenarios = wm.simulate(state, n_steps=5)
    >>> best = wm.plan(state, goal="maximize reward")
    """

    def __init__(
        self,
        n_nodes:     int = 512,
        n_sim_steps: int = 5,
        n_scenarios: int = 4,
    ):
        self.n_nodes     = n_nodes
        self.n_sim_steps = n_sim_steps
        self.n_scenarios = n_scenarios

        self.transition  = TransitionMemory(n_nodes)
        self.reward_model = RewardModel(n_nodes)

        self.state_history: List[WorldState] = []
        self.state_counter = 0
        self.n_predicts    = 0

    # -------------------------------------------------------- #
    # State encoding
    # -------------------------------------------------------- #

    def encode_context(
        self,
        context: str,
        encoder = None,
    ) -> np.ndarray:
        """
        Encode a text context into a sparse spike pattern.

        If encoder (sentence transformer) provided, uses it.
        Otherwise falls back to hash-based sparse encoding.
        """
        if encoder is not None:
            vec    = encoder.encode(
                [context], convert_to_numpy=True)[0]
            # Project to n_nodes with sparse threshold
            if len(vec) != self.n_nodes:
                # Random projection
                np.random.seed(42)
                proj  = np.random.randn(len(vec), self.n_nodes)
                proj /= np.linalg.norm(proj, axis=0, keepdims=True)
                vec   = vec @ proj
            threshold = np.percentile(vec, 90)
            return (vec > threshold).astype(np.float32)
        else:
            # Hash-based sparse encoding (no encoder needed)
            spikes = np.zeros(self.n_nodes, dtype=np.float32)
            words  = context.lower().split()
            for word in words:
                idx = hash(word) % self.n_nodes
                spikes[idx] = 1.0
                # Also activate neighbours
                spikes[(idx + 1) % self.n_nodes] = 0.5
            # Threshold to keep sparse
            spikes = (spikes > 0.3).astype(np.float32)
            return spikes

    def create_state(
        self,
        context:   str,
        encoder    = None,
        parent_id: Optional[int] = None,
    ) -> WorldState:
        """Create a new world state from context."""
        spikes    = self.encode_context(context, encoder)
        reward    = self.reward_model.score(spikes)
        state     = WorldState(
            state_id  = self.state_counter,
            spikes    = spikes,
            context   = context,
            reward    = reward,
            parent_id = parent_id,
        )
        self.state_counter     += 1
        self.state_history.append(state)
        return state

    # -------------------------------------------------------- #
    # Core operations
    # -------------------------------------------------------- #

    def observe(
        self,
        state_a: WorldState,
        state_b: WorldState,
    ) -> None:
        """
        Learn transition: state_a → state_b.
        Call this when you observe a real transition.
        """
        self.transition.learn_transition(
            state_a.spikes, state_b.spikes)

    def predict(self, state: WorldState) -> WorldState:
        """
        Predict the next state from current state.
        """
        self.n_predicts += 1
        predicted_spikes = self.transition.predict_next(
            state.spikes)
        reward = self.reward_model.score(predicted_spikes)
        return WorldState(
            state_id  = self.state_counter,
            spikes    = predicted_spikes,
            context   = f"predicted from {state.context[:30]}",
            reward    = reward,
            parent_id = state.state_id,
        )

    def simulate(
        self,
        initial_state: WorldState,
        n_steps:       Optional[int] = None,
    ) -> List[WorldState]:
        """
        Simulate multiple steps forward from initial state.

        Returns list of predicted future states.
        """
        n_steps = n_steps or self.n_sim_steps
        states  = [initial_state]
        current = initial_state

        for _ in range(n_steps):
            next_state = self.predict(current)
            states.append(next_state)
            current = next_state

        return states

    def plan(
        self,
        initial_state: WorldState,
        goal:          str = "maximize reward",
    ) -> Dict:
        """
        Run multiple simulation scenarios and pick the best.

        Returns
        -------
        dict with: best_path, best_reward, all_scenarios
        """
        scenarios = []

        for scenario_i in range(self.n_scenarios):
            # Add noise to explore different futures
            noisy_spikes = initial_state.spikes.copy()
            noise_idx    = np.random.choice(
                self.n_nodes,
                size=int(self.n_nodes * 0.05),
                replace=False,
            )
            noisy_spikes[noise_idx] = 1 - noisy_spikes[noise_idx]

            noisy_state = WorldState(
                state_id  = self.state_counter + scenario_i,
                spikes    = noisy_spikes,
                context   = f"scenario_{scenario_i}",
                reward    = self.reward_model.score(noisy_spikes),
            )

            path         = self.simulate(noisy_state)
            total_reward = sum(s.reward for s in path)
            scenarios.append({
                'scenario_id'  : scenario_i,
                'path'         : path,
                'total_reward' : round(total_reward, 4),
                'avg_reward'   : round(total_reward/len(path), 4),
                'n_steps'      : len(path),
            })

        # Pick best scenario
        best = max(scenarios, key=lambda x: x['total_reward'])

        return {
            'goal'         : goal,
            'best_scenario': best['scenario_id'],
            'best_reward'  : best['total_reward'],
            'best_path_len': best['n_steps'],
            'all_scenarios': [
                {k: v for k, v in s.items() if k != 'path'}
                for s in scenarios
            ],
        }

    def reinforce(
        self,
        state:  WorldState,
        reward: float,
    ) -> None:
        """Give reward feedback to update the reward model."""
        self.reward_model.update(state.spikes, reward)

    def stats(self) -> dict:
        return {
            'n_nodes'      : self.n_nodes,
            'n_states'     : len(self.state_history),
            'n_transitions': self.transition.n_transitions,
            'n_predicts'   : self.n_predicts,
            'n_scenarios'  : self.n_scenarios,
        }

    def __repr__(self) -> str:
        return (f"WorldModel(nodes={self.n_nodes}, "
                f"states={len(self.state_history)}, "
                f"transitions={self.transition.n_transitions})")
