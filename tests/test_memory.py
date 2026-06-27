"""Unit tests for zinohk.learning.memory"""
import numpy as np
import pytest
from zinohk.learning.memory import FastMemory, SlowMemory, MemorySystem


def test_fast_memory_stores():
    fm = FastMemory(capacity=10)
    fm.store(np.array([0.5, 0.5]), label=0)
    assert fm.size == 1


def test_fast_memory_capacity_limit():
    fm = FastMemory(capacity=5)
    for i in range(10):
        fm.store(np.array([float(i)]), label=0)
    assert fm.size == 5, "Should not exceed capacity"


def test_fast_memory_sample():
    fm = FastMemory(capacity=20)
    for i in range(10):
        fm.store(np.array([float(i)]), label=i % 2)
    samples = fm.sample(5)
    assert len(samples) == 5


def test_slow_memory_learns():
    np.random.seed(42)
    sm = SlowMemory(n_inputs=4, n_outputs=2, lr=0.05)
    from zinohk.learning.memory import MemoryTrace
    trace = MemoryTrace(
        x     = np.array([1.0, 0.0, 1.0, 0.0]),
        label = 0,
        step  = 1,
    )
    for _ in range(50):
        sm.replay_update(trace)
    pred = sm.predict(np.array([1.0, 0.0, 1.0, 0.0]))
    assert pred == 0


def test_memory_system_no_forgetting():
    np.random.seed(42)
    ms = MemorySystem(n_inputs=5, n_outputs=2,
                      capacity=50, replay_k=10, lr_slow=0.005)
    A = np.array([1.0, 0.0, 1.0, 0.0, 0.5])
    B = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
    for _ in range(30):
        ms.learn(A, label=0)
    pred_A_before = ms.predict(A)
    for _ in range(30):
        ms.learn(B, label=1)
    pred_A_after = ms.predict(A)
    assert pred_A_before == pred_A_after == 0, \
        "A should not be forgotten after learning B"


def test_memory_system_stats_keys():
    ms = MemorySystem(n_inputs=3, n_outputs=2)
    ms.learn(np.array([0.5, 0.5, 0.5]), label=0)
    stats = ms.stats()
    for key in ['steps','fast_stored','slow_replays','slow_w_norm']:
        assert key in stats
