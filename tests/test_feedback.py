"""Unit tests for zinohk.feedback.predictor"""
import numpy as np
import pytest
from zinohk.feedback.predictor import PredictiveNode


def test_first_input_high_error():
    pnode = PredictiveNode('p0', n_inputs=4, epsilon=0.05)
    x     = np.array([0.8, 0.6, 0.4, 0.2])
    err   = pnode.forward(x)
    assert np.sum(np.abs(err)) > 0, "First input should produce error"


def test_repeated_input_reduces_error():
    pnode = PredictiveNode('p0', n_inputs=4, epsilon=0.05, lr=0.4)
    x     = np.array([0.8, 0.6, 0.4, 0.2])
    errors = []
    for _ in range(6):
        err = pnode.forward(x)
        errors.append(float(np.sum(np.abs(err))))
    assert errors[-1] < errors[0], "Error should decrease over repetitions"


def test_novel_input_high_error_after_learning():
    pnode = PredictiveNode('p0', n_inputs=4, epsilon=0.05, lr=0.4)
    x     = np.array([0.8, 0.6, 0.4, 0.2])
    for _ in range(8):
        pnode.forward(x)
    novel = np.array([0.1, 0.9, 0.1, 0.9])
    err   = pnode.forward(novel)
    assert np.sum(np.abs(err)) > 0, "Novel input should produce error"


def test_silence_rate_increases():
    pnode = PredictiveNode('p0', n_inputs=3, epsilon=0.05, lr=0.5)
    x     = np.array([0.8, 0.5, 0.3])
    for _ in range(10):
        pnode.forward(x)
    assert pnode.silence_rate > 0, "Silence rate should increase"


def test_stats_keys():
    pnode = PredictiveNode('p0', n_inputs=3)
    pnode.forward(np.array([0.5, 0.5, 0.5]))
    stats = pnode.stats()
    for key in ['node_id','n_updates','silence_rate',
                'surprise_rate','pred_norm']:
        assert key in stats
