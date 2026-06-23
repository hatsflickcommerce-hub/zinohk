"""
Unit tests for zinohk.core.node.ZNode
"""

import numpy as np
import pytest
from zinohk.core.node import ZNode


def test_znode_fires_above_threshold():
    node = ZNode(node_id='n0', n_inputs=2, threshold=0.1)
    node.weights = np.array([0.4, 0.4])
    out = node.forward(np.array([0.9, 0.8]))
    assert out > 0.0, "Node should fire when input > threshold"


def test_znode_silent_below_threshold():
    node = ZNode(node_id='n1', n_inputs=2, threshold=10.0)
    out = node.forward(np.array([0.1, 0.1]))
    assert out == 0.0, "Node should be silent when input < threshold"


def test_znode_threshold_rises_after_firing():
    node = ZNode(node_id='n2', n_inputs=2, threshold=0.1)
    node.weights = np.array([0.4, 0.4])
    theta_before = node.threshold
    node.forward(np.array([0.9, 0.8]))
    assert node.threshold > theta_before, "Threshold should rise after firing"


def test_znode_threshold_drops_when_silent():
    node = ZNode(node_id='n3', n_inputs=2, threshold=10.0)
    theta_before = node.threshold
    node.forward(np.array([0.1, 0.1]))
    assert node.threshold < theta_before, "Threshold should drop when silent"


def test_znode_weights_update_after_firing():
    node = ZNode(node_id='n4', n_inputs=2, threshold=0.1)
    node.weights = np.array([0.4, 0.4])
    w_before = node.weights.copy()
    node.forward(np.array([0.9, 0.8]))
    assert not np.allclose(node.weights, w_before), "Weights should update after firing"


def test_znode_activity_rate():
    node = ZNode(node_id='n5', n_inputs=2, threshold=0.1)
    node.weights = np.array([0.4, 0.4])
    for _ in range(4):
        node.forward(np.array([0.9, 0.8]))
    assert node.activity_rate > 0.0, "Activity rate should be > 0"


def test_znode_wrong_input_shape():
    node = ZNode(node_id='n6', n_inputs=3)
    with pytest.raises(AssertionError):
        node.forward(np.array([0.5, 0.5]))


def test_znode_stats_keys():
    node = ZNode(node_id='n7', n_inputs=2)
    stats = node.stats()
    for key in ['node_id', 'threshold', 'activity_rate', 'weight_norm', 'steps', 'fires']:
        assert key in stats, f"Missing key: {key}"
