"""
Unit tests for zinohk.core.synapse.Synapse
"""

import pytest
from zinohk.core.synapse import Synapse


def test_synapse_init():
    s = Synapse(pre_id='n0', post_id='n1', weight=0.2)
    assert s.weight == 0.2
    assert s.pre_id  == 'n0'
    assert s.post_id == 'n1'


def test_hebbian_strengthens_when_both_fire():
    s = Synapse(pre_id='n0', post_id='n1', weight=0.1)
    w_before = s.weight
    s.hebbian_update(pre_fired=0.9, post_fired=0.8)
    assert s.weight > w_before, "Weight should increase when both fire"


def test_hebbian_no_change_when_silent():
    s = Synapse(pre_id='n0', post_id='n1', weight=0.1)
    w_before = s.weight
    s.hebbian_update(pre_fired=0.0, post_fired=0.0)
    assert s.weight == w_before, "Weight should not change when both silent"


def test_hebbian_no_change_when_pre_silent():
    s = Synapse(pre_id='n0', post_id='n1', weight=0.1)
    w_before = s.weight
    s.hebbian_update(pre_fired=0.0, post_fired=0.9)
    assert s.weight == w_before, "Weight should not change when pre is silent"


def test_hebbian_weight_clipped_max():
    s = Synapse(pre_id='n0', post_id='n1', weight=4.99, lr=1.0)
    s.hebbian_update(pre_fired=1.0, post_fired=1.0)
    assert s.weight <= 5.0, "Weight should not exceed 5.0"


def test_hebbian_weight_clipped_min():
    s = Synapse(pre_id='n0', post_id='n1', weight=-4.99, lr=1.0)
    s.hebbian_update(pre_fired=1.0, post_fired=1.0)
    assert s.weight >= -5.0, "Weight should not go below -5.0"


def test_stdp_potentiation_pre_before_post():
    s = Synapse(pre_id='n0', post_id='n1', weight=0.1)
    w_before = s.weight
    # pre fires at t=5, post fires at t=15 → pre before post → potentiation
    s.stdp_update(pre_time=5.0, post_time=15.0)
    assert s.weight > w_before, "STDP should potentiate when pre fires before post"


def test_stdp_depression_post_before_pre():
    s = Synapse(pre_id='n0', post_id='n1', weight=0.1)
    w_before = s.weight
    # post fires at t=5, pre fires at t=15 → post before pre → depression
    s.stdp_update(pre_time=15.0, post_time=5.0)
    assert s.weight < w_before, "STDP should depress when post fires before pre"


def test_synapse_repr():
    s = Synapse(pre_id='a', post_id='b', weight=0.25)
    assert 'a' in repr(s)
    assert 'b' in repr(s)
