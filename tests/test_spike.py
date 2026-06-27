"""Unit tests for zinohk.encoding.spike"""
import numpy as np
import pytest
from zinohk.encoding.spike import SpikeEncoder, SpikeDecoder, Spike


def test_encode_strong_signal_fires_early():
    enc = SpikeEncoder(T_max=20.0, threshold=0.05)
    spikes = enc.encode(np.array([0.9, 0.1]))
    times  = {s.channel: s.spike_time for s in spikes}
    assert times[0] < times[1], "Strong signal should fire earlier"


def test_encode_silent_channel_no_spike():
    enc    = SpikeEncoder(T_max=20.0, threshold=0.05)
    spikes = enc.encode(np.array([0.9, 0.0, 0.5]))
    channels = [s.channel for s in spikes]
    assert 1 not in channels, "Zero input should produce no spike"


def test_encode_below_threshold_silent():
    enc    = SpikeEncoder(T_max=20.0, threshold=0.1)
    spikes = enc.encode(np.array([0.05, 0.05]))
    assert len(spikes) == 0, "Below threshold should be silent"


def test_decode_roundtrip():
    enc = SpikeEncoder(T_max=20.0, threshold=0.05)
    dec = SpikeDecoder(T_max=20.0)
    x      = np.array([0.9, 0.5, 0.0, 0.7])
    spikes = enc.encode(x)
    x_back = dec.first_spike_decode(spikes, n_channels=4)
    assert x_back[2] == 0.0, "Silent channel should decode to 0"
    assert x_back[0] > x_back[1], "Strong channel should decode higher"


def test_spike_overlap_identical():
    enc    = SpikeEncoder(T_max=20.0, threshold=0.05)
    x      = np.array([0.9, 0.5, 0.7])
    spikes = enc.encode(x)
    corr   = enc.spike_overlap(spikes, spikes)
    assert corr > 0.9, "Identical spike trains should have high overlap"


def test_spike_overlap_different():
    enc = SpikeEncoder(T_max=20.0, threshold=0.05)
    a   = enc.encode(np.array([0.9, 0.0, 0.0]))
    b   = enc.encode(np.array([0.0, 0.0, 0.9]))
    corr = enc.spike_overlap(a, b)
    assert corr == 0.0, "Non-overlapping channels should have zero overlap"


def test_spikes_sorted_by_time():
    enc    = SpikeEncoder(T_max=20.0, threshold=0.05)
    spikes = enc.encode(np.array([0.3, 0.9, 0.1, 0.7]))
    times  = [s.spike_time for s in spikes]
    assert times == sorted(times), "Spikes should be sorted by time"
