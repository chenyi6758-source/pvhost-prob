"""Tests for metrics helpers."""

import numpy as np
import pytest

from pvhostprob.metrics import wilson_interval, summarize_run


def test_wilson_contains_point_estimate():
    for k, n in [(0, 50), (3, 50), (25, 50), (50, 50), (1, 5)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_wilson_edges():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and hi < 0.05
    lo, hi = wilson_interval(100, 100)
    assert hi == 1.0 and lo > 0.95


def test_wilson_invalid():
    with pytest.raises(ValueError):
        wilson_interval(5, 0)
    with pytest.raises(ValueError):
        wilson_interval(6, 5)


def test_summarize_run():
    n = 40
    res = {
        "violated": np.arange(n) < 4,
        "converged": np.ones(n, dtype=bool),
        "max_vm": np.linspace(1.0, 1.08, n),
        "max_loading": np.linspace(20, 90, n),
        "n_scenarios": n,
        "n_failed": 0,
    }
    s = summarize_run(res)
    assert s["p_viol"] == pytest.approx(0.1)
    assert s["n_failed"] == 0
    lo, hi = s["p_viol_ci95"]
    assert lo <= 0.1 <= hi
    assert s["max_vm_p99"] > s["max_vm_mean"]
