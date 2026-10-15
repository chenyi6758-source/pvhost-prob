"""Tests for the hosting-capacity estimators."""

import numpy as np
import pandas as pd
import pytest

from pvhostprob.feeder import build_lv_feeder
from pvhostprob.hc import (
    deterministic_hc,
    probabilistic_hc,
    locational_probabilistic_hc,
    exceedance_curve,
    violation_probability,
    diagnose_binding,
)
from pvhostprob.scenarios import correlated_capacity_factors


def _net():
    return build_lv_feeder(n_buses=6)


def _cf(n=60, k=3, seed=0):
    return correlated_capacity_factors(n, k, seed=seed)


def test_violation_probability():
    res = {"violated": np.array([True, False, False, False])}
    assert violation_probability(res) == pytest.approx(0.25)


def test_deterministic_hc_sane():
    net = _net()
    det = deterministic_hc(net, [2, 4, 5])
    assert det["hc_kw_per_bus"] > 1.0
    assert det["hc_kw_total"] == pytest.approx(det["hc_kw_per_bus"] * 3)
    assert det["binding"]["binding"] == "voltage"  # LV feeder: overvoltage binds first
    assert not det["capped"]


def test_probabilistic_hc_bounds_deterministic():
    # With nominal loads, every stochastic scenario is bounded above by the
    # worst-case snapshot (cf <= 1 everywhere), so the 5%-risk HC must be at
    # least the deterministic (0%-risk) HC, up to bisection tolerance.
    net = _net()
    buses = [2, 4, 5]
    det = deterministic_hc(net, buses)
    pro = probabilistic_hc(net, buses, _cf(k=3), risk=0.05, tol_kw=2.0)
    assert pro["hc_kw_per_bus"] >= det["hc_kw_per_bus"] - 3.0
    assert 0.0 <= pro["p_viol"] <= 0.05 + 1.0 / 60 + 1e-9
    lo, hi = pro["p_viol_ci95"]
    assert lo <= pro["p_viol"] <= hi
    assert pro["n_failed"] == 0


def test_exceedance_curve_monotone():
    net = _net()
    curve = exceedance_curve(net, [2, 4, 5], [2, 8, 16, 30], _cf(k=3))
    assert isinstance(curve, pd.DataFrame)
    assert (np.diff(curve["p_viol"].values) >= 0).all()
    assert (curve["ci95_lo"] <= curve["p_viol"]).all()
    assert (curve["p_viol"] <= curve["ci95_hi"]).all()


def test_locational_far_end_lower():
    # Classic hosting-capacity physics: the far-end bus hosts less than a
    # bus next to the substation.
    net = _net()
    loc = locational_probabilistic_hc(net, [1, 5], _cf(n=50, k=1), risk=0.05,
                                      tol_kw=2.0)
    hc_near = loc.loc[loc["bus"] == 1, "hc_kw"].iloc[0]
    hc_far = loc.loc[loc["bus"] == 5, "hc_kw"].iloc[0]
    assert hc_far < hc_near, loc[["bus_name", "hc_kw"]].to_string()
    assert set(loc["binding"]).issubset({"voltage", "thermal"})


def test_diagnose_binding_voltage():
    net = _net()
    det = deterministic_hc(net, [5], tol_kw=0.5)
    d = diagnose_binding(net, [5], [det["hc_kw_per_bus"]])
    assert d["binding"] == "voltage"
    assert d["location"] == "LV5"
    assert 1.0 < d["value"] <= 1.05 + 1e-6
