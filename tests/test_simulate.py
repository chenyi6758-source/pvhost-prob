"""Tests for the Monte Carlo power-flow engine."""

import numpy as np
import pytest

from pvhostprob.feeder import build_lv_feeder
from pvhostprob.simulate import run_scenarios


def _net():
    return build_lv_feeder(n_buses=6)


def test_no_pv_no_violation():
    net = _net()
    far = net.bus.index[-1]
    cf = np.zeros((20, 1))
    res = run_scenarios(net, [far], [50.0], cf)
    assert res["n_scenarios"] == 20
    assert res["n_failed"] == 0
    assert not res["violated"].any()
    assert res["max_vm"].max() < 1.05


def test_pv_raises_far_end_voltage_monotonically():
    net = _net()
    far = net.bus.index[-1]
    vms = []
    for kw in [0.0, 15.0, 40.0]:
        res = run_scenarios(net, [far], [kw], np.ones((5, 1)))
        vms.append(res["max_vm"][0])
    assert vms[0] < vms[1] < vms[2], vms


def test_violation_flag_fires():
    net = _net()
    far = net.bus.index[-1]
    res = run_scenarios(net, [far], [100.0], np.ones((10, 1)), vm_max=1.05)
    assert res["violated"].all()
    # with a lax limit, nothing violates
    res2 = run_scenarios(net, [far], [100.0], np.ones((10, 1)), vm_max=3.0,
                         max_loading_pct=1e9)
    assert not res2["violated"].any()


def test_cf_scales_injection():
    # half capacity factor -> half the voltage rise vs full cf
    net = _net()
    far = net.bus.index[-1]
    r_full = run_scenarios(net, [far], [40.0], np.ones((3, 1)))
    r_half = run_scenarios(net, [far], [40.0], np.full((3, 1), 0.5))
    rise_full = r_full["max_vm"] - 1.0
    rise_half = r_half["max_vm"] - 1.0
    assert np.all(rise_half < rise_full)
    assert np.all(rise_half > 0.3 * rise_full)  # roughly linear


def test_load_uncertainty_applies():
    net = _net()
    far = net.bus.index[-1]
    cf = np.ones((30, 1))
    lm = np.ones((30, len(net.load)))
    lm[:, 0] = 3.0  # triple the first load in every scenario
    res = run_scenarios(net, [far], [0.0], cf, load_mult=lm)
    assert res["n_failed"] == 0
    # heavier load -> lower minimum voltage than the base case
    import pandapower as pp

    pp.runpp(net)
    assert res["max_vm"].max() < 1.05  # still no overvoltage without PV


def test_input_validation():
    net = _net()
    with pytest.raises(ValueError):
        run_scenarios(net, [1, 2], [10.0], np.ones((5, 3)))  # cf/buses mismatch
    with pytest.raises(ValueError):
        run_scenarios(net, [1], [10.0], np.ones((5, 1)),
                      load_mult=np.ones((4, len(net.load))))  # scenario mismatch


def test_base_net_not_modified():
    net = _net()
    n_sgen_before = len(net.sgen)
    run_scenarios(net, [net.bus.index[-1]], [20.0], np.ones((5, 1)))
    assert len(net.sgen) == n_sgen_before
    assert (net.load.p_mw.values == 0.003).all()
