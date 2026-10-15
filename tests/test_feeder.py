"""Tests for the synthetic LV feeder builder."""

import numpy as np
import pandapower as pp

from pvhostprob.feeder import build_lv_feeder


def test_feeder_structure():
    net = build_lv_feeder(n_buses=8)
    assert len(net.bus) == 8
    assert len(net.line) == 7
    assert len(net.load) == 7  # every bus except the substation
    assert len(net.ext_grid) == 1
    assert list(net.bus.name) == [f"LV{i}" for i in range(8)]


def test_feeder_converges_and_voltage_drops():
    net = build_lv_feeder(n_buses=8)
    pp.runpp(net)
    assert net.converged
    vm = net.res_bus.vm_pu.values
    # voltage strictly decreases with distance under passive load
    assert np.all(np.diff(vm) < 0), vm
    assert vm[0] == 1.0  # slack


def test_feeder_custom_size():
    net = build_lv_feeder(n_buses=4, load_kw=5.0)
    assert len(net.bus) == 4
    pp.runpp(net)
    assert net.converged
