"""Smoke tests for the plotting helpers (Agg backend, no display)."""

import matplotlib

matplotlib.use("Agg")

from pvhostprob.feeder import build_lv_feeder
from pvhostprob.hc import exceedance_curve, locational_probabilistic_hc
from pvhostprob.plot import plot_exceedance, plot_locational
from pvhostprob.scenarios import correlated_capacity_factors


def test_plot_exceedance_smoke(tmp_path):
    net = build_lv_feeder(n_buses=6)
    cf = correlated_capacity_factors(40, 2, seed=0)
    curve = exceedance_curve(net, [2, 4], [2, 8, 16], cf)
    fig = plot_exceedance(curve, risk=0.05, markers={"demo": 8.0},
                          path=str(tmp_path / "exc.png"))
    assert (tmp_path / "exc.png").exists()
    assert len(fig.axes) == 1


def test_plot_locational_smoke(tmp_path):
    net = build_lv_feeder(n_buses=6)
    cf = correlated_capacity_factors(40, 1, seed=0)
    loc = locational_probabilistic_hc(net, [1, 5], cf, risk=0.05, tol_kw=2.0)
    fig = plot_locational(loc, path=str(tmp_path / "loc.png"))
    assert (tmp_path / "loc.png").exists()
    assert len(fig.axes[0].patches) == 2
