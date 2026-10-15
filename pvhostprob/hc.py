"""Hosting-capacity estimators.

Two complementary answers for the same feeder:

* :func:`deterministic_hc` — worst-case snapshot (every candidate bus at full
  nameplate PV, nominal load), the classic deterministic screening baseline
  (cf. the bisection screening in pvhostcap).
* :func:`probabilistic_hc` — the largest uniform per-bus PV size whose Monte
  Carlo violation probability stays at or below an accepted ``risk`` level,
  evaluated over correlated PV scenarios (and optional load uncertainty).
  This is the stochastic hosting-capacity view used by EPRI's stochastic HC
  methodology and recent Monte Carlo HC studies: the planner names an
  explicit risk instead of hiding it inside a worst-case assumption.
* :func:`locational_probabilistic_hc` — the same, bus by bus: how much PV a
  single bus can take on its own at the accepted risk, plus which limit binds
  there (voltage or thermal).
* :func:`exceedance_curve` — violation probability as a function of installed
  PV, the curve every HC number below is read off.

Violation probability is monotone non-decreasing in installed PV (more
injection can only push voltages higher and reverse flows larger), so plain
bisection converges.
"""

import numpy as np
import pandas as pd

from pvhostprob.simulate import run_scenarios
from pvhostprob.metrics import wilson_interval


def violation_probability(res):
    """Empirical P(violation) from a :func:`run_scenarios` result."""
    return float(np.asarray(res["violated"]).mean())


def _bisect(p_of_kw, target, tol_kw=0.5, hi_cap_kw=500.0, max_iter=60):
    """Bisection for the largest kw with p_of_kw(kw) <= target.

    Assumes p_of_kw is non-decreasing. Returns (kw_lo, p_lo, capped) where
    capped=True means even hi_cap_kw stayed within target.
    """
    p_lo = p_of_kw(0.0)
    if p_lo > target:
        return 0.0, p_lo, False
    hi = 10.0
    p_hi = p_of_kw(hi)
    while p_hi <= target and hi < hi_cap_kw:
        hi = min(hi * 2.0, hi_cap_kw)
        p_hi = p_of_kw(hi)
        if hi >= hi_cap_kw:
            break
    if p_hi <= target:
        return hi_cap_kw, p_hi, True
    lo = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo <= tol_kw:
            break
        if p_of_kw(mid) <= target:
            lo = mid
        else:
            hi = mid
    p_lo = p_of_kw(lo)
    return lo, p_lo, False


def _worst_case_net(net, pv_buses, pv_kw):
    """Deep-copied net with PV sgens at nameplate, solved at flat start."""
    import copy as _copy

    import pandapower as pp

    work = _copy.deepcopy(net)
    sgen_idx = [
        pp.create_sgen(work, bus=b, p_mw=0.0, name=f"pv_diag_{b}")
        for b in pv_buses
    ]
    work.sgen.loc[sgen_idx, "p_mw"] = np.asarray(pv_kw, dtype=float) / 1000.0
    pp.runpp(work, init="flat", calculate_voltage_angles=False, verbose=False)
    return work


def diagnose_binding(net, pv_buses, pv_kw, vm_max=1.05, max_loading_pct=100.0):
    """Which limit binds first under worst-case conditions (cf = 1, nominal load).

    Compares *normalised* headrooms: voltage headroom is measured against the
    allowed rise above 1.0 p.u., thermal headroom against the line rating, so
    the two are on a comparable 0..1 scale. The smaller headroom binds first.

    Returns dict(binding="voltage"|"thermal", location=str, value=float).
    """
    pv_buses = list(pv_buses)
    work = _worst_case_net(net, pv_buses, pv_kw)
    max_vm = float(work.res_bus.vm_pu.max())
    max_ld = (
        float(work.res_line.loading_percent.max()) if len(work.line) else 0.0
    )
    vm_headroom = (vm_max - max_vm) / max(vm_max - 1.0, 1e-9)
    ld_headroom = (max_loading_pct - max_ld) / max(max_loading_pct, 1e-9)
    if vm_headroom <= ld_headroom:
        worst_bus = int(work.res_bus.vm_pu.idxmax())
        return {
            "binding": "voltage",
            "location": str(work.bus.name.at[worst_bus]),
            "value": max_vm,
        }
    worst_line = int(work.res_line.loading_percent.idxmax())
    return {
        "binding": "thermal",
        "location": str(work.line.name.at[worst_line]),
        "value": max_ld,
    }


def deterministic_hc(
    net,
    pv_buses,
    vm_max=1.05,
    max_loading_pct=100.0,
    tol_kw=0.25,
    hi_cap_kw=500.0,
):
    """Worst-case deterministic hosting capacity (uniform kW per candidate bus).

    Single snapshot: every candidate bus injects its full nameplate (cf = 1),
    loads at nominal. Returns the largest per-bus kW with zero violations.
    """
    pv_buses = list(pv_buses)
    cf = np.ones((1, len(pv_buses)))

    def p_of_kw(kw):
        res = run_scenarios(
            net,
            pv_buses,
            [kw] * len(pv_buses),
            cf,
            vm_max=vm_max,
            max_loading_pct=max_loading_pct,
        )
        return violation_probability(res)

    hc, _, capped = _bisect(p_of_kw, 0.0, tol_kw=tol_kw, hi_cap_kw=hi_cap_kw)
    return {
        "hc_kw_per_bus": hc,
        "hc_kw_total": hc * len(pv_buses),
        "capped": capped,
        "binding": diagnose_binding(
            net, pv_buses, [hc] * len(pv_buses), vm_max, max_loading_pct
        ),
    }


def probabilistic_hc(
    net,
    pv_buses,
    cf,
    load_mult=None,
    risk=0.05,
    vm_max=1.05,
    max_loading_pct=100.0,
    tol_kw=0.5,
    hi_cap_kw=500.0,
):
    """Probabilistic hosting capacity at an accepted violation risk.

    Largest uniform per-bus PV size (kW) with empirical P(violation) <= risk
    over the given scenarios. Also reports the Wilson 95% interval of the
    violation probability at the returned size, so the Monte Carlo sampling
    noise is explicit instead of hidden.
    """
    pv_buses = list(pv_buses)
    cf = np.asarray(cf, dtype=float)
    if cf.shape[1] != len(pv_buses):
        raise ValueError("cf columns must match len(pv_buses)")
    if not 0.0 <= risk < 1.0:
        raise ValueError("risk must be in [0, 1)")

    last_res = {}

    def p_of_kw(kw):
        res = run_scenarios(
            net,
            pv_buses,
            [kw] * len(pv_buses),
            cf,
            load_mult=load_mult,
            vm_max=vm_max,
            max_loading_pct=max_loading_pct,
        )
        last_res["res"] = res
        return violation_probability(res)

    hc, p_lo, capped = _bisect(
        p_of_kw, risk, tol_kw=tol_kw, hi_cap_kw=hi_cap_kw
    )
    res = last_res.get("res")
    k = int(res["violated"].sum()) if res is not None else 0
    n = int(res["n_scenarios"]) if res is not None else cf.shape[0]
    ci_lo, ci_hi = wilson_interval(k, n)
    return {
        "hc_kw_per_bus": hc,
        "hc_kw_total": hc * len(pv_buses),
        "p_viol": p_lo,
        "p_viol_ci95": (ci_lo, ci_hi),
        "n_scenarios": n,
        "n_failed": int(res["n_failed"]) if res is not None else 0,
        "capped": capped,
        "binding": diagnose_binding(
            net, pv_buses, [hc] * len(pv_buses), vm_max, max_loading_pct
        ),
    }


def locational_probabilistic_hc(
    net,
    candidate_buses,
    cf,
    load_mult=None,
    risk=0.05,
    vm_max=1.05,
    max_loading_pct=100.0,
    tol_kw=0.5,
    hi_cap_kw=500.0,
):
    """Per-bus probabilistic hosting capacity (PV at one bus at a time).

    ``cf`` may have several columns; only its first column is used, since a
    single-site run depends only on the marginal distribution, not on the
    spatial correlation structure.
    """
    candidate_buses = list(candidate_buses)
    cf = np.asarray(cf, dtype=float)
    cf_1 = cf[:, [0]]
    rows = []
    for b in candidate_buses:
        r = probabilistic_hc(
            net,
            [b],
            cf_1,
            load_mult=load_mult,
            risk=risk,
            vm_max=vm_max,
            max_loading_pct=max_loading_pct,
            tol_kw=tol_kw,
            hi_cap_kw=hi_cap_kw,
        )
        bus_name = str(net.bus.name.at[b])
        rows.append(
            {
                "bus": b,
                "bus_name": bus_name,
                "hc_kw": r["hc_kw_per_bus"],
                "p_viol": r["p_viol"],
                "ci95_lo": r["p_viol_ci95"][0],
                "ci95_hi": r["p_viol_ci95"][1],
                "binding": r["binding"]["binding"],
                "binding_location": r["binding"]["location"],
                "n_failed": r["n_failed"],
            }
        )
    return pd.DataFrame(rows)


def exceedance_curve(
    net,
    pv_buses,
    capacities_kw,
    cf,
    load_mult=None,
    vm_max=1.05,
    max_loading_pct=100.0,
):
    """Violation probability (with Wilson 95% CI) at each PV size."""
    pv_buses = list(pv_buses)
    rows = []
    for kw in capacities_kw:
        res = run_scenarios(
            net,
            pv_buses,
            [kw] * len(pv_buses),
            cf,
            load_mult=load_mult,
            vm_max=vm_max,
            max_loading_pct=max_loading_pct,
        )
        k = int(res["violated"].sum())
        n = int(res["n_scenarios"])
        ci_lo, ci_hi = wilson_interval(k, n)
        rows.append(
            {
                "kw_per_bus": float(kw),
                "p_viol": k / n,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "n_failed": int(res["n_failed"]),
            }
        )
    return pd.DataFrame(rows)
