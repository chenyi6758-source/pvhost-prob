"""Monte Carlo AC power-flow engine.

For each scenario the engine sets every candidate bus's PV injection to
``pv_kw * capacity_factor`` (and optionally jitters the loads), runs a full AC
power flow with pandapower, and records the worst bus voltage and worst line
loading. A scenario *violates* if any bus exceeds ``vm_max`` or any line
exceeds ``max_loading_pct``. Non-converged scenarios are counted as violations
(conservative: the planner cannot rely on an operating point the solver cannot
find) and reported via ``n_failed``.
"""

import copy

import numpy as np
import pandapower as pp


def run_scenarios(
    net,
    pv_buses,
    pv_kw,
    cf,
    load_mult=None,
    vm_max=1.05,
    max_loading_pct=100.0,
):
    """Run one AC power flow per scenario.

    Parameters
    ----------
    net : pandapowerNet
        Base network (not modified; an internal copy is used).
    pv_buses : array-like of int
        Bus indices where PV may connect.
    pv_kw : array-like of float
        Installed PV (kW) at each bus in ``pv_buses``.
    cf : numpy.ndarray, shape (n_scenarios, len(pv_buses))
        Per-scenario capacity factors in [0, 1].
    load_mult : numpy.ndarray, shape (n_scenarios, n_loads), optional
        Per-scenario load multipliers (see :func:`load_multipliers`).
    vm_max : float
        Bus voltage magnitude limit (p.u.).
    max_loading_pct : float
        Line loading limit (percent of rating).

    Returns
    -------
    dict with keys ``max_vm``, ``max_loading``, ``converged``, ``violated``
    (per-scenario arrays), ``n_scenarios`` and ``n_failed``.
    """
    pv_buses = list(pv_buses)
    pv_kw = np.asarray(pv_kw, dtype=float)
    cf = np.asarray(cf, dtype=float)
    n = cf.shape[0]
    if cf.shape[1] != len(pv_buses):
        raise ValueError(
            f"cf has {cf.shape[1]} columns but {len(pv_buses)} pv_buses"
        )
    if load_mult is not None:
        load_mult = np.asarray(load_mult, dtype=float)
        if load_mult.shape[0] != n:
            raise ValueError("load_mult scenario count must match cf")

    work = copy.deepcopy(net)
    sgen_idx = [
        pp.create_sgen(work, bus=b, p_mw=0.0, q_mvar=0.0, name=f"pv_hostprob_{b}")
        for b in pv_buses
    ]
    base_load_p = work.load.p_mw.values.copy()
    if load_mult is not None and load_mult.shape[1] != len(base_load_p):
        raise ValueError("load_mult load count must match net loads")

    max_vm = np.empty(n)
    max_loading = np.empty(n)
    converged = np.zeros(n, dtype=bool)
    has_lines = len(work.line) > 0

    for s in range(n):
        work.sgen.loc[sgen_idx, "p_mw"] = pv_kw / 1000.0 * cf[s]
        if load_mult is not None:
            work.load.p_mw = base_load_p * load_mult[s]
        try:
            pp.runpp(
                work,
                init="flat",
                calculate_voltage_angles=False,
                verbose=False,
            )
            ok = bool(work.converged)
        except Exception:
            ok = False
        converged[s] = ok
        if ok:
            max_vm[s] = work.res_bus.vm_pu.max()
            max_loading[s] = (
                work.res_line.loading_percent.max() if has_lines else 0.0
            )
        else:
            max_vm[s] = np.inf
            max_loading[s] = np.inf

    violated = (max_vm > vm_max) | (max_loading > max_loading_pct)
    return {
        "max_vm": max_vm,
        "max_loading": max_loading,
        "converged": converged,
        "violated": violated,
        "n_scenarios": n,
        "n_failed": int((~converged).sum()),
    }
