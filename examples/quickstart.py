"""One-command demo: probabilistic hosting capacity on a synthetic LV feeder.

Runs in ~8 minutes on a laptop (Monte Carlo AC power flows dominate).
Outputs: examples/exceedance.png, examples/locational.png, examples/report.csv
"""

import os
import time

import numpy as np
import pandas as pd

from pvhostprob import (
    build_lv_feeder,
    correlated_capacity_factors,
    load_multipliers,
    deterministic_hc,
    probabilistic_hc,
    locational_probabilistic_hc,
    exceedance_curve,
)
from pvhostprob.plot import plot_exceedance, plot_locational

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
PROGRESS = os.path.join(OUT, "progress.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")

RISK_LEVELS = [0.01, 0.05, 0.10]
VM_MAX = 1.05


def main():
    t0 = time.time()
    net = build_lv_feeder(n_buses=12)
    log(f"[1/6] built 12-bus LV feeder "
          f"({time.time() - t0:.1f}s)")

    # PV candidates spread along the feeder; correlated irradiance scenarios
    # plus modest load uncertainty.
    candidates = [2, 4, 6, 8, 10, 11]
    n_scen = 400
    cf = correlated_capacity_factors(n_scen, len(candidates), corr=0.85, seed=0)
    lm = load_multipliers(n_scen, len(net.load), sigma=0.10, seed=1)
    log(f"[2/6] drew {n_scen} correlated PV scenarios "
          f"(corr=0.85) + load uncertainty (sigma=0.10)")

    det = deterministic_hc(net, candidates, vm_max=VM_MAX)
    log(f"[3/6] deterministic HC: {det['hc_kw_per_bus']:.2f} kW/bus "
          f"({det['hc_kw_total']:.1f} kW total), binds: "
          f"{det['binding']['binding']} at {det['binding']['location']} "
          f"({time.time() - t0:.0f}s)")

    prob = {}
    for risk in RISK_LEVELS:
        r = probabilistic_hc(net, candidates, cf, load_mult=lm, risk=risk,
                             vm_max=VM_MAX, tol_kw=1.0)
        prob[risk] = r
        lo, hi = r["p_viol_ci95"]
        log(f"[4/6] probabilistic HC at {risk:.0%} risk: "
              f"{r['hc_kw_per_bus']:.2f} kW/bus "
              f"(P_viol={r['p_viol']:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]) "
              f"({time.time() - t0:.0f}s)")

    loc = locational_probabilistic_hc(
        net, [1, 5, 9, 11], cf[:, [0]], load_mult=lm, risk=0.05,
        vm_max=VM_MAX, tol_kw=1.0,
    )
    log(f"[5/6] locational HC @5% risk:\n"
          f"{loc[['bus_name', 'hc_kw', 'binding']].to_string(index=False)} "
          f"({time.time() - t0:.0f}s)")

    caps = np.linspace(2, 34, 9)
    curve = exceedance_curve(net, candidates, caps, cf, load_mult=lm,
                             vm_max=VM_MAX)
    markers = {"deterministic HC": det["hc_kw_per_bus"]}
    markers.update({f"{r:.0%} risk HC": prob[r]["hc_kw_per_bus"]
                    for r in RISK_LEVELS})
    plot_exceedance(curve, risk=0.05, markers=markers,
                    title="Violation probability vs installed PV (12-bus LV feeder)",
                    path=os.path.join(OUT, "exceedance.png"))
    plot_locational(loc, risk=0.05,
                    title="Locational probabilistic HC @5% risk (12-bus LV feeder)",
                    path=os.path.join(OUT, "locational.png"))
    log(f"[6/6] figures saved to {OUT}/")

    report = pd.DataFrame([
        {"metric": "deterministic_hc_kw_per_bus",
         "value": round(det["hc_kw_per_bus"], 2)},
        {"metric": "deterministic_hc_kw_total",
         "value": round(det["hc_kw_total"], 2)},
        *[{"metric": f"probabilistic_hc_kw_per_bus_risk_{r:.0%}",
            "value": round(prob[r]["hc_kw_per_bus"], 2)} for r in RISK_LEVELS],
        *[{"metric": f"locational_hc_kw_{row.bus_name}",
            "value": round(row.hc_kw, 2)} for row in loc.itertuples()],
    ])
    report.to_csv(os.path.join(OUT, "report.csv"), index=False)

    base = det["hc_kw_per_bus"]
    log("\n=== summary ===")
    log(f"deterministic (worst-case) HC : {base:.2f} kW/bus")
    for r in RISK_LEVELS:
        lift = (prob[r]["hc_kw_per_bus"] / base - 1.0) * 100
        log(f"probabilistic HC @ {r:.0%} risk   : "
              f"{prob[r]['hc_kw_per_bus']:.2f} kW/bus ({lift:+.1f}% vs deterministic)")
    log(f"far-end bus {loc['bus_name'].iloc[-1]} hosts "
          f"{loc['hc_kw'].iloc[-1]:.1f} kW vs "
          f"{loc['hc_kw'].iloc[0]:.1f} kW next to the substation")
    log(f"total wall time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
