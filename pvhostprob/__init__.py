"""pvhost-prob: probabilistic (stochastic) PV hosting capacity for distribution feeders.

Deterministic hosting-capacity screening answers "how much PV can connect under
the worst-case snapshot". pvhost-prob answers the question planners actually ask:
"how much PV can connect while keeping the probability of a limit violation
below an accepted risk level?" — using Monte Carlo power flows with spatially
correlated PV scenarios on pandapower networks.
"""

from pvhostprob.scenarios import correlated_capacity_factors, load_multipliers
from pvhostprob.feeder import build_lv_feeder
from pvhostprob.simulate import run_scenarios
from pvhostprob.hc import (
    deterministic_hc,
    probabilistic_hc,
    locational_probabilistic_hc,
    exceedance_curve,
    violation_probability,
)
from pvhostprob.metrics import wilson_interval

__version__ = "0.1.0"
__all__ = [
    "correlated_capacity_factors",
    "load_multipliers",
    "build_lv_feeder",
    "run_scenarios",
    "deterministic_hc",
    "probabilistic_hc",
    "locational_probabilistic_hc",
    "exceedance_curve",
    "violation_probability",
    "wilson_interval",
]
