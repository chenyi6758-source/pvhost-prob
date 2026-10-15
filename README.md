# pvhost-prob

Probabilistic (stochastic) PV hosting capacity for distribution feeders, built on pandapower.

Deterministic hosting-capacity screening answers "how much PV can connect under the worst-case snapshot". pvhost-prob answers the question planners actually ask: "how much PV can connect while keeping the probability of a limit violation below an accepted risk level?" — using Monte Carlo power flows with spatially correlated PV scenarios on pandapower networks.
