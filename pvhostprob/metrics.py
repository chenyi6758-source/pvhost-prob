"""Small statistics helpers: Wilson confidence intervals and run summaries."""

import numpy as np


def wilson_interval(k, n, confidence=0.95):
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of "successes" (here: violations).
    n : int
        Number of trials (scenarios).
    confidence : float
        0.95 -> z = 1.96.

    Returns
    -------
    (lo, hi) tuple, clipped to [0, 1].
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}[
        confidence
    ] if confidence in (0.90, 0.95, 0.99) else None
    if z is None:
        from scipy.stats import norm

        z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denom
    lo, hi = center - half, center + half
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return max(0.0, lo), min(1.0, hi)


def summarize_run(res, confidence=0.95):
    """One-line statistics dict for a :func:`run_scenarios` result."""
    k = int(res["violated"].sum())
    n = int(res["n_scenarios"])
    lo, hi = wilson_interval(k, n, confidence=confidence)
    ok = res["converged"]
    return {
        "n_scenarios": n,
        "n_failed": int(res["n_failed"]),
        "p_viol": k / n,
        "p_viol_ci95": (lo, hi),
        "max_vm_mean": float(res["max_vm"][ok].mean()) if ok.any() else float("nan"),
        "max_vm_p99": float(np.quantile(res["max_vm"][ok], 0.99))
        if ok.any()
        else float("nan"),
        "max_loading_mean": float(res["max_loading"][ok].mean())
        if ok.any()
        else float("nan"),
    }
