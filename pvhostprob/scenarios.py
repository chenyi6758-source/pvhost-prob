"""Correlated PV capacity-factor scenarios and load-uncertainty scenarios.

PV output across nearby buses is not independent: one cloud shades the whole
street. Sampling each bus independently therefore understates coincident
high-output events — exactly the events that drive overvoltage violations.
This module generates capacity-factor scenarios whose *marginals* follow a
Beta distribution (the standard choice for clear-sky-index-like quantities on
[0, 1]) while their *rank dependence* follows a Gaussian copula with a tunable
spatial correlation. Correlated Monte Carlo sampling for hosting-capacity work
is the established stochastic approach (EPRI's stochastic HC methodology,
2012; recent Monte Carlo HC studies e.g. Masic et al., STET 2025).
"""

import numpy as np
from scipy import stats


def beta_params(mean, std):
    """Method-of-moments Beta(a, b) parameters for a given mean and std.

    Raises ValueError if the requested variance is infeasible for a Beta
    distribution (var must be < mean * (1 - mean)).
    """
    mean = float(mean)
    std = float(std)
    if not 0.0 < mean < 1.0:
        raise ValueError(f"mean must be in (0, 1), got {mean}")
    var = std**2
    vmax = mean * (1.0 - mean)
    if not 0.0 < var < vmax:
        raise ValueError(
            f"infeasible (mean, std): need 0 < std^2 < mean*(1-mean) = {vmax:.4f}, "
            f"got std^2 = {var:.4f}"
        )
    scale = vmax / var - 1.0
    return mean * scale, (1.0 - mean) * scale


def correlated_capacity_factors(
    n_scenarios, n_sites, mean=0.55, std=0.25, corr=0.85, seed=0
):
    """Draw (n_scenarios, n_sites) PV capacity factors in [0, 1].

    Each site's marginal distribution is Beta(mean, std); dependence across
    sites comes from a Gaussian copula with uniform pairwise correlation
    ``corr``. ``corr = 1`` gives perfectly co-moving sites, ``corr = 0``
    independent sites.

    Parameters
    ----------
    n_scenarios, n_sites : int
    mean, std : float
        Target marginal mean / std of the capacity factor.
    corr : float
        Target Pearson correlation of the underlying Gaussian copula
        (rank correlation of the output is ~= (6/pi)*arcsin(corr/2)).
    seed : int
        Reproducibility seed (numpy Generator).

    Returns
    -------
    numpy.ndarray, shape (n_scenarios, n_sites), values in [0, 1].
    """
    if n_scenarios < 1 or n_sites < 1:
        raise ValueError("n_scenarios and n_sites must be >= 1")
    if not 0.0 <= corr <= 1.0:
        raise ValueError(f"corr must be in [0, 1], got {corr}")
    a, b = beta_params(mean, std)
    rng = np.random.default_rng(seed)

    # Gaussian copula: Z ~ N(0, R), U = Phi(Z), cf = Beta_ppf(U).
    r_matrix = np.full((n_sites, n_sites), corr)
    np.fill_diagonal(r_matrix, 1.0)
    try:
        chol = np.linalg.cholesky(r_matrix)
    except np.linalg.LinAlgError:  # corr == 1.0 exactly: rank-deficient
        chol = np.ones((n_sites, 1)) / np.sqrt(1.0)
        z = rng.standard_normal((n_scenarios, 1)) @ chol.T
    else:
        z = rng.standard_normal((n_scenarios, n_sites)) @ chol.T
    u = stats.norm.cdf(z)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return stats.beta.ppf(u, a, b)


def load_multipliers(n_scenarios, n_loads, sigma=0.10, seed=1):
    """Lognormal load multipliers with mean ~1 (median 1) for load uncertainty.

    Returns shape (n_scenarios, n_loads); multiply base load p_mw by these.
    ``sigma = 0`` returns all ones.
    """
    if n_scenarios < 1 or n_loads < 1:
        raise ValueError("n_scenarios and n_loads must be >= 1")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    rng = np.random.default_rng(seed)
    if sigma == 0:
        return np.ones((n_scenarios, n_loads))
    return rng.lognormal(mean=0.0, sigma=sigma, size=(n_scenarios, n_loads))
