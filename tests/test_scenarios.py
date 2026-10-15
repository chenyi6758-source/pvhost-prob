"""Tests for correlated scenario generation."""

import numpy as np
import pytest
from scipy import stats

from pvhostprob.scenarios import (
    beta_params,
    correlated_capacity_factors,
    load_multipliers,
)


def test_beta_params_moments():
    a, b = beta_params(0.55, 0.25)
    # method of moments: recovered mean/std must match inputs
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    assert mean == pytest.approx(0.55, abs=1e-9)
    assert np.sqrt(var) == pytest.approx(0.25, abs=1e-9)


def test_beta_params_infeasible():
    with pytest.raises(ValueError):
        beta_params(0.5, 0.6)  # std^2 > mean*(1-mean)
    with pytest.raises(ValueError):
        beta_params(1.5, 0.1)


def test_cf_shape_bounds_reproducible():
    cf1 = correlated_capacity_factors(200, 4, seed=7)
    cf2 = correlated_capacity_factors(200, 4, seed=7)
    assert cf1.shape == (200, 4)
    assert np.all((cf1 >= 0.0) & (cf1 <= 1.0))
    assert np.array_equal(cf1, cf2)
    cf3 = correlated_capacity_factors(200, 4, seed=8)
    assert not np.array_equal(cf1, cf3)


def test_cf_marginals_match_targets():
    cf = correlated_capacity_factors(8000, 3, mean=0.55, std=0.25, corr=0.8, seed=0)
    assert cf.mean() == pytest.approx(0.55, abs=0.02)
    assert cf.std() == pytest.approx(0.25, abs=0.02)


def test_cf_spatial_correlation():
    # Gaussian copula: Spearman rho ~= (6/pi) * arcsin(corr/2)
    corr = 0.85
    cf = correlated_capacity_factors(4000, 3, corr=corr, seed=1)
    expected = 6.0 / np.pi * np.arcsin(corr / 2.0)
    got, _ = stats.spearmanr(cf[:, 0], cf[:, 1])
    assert got == pytest.approx(expected, abs=0.06)


def test_cf_independent_when_corr_zero():
    cf = correlated_capacity_factors(4000, 2, corr=0.0, seed=2)
    got, _ = stats.spearmanr(cf[:, 0], cf[:, 1])
    assert abs(got) < 0.08


def test_cf_perfect_correlation():
    cf = correlated_capacity_factors(500, 3, corr=1.0, seed=3)
    assert np.allclose(cf[:, 0], cf[:, 1])
    assert np.allclose(cf[:, 1], cf[:, 2])


def test_load_multipliers():
    lm = load_multipliers(300, 5, sigma=0.1, seed=0)
    assert lm.shape == (300, 5)
    assert np.all(lm > 0)
    assert lm.mean() == pytest.approx(1.0, abs=0.05)
    assert np.array_equal(lm, load_multipliers(300, 5, sigma=0.1, seed=0))
    assert np.all(load_multipliers(50, 4, sigma=0.0) == 1.0)
