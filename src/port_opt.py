
"""Mean-variance portfolio optimization, following Markowitz (1952)."""

import numpy as np


def portfolio_mean(w, mu):
    """Calculate expected portfolio return."""
    return w @ mu


def portfolio_variance(w, Sigma):
    """Calculate portfolio return variance."""
    return w @ Sigma @ w


def find_global_minimum_variance_portfolio(Sigma):
    n = Sigma.shape[0]
    ones = np.ones(n)
    x = np.linalg.solve(Sigma, ones)
    wstar = x / x.sum()
    return wstar


def find_tangency_portfolio(mu, Sigma, rf):
    n = Sigma.shape[0]
    ones = np.ones(n)
    excess_returns = mu - rf * ones
    x = np.linalg.solve(Sigma, excess_returns)
    wstar = x / x.sum()
    return wstar