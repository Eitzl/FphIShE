"""Monte Carlo nulls, permutation tests, bootstrap CIs, run-time regression."""

import numpy as np


def bootstrap_ci(x, n=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = rng.choice(x, (n, len(x))).mean(axis=1)
    return np.quantile(means, [alpha / 2, 1 - alpha / 2])


def paired_permutation_test(a, b, n=10000, seed=0):
    """Two-sided p-value for mean(a - b) == 0 by random sign flips."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    null = (rng.choice([-1, 1], (n, len(d))) * d).mean(axis=1)
    return (np.abs(null) >= abs(d.mean())).mean()
