"""
pelt.py: a simple implementation of the PELT algorithm.
"""

import numpy as np
import scipy.stats as stats

# Define some helper functions
def grab_ss(counts, t1, t2):
    """
    Given start and end times, get sufficient statistics for data in the
    closed interval [t1, t2] from counts.
    """
    this_counts = counts[t1:(t2 + 1)]
    return sum(this_counts), len(this_counts)

def base_LL(counts, t1, t2, lam, dt):
    """
    For the closed interval [t1, t2], calculate the log likelihood of the
    data in counts.
    """
    c = counts[t1:(t2 + 1)]
    return np.sum(stats.poisson.logpmf(c, lam * dt))

def kappa(N, ell, lam, nu, dt):
    """
    Calculate the differential log likelihood for assigning count data to
    state 1 vs state 0.
    """
    return N * np.log(nu) - lam * dt * (nu - 1) * ell

def C(counts, t1, t2, lam, nu, dt):
    """
    Calculate the cost function for data in the closed interval [t1, t2].
    """
    N, ell = grab_ss(counts, t1, t2)
    kap = kappa(N, ell, lam, nu, dt)
    return -(base_LL(counts, t1, t2, lam, dt) + kap + np.logaddexp(0, -kap))

def find_changepoints(counts, lam, nu, dt, beta):
    """
    Given the data in counts, baseline event rate lam, state 1 event rate
    nu * lam, binsize dt, and changepoint penalty (likelihood ratio) beta,
    find changepoints in the data. Points mark the beginning of a new section,
    (meaning: the changepoint bin is *included*).
    """
    # allocate containers
    T = len(counts)  # number of time points
    F = np.empty(T + 1)  # F(t) = minimum cost for all data up to time t
    R = set({})  # set of times over which to search
    CP = set({})  # list of changepoints (ordered)
    K = 0.0  # bound constant for objective gain on changepoint insertion

    # initialize
    R.add(0)
    F[0] = -beta

    # iterate
    for tau in xrange(1, T + 1):
        mincost = np.inf
        new_tau = 0
        for t in R:
            cost = F[t] + C(counts, t + 1, tau, lam, nu, dt) + beta
            if cost < mincost:
                mincost = cost
                new_tau = t

        F[tau] = mincost

        CP.add(new_tau)
        R = ({r for r in R if F[r] + C(counts, r + 1, tau, lam, nu, dt) +
            K < F[tau]})
        R.add(tau)

    # extract changepoints
    return sorted(list(CP))

def calc_state_probs(counts, cplist, lam, nu, dt):
    """
    Given a sorted iterable of changepoints (bin locations) and a maximum
    number of bins, return an array with each bin containing the probability
    of being in state 1.
    """
    Ncp = len(cplist)
    T = len(counts)
    inferred = np.empty(T)

    for tau in xrange(Ncp):
        run_start = cplist[tau] + 1
        if tau == Ncp - 1:
            run_end = T
        else:
            run_end = cplist[tau + 1]

        N, ell = grab_ss(counts, run_start, run_end)
        kap = kappa(N, ell, lam, nu, dt)
        Ez = stats.logistic.cdf(kap)
        inferred[run_start:(run_end + 1)] = Ez

    return inferred