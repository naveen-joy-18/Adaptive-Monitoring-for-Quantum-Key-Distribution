"""Each method controls its own run-level alpha; no combined-alert guarantee.
Null: conditional error probability of each revealed monitoring bit <= q0.
Alpha concerns channel anomalies, not Eve's information or QKD secrecy.
"""
import numpy as np
from scipy.stats import binom

class StartMixture:
    """Finite-horizon mixture over start times and fixed Bernoulli alternatives.
    Dormant likelihood-ratio components remain at one until activation.
    Restarting a detector without allocating new alpha is intentionally absent.
    """
    def __init__(self, max_observations, q0=.05, alpha=.01, stride=25,
                 alternatives=(.10, .18, .28)):
        if max_observations < 1 or stride < 1 or not 0 < alpha < 1:
            raise ValueError("Invalid horizon, stride or alpha")
        self.q0, self.alpha, self.stride = q0, alpha, stride
        self.alternatives = np.array(alternatives)
        if not (0 < q0 < 1 and np.all((self.alternatives>q0)&(self.alternatives<1))):
            raise ValueError("Alternatives must be between q0 and one")
        self.k = (max_observations-1)//stride+1
        self.max_observations = max_observations
        self.lr = np.ones((self.k, len(alternatives)))
        self.active, self.n = 0, 0
        self.value, self.max_value = 1., 1.
        self.first_alarm = None
        self.factor1 = self.alternatives/q0
        self.factor0 = (1-self.alternatives)/(1-q0)

    def update(self, x, time):
        if self.n >= self.max_observations:
            raise ValueError("Predeclared monitoring horizon exceeded")
        if self.n % self.stride == 0:
            self.active += 1
        self.lr[:self.active] *= self.factor1 if x else self.factor0
        # Upper clipping only reduces evidence and avoids overflow far beyond alarm.
        np.minimum(self.lr[:self.active], 1e250, out=self.lr[:self.active])
        self.n += 1
        active = np.sum(self.lr[:self.active])/(self.k*len(self.alternatives))
        self.value = (self.k-self.active)/self.k + active
        self.max_value = max(self.max_value, self.value)
        if self.first_alarm is None and self.value >= 1/self.alpha:
            self.first_alarm = int(time)
        return self.value


def batch_baselines(x, times, max_observations, q0=.05, alpha=.01, window=100, block=250):
    """Binomial tails remain conservative for conditional probabilities <= q0.
    Counts and look times here are outcome independent (fixed sampling only).
    """
    n = len(x)
    end = max_observations-1 if n else None
    fixed = end if n and binom.sf(int(np.sum(x))-1, n, q0) <= alpha else None
    blocks = (max_observations+block-1)//block
    block_alarm = None
    for start in range(0, n, block):
        stop = min(n, start+block)
        if binom.sf(int(np.sum(x[start:stop]))-1, stop-start, q0) <= alpha/blocks:
            block_alarm = int(times[stop-1]) if stop-start == block else max_observations-1
            break
    sliding_alarm = None
    if n >= window:
        counts = np.convolve(x, np.ones(window, dtype=int), mode="valid")
        tails = binom.sf(counts-1, window, q0)
        look_budget = max(1, max_observations-window+1)
        hit = np.flatnonzero(tails <= alpha/look_budget)
        if len(hit):
            sliding_alarm = int(times[hit[0]+window-1])
    # All-start Page CUSUM, with conservative union-of-starts Ville threshold.
    q1 = .18
    increment = np.where(x, np.log(q1/q0), np.log((1-q1)/(1-q0)))
    wealth, cusum_alarm = 0., None
    threshold = np.log(max_observations/alpha)
    for value, timestamp in zip(increment, times):
        wealth = max(0., wealth+value)
        if wealth >= threshold:
            cusum_alarm = int(timestamp)
            break
    return dict(end_block=fixed, disjoint_blocks=block_alarm,
                sliding_window=sliding_alarm, cusum_union=cusum_alarm)
