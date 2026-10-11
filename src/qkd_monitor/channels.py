"""Predeclared synthetic channels; independent loss is explicitly assumed."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Scenario:
    name: str
    kind: str
    q: np.ndarray
    intercept_fraction: np.ndarray
    loss: np.ndarray
    start: int = -1
    end: int = -1


def make_scenarios(n, transmission=.7):
    t = np.arange(n)
    stable = np.full(n, .02)
    bounded = .025 + .015*np.sin(2*np.pi*t/n*3)
    def sc(name, kind, q, fraction=0., start=-1, end=-1, loss=None):
        a = np.zeros(n)
        if start >= 0:
            a[start:end] = fraction
        return Scenario(name, kind, np.asarray(q), a,
                        np.full(n, transmission) if loss is None else loss, start, end)
    s, e = n//4, 3*n//8
    return [sc("stable", "null", stable),
            sc("bounded_drift", "null", bounded),
            sc("null_boundary", "null", np.full(n, (.05-.003)/(1-2*.003))),
            sc("benign_spike", "benign_anomaly", stable + .10*((t>=s)&(t<e)),
               start=s, end=e),
            sc("burst_strong", "attack", stable, .8, s, e),
            sc("burst_short", "attack", stable, .8, s, s+n//24),
            sc("burst_weak", "attack", stable, .25, s, e),
            sc("sustained", "attack", stable, .6, s, n),
            sc("bursts_repeated", "attack", stable,
               .65, s, e),
            sc("loss_drift", "null", bounded, loss=.55+.20*np.sin(2*np.pi*t/n)**2)]


def simulate(scenario, kernel, seed, readout_flip=.003):
    n = len(scenario.q)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n)
    a, b, e = (rng.integers(0, 2, n) for _ in range(3))
    attack_fraction = scenario.intercept_fraction.copy()
    if scenario.name == "bursts_repeated":
        # Three equal short bursts spanning the same overall event interval.
        attack_fraction[:] = 0
        width = n//32
        for s in (n//4, n//2, 3*n//4):
            attack_fraction[s:s+width] = .65
    attacked = rng.random(n) < attack_fraction
    p1 = kernel.probabilities(bits, a, b, e, attacked, scenario.q)
    bob = (rng.random(n) < p1).astype(int)
    bob ^= (rng.random(n) < readout_flip).astype(int)
    detected = rng.random(n) < scenario.loss
    sift = detected & (a == b)
    idx = np.flatnonzero(sift)
    return dict(time=idx, error=(bits[sift] != bob[sift]).astype(int),
                basis=a[sift], n_emitted=n,
                event_start=scenario.start,
                event_end=(n if scenario.name == "bursts_repeated" else scenario.end))
