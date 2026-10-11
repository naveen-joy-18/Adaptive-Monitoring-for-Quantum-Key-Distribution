"""Classical statistics only. No mapping from bit errors to phase errors."""
import numpy as np
from scipy.stats import beta


def binomial_ci(k, n, confidence=.95):
    a = (1-confidence)/2
    return (0. if k == 0 else float(beta.ppf(a, k, n-k+1)),
            1. if k == n else float(beta.ppf(1-a, k+1, n-k)))


def audit_remainder_bound(n, audit_count, audit_errors, monitor_count, monitor_errors, epsilon=.001):
    """Serfling one-sided bound for uniform audit, then subtract known errors.
    The audit sample is selected independently of the complete error vector,
    and is concealed from the adaptive monitor until the horizon ends.
    Validity is classical even when subsequent monitored removals adapt.
    """
    remaining = n-audit_count-monitor_count
    if remaining <= 0:
        return 0.
    if audit_count == 0 or audit_count >= n:
        return 1.
    mean = audit_errors/audit_count
    radius = np.sqrt((n-audit_count+1)*np.log(1/epsilon)/(2*audit_count*n))
    total_upper = min(float(n), n*(mean+radius))
    unseen_upper = max(0., total_upper-audit_errors-monitor_errors)
    return min(1., unseen_upper/remaining)


def binary_entropy(q):
    if q <= 0 or q >= 1:
        return 0.
    return -q*np.log2(q)-(1-q)*np.log2(1-q)


def idealized_yield_proxy(n_remaining, q_upper, alarm):
    """Hypothetical symmetric-BB84 entropy proxy; NOT a finite-key guarantee.
    Does not implement phase estimation, EC, PA or authentication.
    """
    if alarm is not None or q_upper >= .5:
        return 0.
    return float(n_remaining*max(0., 1-2*binary_entropy(q_upper)))
