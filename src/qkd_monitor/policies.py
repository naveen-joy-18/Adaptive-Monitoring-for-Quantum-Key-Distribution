"""Predictable monitoring coin; fixed independent audit revealed at the end."""
import numpy as np
from .detectors import StartMixture, batch_baselines
from .statistics import audit_remainder_bound, idealized_yield_proxy


def monitor(stream, config, seed, adaptive=False, fixed_p=None, capture=False):
    x, time = stream["error"], stream["time"]
    n = len(x)
    rng = np.random.default_rng(seed)
    # Preselected, outcome-independent uniform audit. Never feeds adaptation.
    audit = rng.random(n) < config["audit_probability"]
    coins = rng.random(n)
    detector = StartMixture(stream["n_emitted"], config["q0"], config["alpha"],
                            config["stride"], config["alternatives"])
    p_low = config["monitor_probability"] if fixed_p is None else fixed_p
    p_high = config["adaptive_high_probability"]
    revealed = np.zeros(n, dtype=bool)
    chosen_p = np.zeros(n)
    evidence = np.ones(n)
    # Suspicion changes NEXT eligible round's reveal probability, never its own.
    ewma, warmup = config["q0"]/2, 0
    for i in range(n):
        p = p_low
        if adaptive and warmup >= config["warmup_tests"] and ewma > config["suspicion_threshold"]:
            p = p_high
        chosen_p[i] = p
        if not audit[i] and coins[i] < p:
            revealed[i] = True
            detector.update(int(x[i]), int(time[i]))
            ewma = (1-config["ewma_rate"])*ewma + config["ewma_rate"]*x[i]
            warmup += 1
        evidence[i] = detector.value
    k = int(np.sum(audit))
    m = int(np.sum(revealed))
    ae, me = int(np.sum(x[audit])), int(np.sum(x[revealed]))
    remaining = ~(audit | revealed)
    bound = audit_remainder_bound(n, k, ae, m, me, config["audit_epsilon"])
    actual = float(np.mean(x[remaining])) if np.any(remaining) else 0.
    alarms = {"adaptive_mixture" if adaptive else "fixed_mixture": detector.first_alarm}
    if not adaptive:
        alarms.update(batch_baselines(x[revealed], time[revealed], stream["n_emitted"],
                                      config["q0"], config["alpha"],
                                      config["window"], config["block"]))
    records = []
    for method, alarm in alarms.items():
        start, end = stream["event_start"], stream["event_end"]
        in_event = alarm is not None and start <= alarm < end and start >= 0
        after_start = alarm is not None and start >= 0 and alarm >= start
        records.append(dict(method=method, alarm_time=alarm,
                            alert=alarm is not None, in_event_detection=in_event,
                            eventual_detection=after_start,
                            pre_event_alarm=alarm is not None and start>=0 and alarm<start,
                            delay_emitted_rounds=(alarm-start if after_start else None),
                            sifted_rounds=n, audit_count=k, monitor_count=m,
                            revealed_fraction=(k+m)/n if n else 0.,
                            monitor_fraction_of_eligible=m/(n-k) if n>k else 0.,
                            remaining_rounds=int(np.sum(remaining)),
                            audit_qber=ae/k if k else None,
                            remaining_bit_error_upper=bound, actual_remaining_qber=actual,
                            classical_bound_covered=actual <= bound+1e-12,
                            idealized_yield_proxy_bits=idealized_yield_proxy(int(np.sum(remaining)), bound, alarm),
                            certified_secret_key_bits=None))
    trace = None
    if capture:
        trace = dict(time=time, error=x, audit=audit, monitor=revealed,
                     monitor_probability=chosen_p, e_value=evidence)
    return records, trace
