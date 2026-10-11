#!/usr/bin/env python3
"""Run from project root: python experiments/run_benchmarks.py --config configs/default.json"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
import qiskit
import qiskit_aer
from qkd_monitor.quantum import QiskitKernel, validate_aer
from qkd_monitor.channels import make_scenarios, simulate
from qkd_monitor.policies import monitor
from qkd_monitor.statistics import binomial_ci


def summarize(raw):
    rows = []
    for (scenario, kind, method), g in raw.groupby(["scenario", "kind", "method"], sort=False):
        row = dict(scenario=scenario, kind=kind, method=method, replicates=len(g))
        for label in ("alert", "in_event_detection", "eventual_detection", "pre_event_alarm", "classical_bound_covered"):
            hits = int(g[label].sum())
            lo, hi = binomial_ci(hits, len(g))
            row.update({label+"_rate":hits/len(g), label+"_ci_low":lo, label+"_ci_high":hi})
        for label in ("revealed_fraction", "monitor_fraction_of_eligible", "remaining_rounds", "idealized_yield_proxy_bits", "audit_qber", "remaining_bit_error_upper"):
            row["mean_"+label] = float(g[label].mean())
        delays = g.loc[g["eventual_detection"], "delay_emitted_rounds"].dropna()
        row["median_delay_conditional_on_detection"] = float(delays.median()) if len(delays) else None
        row["number_detected_for_delay"] = len(delays)
        # Restricted delay: misses/pre-event alerts get full remaining horizon penalty.
        # Conditional medians alone can misleadingly reward detectors that miss events.
        if kind in ("attack", "benign_anomaly"):
            penalties = np.where(g["eventual_detection"], g["delay_emitted_rounds"], g["n_emitted"]-g["event_start"])
            row["mean_restricted_delay"] = float(np.mean(penalties))
        else:
            row["mean_restricted_delay"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="results")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--skip-aer", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.replicates is not None:
        config["replicates"] = args.replicates
    if config["replicates"] < 1 or config["n_emitted"] < 1000:
        raise ValueError("Use positive replicates and at least 1000 emitted rounds")
    if config["matched_budget_pilot_replicates"] < 1:
        raise ValueError("Pilot replicates must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    start_clock = time.perf_counter()
    kernel = QiskitKernel()
    if not args.skip_aer:
        validation = pd.DataFrame(validate_aer(config["aer_validation_shots"], config["seed"]))
        validation.to_csv(out/"aer_validation.csv", index=False)
        if not validation.passed.all():
            raise RuntimeError("Aer validation failed")
        print(f"Aer: {len(validation)} settings, maximum error {validation.abs_error.max():.5f}", flush=True)
    scenarios = make_scenarios(config["n_emitted"], config["transmission"])
    all_rows, budgets = [], []
    for si, scenario in enumerate(scenarios):
        # Independent pilot provides a scenario-specific matched-cost comparator.
        # This is an oracle benchmark (scenario known), not a deployable policy.
        pilot_costs = []
        for j in range(config["matched_budget_pilot_replicates"]):
            base = config["seed"] + 1000000 + 1000*si + 10*j
            stream = simulate(scenario, kernel, base, config["readout_flip"])
            rows, _ = monitor(stream, config, base+1, adaptive=True)
            pilot_costs.append(rows[0]["monitor_fraction_of_eligible"])
        matched_p = float(np.mean(pilot_costs))
        budgets.append(dict(scenario=scenario.name, oracle_matched_probability=matched_p,
                            independent_pilot_replicates=len(pilot_costs)))
        for replicate in range(config["replicates"]):
            base = config["seed"] + si*10000 + replicate*10
            stream = simulate(scenario, kernel, base, config["readout_flip"])
            fixed, _ = monitor(stream, config, base+1)
            adaptive, trace = monitor(stream, config, base+1, adaptive=True,
                                       capture=(replicate == 0))
            matched, _ = monitor(stream, config, base+1, fixed_p=matched_p)
            for r in matched:
                r["method"] = "matched_"+r["method"]
            for r in fixed+adaptive+matched:
                r.update(scenario=scenario.name, kind=scenario.kind, replicate=replicate,
                         seed=base, n_emitted=config["n_emitted"],
                         event_start=stream["event_start"], event_end=stream["event_end"])
                all_rows.append(r)
            if trace is not None:
                pd.DataFrame(trace).to_csv(out/("trace_"+scenario.name+".csv"), index=False)
        print(f"Finished {scenario.name}: {config['replicates']} replicates, matched p={matched_p:.3f}", flush=True)
    raw = pd.DataFrame(all_rows)
    raw.to_csv(out/"trials.csv", index=False)
    summarize(raw).to_csv(out/"summary.csv", index=False)
    pd.DataFrame(budgets).to_csv(out/"pilot_budgets.csv", index=False)
    metadata = dict(config=config, python=platform.python_version(), numpy=np.__version__,
                    qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__,
                    elapsed_seconds=time.perf_counter()-start_clock,
                    quantum_backend="Qiskit DensityMatrix/Kraus exact transition kernel; Aer shot validation",
                    certified_secret_key_bits=None,
                    security_status="NO COMPOSABLE QKD SECURITY PROOF; classical bounds only")
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {len(raw)} rows in {metadata['elapsed_seconds']:.1f} seconds", flush=True)

if __name__ == "__main__":
    main()
