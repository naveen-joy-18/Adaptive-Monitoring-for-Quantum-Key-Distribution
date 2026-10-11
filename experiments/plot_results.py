#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results")
args = parser.parse_args()
out = Path(args.results)
s = pd.read_csv(out/"summary.csv")
plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                     "svg.fonttype":"none", "figure.dpi":150})
methods = ["end_block", "disjoint_blocks", "sliding_window", "cusum_union", "fixed_mixture", "adaptive_mixture", "matched_fixed_mixture"]
labels = ["End block", "Disjoint blocks", "Sliding window", "CUSUM (union)", "Start mixture", "Adaptive mixture", "Cost-matched mixture"]
scenarios = ["burst_strong", "burst_short", "burst_weak", "sustained", "bursts_repeated", "benign_spike"]
fig, axes = plt.subplots(2, 3, figsize=(13,7), constrained_layout=True)
for ax, scenario in zip(axes.flat, scenarios):
    g = s[s.scenario==scenario].set_index("method").loc[methods]
    y = g.in_event_detection_rate.to_numpy()
    err = np.vstack([y-g.in_event_detection_ci_low, g.in_event_detection_ci_high-y])
    ax.bar(np.arange(len(methods)), y, color=["#9ba7b4"]*4+["#397fa4", "#c26741", "#6a9271"])
    ax.errorbar(np.arange(len(methods)), y, yerr=err, fmt="none", color="#273746", capsize=3)
    ax.set_xticks(np.arange(len(methods)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0,1.04)
    ax.set_title(scenario.replace("_", " "))
    ax.set_ylabel("Alert during event (95% exact CI)")
fig.savefig(out/"event_detection.png")
fig.savefig(out/"event_detection.svg")
plt.close(fig)
fig, axes = plt.subplots(1,2, figsize=(12,4), constrained_layout=True)
for method, label in zip(methods, labels):
    g=s[(s.method==method)&(s.kind=="null")]
    axes[0].plot(g.scenario, g.alert_rate, "o-", label=label)
axes[0].axhline(.01, ls="--", color="black", label="alpha=0.01")
axes[0].tick_params(axis="x", rotation=30)
axes[0].set_ylabel("Run-level alert fraction")
axes[0].set_title("Null channels (160 runs; small rates are uncertain)")
axes[0].legend(fontsize=7)
for method, label in zip(methods[-3:], labels[-3:]):
    g=s[(s.method==method)&s.scenario.isin(scenarios)]
    axes[1].plot(g.mean_revealed_fraction, g.in_event_detection_rate, "o", label=label)
    for _, r in g.iterrows():
        if method=="adaptive_mixture":
            axes[1].annotate(r.scenario, (r.mean_revealed_fraction,r.in_event_detection_rate),fontsize=7)
axes[1].set_xlabel("Mean audit + monitoring reveal fraction")
axes[1].set_ylabel("Alert during event")
axes[1].set_title("Detection–sampling trade-off")
axes[1].legend(fontsize=8)
fig.savefig(out/"tradeoffs.png")
fig.savefig(out/"tradeoffs.svg")
plt.close(fig)
trace = pd.read_csv(out/"trace_burst_strong.csv")
fig, axes=plt.subplots(3,1,figsize=(10,7),sharex=True,constrained_layout=True)
axes[0].plot(trace.time, trace.error.rolling(150, min_periods=150).mean(),lw=1)
axes[0].axhline(.05,ls="--",color="black")
axes[0].set_ylabel("Rolling true QBER\n(simulator only)")
axes[1].plot(trace.time,trace.monitor_probability,lw=.8,color="#c26741")
axes[1].set_ylabel("Monitor coin\nprobability")
axes[2].semilogy(trace.time,np.maximum(trace.e_value,1e-8),color="#397fa4")
axes[2].axhline(100,ls="--",color="black",label="Alarm threshold")
axes[2].set_ylabel("Mixture e-value")
axes[2].set_xlabel("Emitted-round index")
for ax in axes:
    ax.axvspan(3000,4500,alpha=.12,color="#c26741")
axes[2].legend()
fig.savefig(out/"example_trace.png")
fig.savefig(out/"example_trace.svg")
