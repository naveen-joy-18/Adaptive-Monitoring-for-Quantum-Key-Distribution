#!/usr/bin/env python3
"""Independent parameter sweep; does not retune main detector parameters."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from qkd_monitor.quantum import QiskitKernel
from qkd_monitor.channels import Scenario, simulate
from qkd_monitor.policies import monitor
from run_benchmarks import summarize

parser=argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.json")
parser.add_argument("--replicates",type=int,default=40)
parser.add_argument("--output",default="results")
args=parser.parse_args()
cfg=json.loads(Path(args.config).read_text())
if args.replicates<1: raise ValueError("replicates must be positive")
kernel=QiskitKernel(); n=cfg["n_emitted"]; rows=[]; case=0
for duration in (250,500,1500):
    for fraction in (.25,.50,.80):
        start=n//4
        a=np.zeros(n);a[start:start+duration]=fraction
        scene=Scenario(f"d{duration}_f{fraction:.2f}","attack",np.full(n,.02),a,
                       np.full(n,cfg["transmission"]),start,start+duration)
        for rep in range(args.replicates):
            seed=cfg["seed"]+2000000+case*10000+rep*10
            stream=simulate(scene,kernel,seed,cfg["readout_flip"])
            fixed,_=monitor(stream,cfg,seed+1)
            adaptive,_=monitor(stream,cfg,seed+1,adaptive=True)
            for row in fixed+adaptive:
                row.update(scenario=scene.name,kind=scene.kind,replicate=rep,seed=seed,
                           event_start=start,event_end=start+duration,n_emitted=n,
                           duration=duration,intercept_fraction=fraction)
                rows.append(row)
        print(f"Sweep {scene.name}: {args.replicates} replicates",flush=True)
        case+=1
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
raw=pd.DataFrame(rows)
raw.to_csv(out/"sweep_trials.csv",index=False)
summary=summarize(raw)
params=raw[["scenario","duration","intercept_fraction"]].drop_duplicates()
summary.merge(params,on="scenario").to_csv(out/"sweep_summary.csv",index=False)
(out/"sweep_metadata.json").write_text(json.dumps(dict(config=cfg,replicates=args.replicates,
                                                     seed_offset=2000000),indent=2))
