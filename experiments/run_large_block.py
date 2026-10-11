#!/usr/bin/env python3
"""Independently calibrated large-block operational benchmarks."""
import argparse,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from qkd_monitor.large_block import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/large_block.json')
    parser.add_argument('--output',default='results_v2')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    if args.smoke:
        cfg.update(calibration_replicates=99,holdout_null_replicates=10,evaluation_replicates=10,pilot_replicates=5)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();qerr=channel_kernel_check()
    trace=build_trace(cfg)
    pd.DataFrame({k:v for k,v in trace.items() if isinstance(v,np.ndarray)}).to_csv(out/'synthetic_drift_trace.csv',index=False)
    alpha=cfg['family_alpha']/len(METHODS)
    nulls=['stationary','drift_replay','null_boundary']
    records=[];calrows=[];holdrows=[];pilots=[]
    # Adaptive policy calibration is shared by every scenario. Frozen before
    # evaluations. Equal per-method alpha and maximum over the same null strata.
    adaptive_cal=[]
    for ni,name in enumerate(nulls):
        m,e,tot,profile,tr=simulate_frames(name,cfg,trace,cfg['calibration_replicates'],cfg['seed']+100000+ni,adaptive=True)
        stats,_=statistics(m,e,cfg)
        values=stats[:,4];threshold=calibration_threshold(values,alpha)
        adaptive_cal.append((name,values,threshold))
        calrows.append(dict(benchmark='all',method='adaptive_mixture',null_stratum=name,threshold=threshold,n=len(values),alpha=alpha))
        print(f'Adaptive calibration {name}, threshold={threshold:.4f}',flush=True)
    adaptive_threshold=max(x[2] for x in adaptive_cal)
    np.savez_compressed(out/'adaptive_calibration.npz',**{x[0]:x[1] for x in adaptive_cal})
    for si,name in enumerate(SCENARIOS):
        # No detector thresholds are tuned with attack outcomes. The independent
        # pilot only determines a known-scenario fixed reveal-cost comparator.
        pm,pe,pt,_,_=simulate_frames(name,cfg,trace,cfg['pilot_replicates'],cfg['seed']+200000+si,adaptive=True)
        fixed_p=float(np.sum(pt['monitor_n'])/np.sum(pt['sifted']-pt['audit_n']))
        pilots.append(dict(scenario=name,fixed_probability=fixed_p,n=cfg['pilot_replicates']))
        calibrations=[];thresholds=np.zeros(5)
        for ni,null in enumerate(nulls):
            m,e,tot,profile,tr=simulate_frames(null,cfg,trace,cfg['calibration_replicates'],cfg['seed']+300000+100*si+ni,fixed_p=fixed_p)
            stats,_=statistics(m,e,cfg);calibrations.append((null,stats))
            for mi in range(5):
                th=calibration_threshold(stats[:,mi],alpha);thresholds[mi]=max(thresholds[mi],th)
                calrows.append(dict(benchmark=name,method=METHODS[mi],null_stratum=null,threshold=th,n=len(stats),alpha=alpha))
        np.savez_compressed(out/(name+'_fixed_calibration.npz'),**{x[0]:x[1] for x in calibrations})
        # Evaluation maxima and first crossings use thresholds frozen above.
        fixed=simulate_frames(name,cfg,trace,cfg['evaluation_replicates'],cfg['seed']+400000+si,fixed_p=fixed_p,capture=True)
        adaptive=simulate_frames(name,cfg,trace,cfg['evaluation_replicates'],cfg['seed']+400000+si,adaptive=True,capture=True)
        for is_adapt,batch in [(False,fixed),(True,adaptive)]:
            m,e,tot,profile,tr=batch
            stats,paths=statistics(m,e,cfg,with_paths=True)
            bound,actual=classical_audit(tot,cfg['audit_epsilon'],cfg['audit_probability'])
            pd.DataFrame(tr).to_csv(out/(name+('_adaptive_trace.csv' if is_adapt else '_fixed_trace.csv')),index=False)
            indices=[4] if is_adapt else range(5)
            for mi in indices:
                threshold=adaptive_threshold if is_adapt else thresholds[mi]
                method='adaptive_mixture' if is_adapt else METHODS[mi]
                cal=adaptive_cal if is_adapt else [(null,v[:,mi],0) for null,v in calibrations]
                pvals=np.max(np.column_stack([rank_pvalues(stats[:,mi],v) for _,v,_ in cal]),axis=1)
                for rep in range(cfg['evaluation_replicates']):
                    hits=np.flatnonzero(paths[rep,:,mi]>threshold)
                    first=int(hits[0]) if len(hits) else -1
                    event=np.flatnonzero(profile['active']);event_start=event[0] if len(event) else -1
                    after=first>=event_start and event_start>=0 and first>=0
                    during=first>=0 and bool(profile['active'][first])
                    records.append(dict(scenario=name,kind=profile['kind'],replicate=rep,method=method,seed=cfg['seed']+400000+si,
                       alert=first>=0,attack_executed=bool(tot['intercepted_total'][rep]>0),event_detection=during,eventual_detection=after,
                       pre_event_alert=first>=0 and event_start>=0 and first<event_start,
                       first_alert_s=(first+1)*cfg['frame_s'] if first>=0 else None,
                       delay_s=(first-event_start+1)*cfg['frame_s'] if after else None,
                       revealed_fraction=(tot['audit_n'][rep]+tot['monitor_n'][rep])/tot['sifted'][rep],
                       monitor_fraction=tot['monitor_n'][rep]/(tot['sifted'][rep]-tot['audit_n'][rep]),
                       sifted_rounds=int(tot['sifted'][rep]),remaining_rounds=int(tot['remaining_n'][rep]),
                       intercepted_remaining_rounds=int(tot['intercepted_remaining'][rep]),intercepted_total=int(tot['intercepted_total'][rep]),
                       bit_error_upper=bound[rep],actual_remaining_qber=actual[rep],classical_covered=bool(actual[rep]<=bound[rep]),
                       empirical_pvalue=pvals[rep],threshold=threshold,certified_secret_key_bits=None))
        # Independent family validation at each matched policy and null stratum.
        for ni,null in enumerate(nulls):
            fm,fe,_,_,_=simulate_frames(null,cfg,trace,cfg['holdout_null_replicates'],cfg['seed']+500000+100*si+ni,fixed_p=fixed_p)
            am,ae,_,_,_=simulate_frames(null,cfg,trace,cfg['holdout_null_replicates'],cfg['seed']+500000+100*si+ni,adaptive=True)
            fs,_=statistics(fm,fe,cfg);ads,_=statistics(am,ae,cfg)
            alerts=np.column_stack([fs>thresholds,ads[:,4]>adaptive_threshold])
            for mi,method in enumerate(METHODS):
                k=int(alerts[:,mi].sum());lo,hi=ci(k,len(alerts))
                holdrows.append(dict(benchmark=name,null_stratum=null,method=method,alerts=k,n=len(alerts),rate=k/len(alerts),ci_low=lo,ci_high=hi))
            k=int(np.any(alerts,axis=1).sum());lo,hi=ci(k,len(alerts))
            holdrows.append(dict(benchmark=name,null_stratum=null,method='family_any',alerts=k,n=len(alerts),rate=k/len(alerts),ci_low=lo,ci_high=hi))
        print(f'Finished {name}: fixed p={fixed_p:.4f}',flush=True)
        # Write incrementally to preserve expensive completed work.
        pd.DataFrame(records).to_csv(out/'trials.csv',index=False)
        pd.DataFrame(holdrows).to_csv(out/'null_holdout.csv',index=False)
        pd.DataFrame(calrows).to_csv(out/'calibration_thresholds.csv',index=False)
        pd.DataFrame(pilots).to_csv(out/'pilot_costs.csv',index=False)
    raw=pd.DataFrame(records);summary=[]
    for (scenario,kind,method),g in raw.groupby(['scenario','kind','method'],sort=False):
        row=dict(scenario=scenario,kind=kind,method=method,n=len(g))
        for column in ['alert','event_detection','eventual_detection','pre_event_alert','classical_covered','attack_executed']:
            k=int(g[column].sum());lo,hi=ci(k,len(g));prefix='attack_execution' if column=='attack_executed' else column;row.update({prefix+'_rate':k/len(g),prefix+'_ci_low':lo,prefix+'_ci_high':hi})
        for column in ['revealed_fraction','monitor_fraction','sifted_rounds','remaining_rounds','intercepted_remaining_rounds','intercepted_total','bit_error_upper','actual_remaining_qber']:
            row['mean_'+column]=float(g[column].mean())
        row['median_delay_s']=float(g.delay_s.median()) if g.delay_s.notna().any() else None
        row['delay_detection_count']=int(g.delay_s.notna().sum())
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    meta=dict(config=cfg,elapsed_seconds=time.perf_counter()-start,
              qiskit_marginal_max_discrepancy=qerr,trace_provenance=trace['provenance'],
              emitted_rounds=int(cfg['pulse_rate_hz']*cfg['duration_s']),
              calibration_rule='inclusive rank p, maximum over three strata; same alpha/6 for all six methods',
              security_status='NO QKD SECRECY PROOF; empirical guarantees under exchangeability within the registered null strata')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(f'Complete in {meta["elapsed_seconds"]:.1f}s',flush=True)

if __name__=='__main__':main()
