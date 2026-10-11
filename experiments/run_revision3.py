#!/usr/bin/env python3
"""Revision 3: no threshold fitting on attack outcomes; full archived banks reused."""
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qkd_monitor.large_block import build_trace,simulate_frames,statistics,ci,channel_kernel_check
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'configs/large_block.json').read_text())
plan=json.loads((ROOT/'configs/revision3.json').read_text())
out=ROOT/'results_v3';out.mkdir(exist_ok=True)
trace=build_trace(cfg)
cal=pd.read_csv(ROOT/'calibration/calibration_thresholds.csv')
pil=pd.read_csv(ROOT/'calibration/pilot_costs.csv')
fixed_p=float(pil.loc[pil.scenario=='burst_50ms','fixed_probability'].iloc[0])
alpha=cfg['family_alpha']/6;analytic=float(np.log(1/alpha))
adth=float(cal.loc[cal.method=='adaptive_mixture','threshold'].max())
fxth=float(cal.loc[(cal.benchmark=='burst_50ms')&(cal.method=='fixed_mixture'),'threshold'].max())
thresholds={'adaptive':adth,'fixed':fxth}
start=time.perf_counter();records=[];summary=[]

def batch(label,policy,settings,reps,seed,phase='map',fp=fixed_p):
    c={**cfg,**settings}
    m,e,tot,profile,rows=simulate_frames('burst_custom',c,trace,reps,seed,adaptive=policy=='adaptive',fixed_p=fp,capture=True)
    _,paths=statistics(m,e,c,with_paths=True);path=paths[:,:,4]
    active=profile['active'];onset=float(c['burst_onset_s']);duration=float(c['burst_duration_s'])
    bn=m[:,active].sum(axis=1);sn=[]
    # Total burst sifted counts are available for representative trace only;
    # aggregate expected testing mass uses the declared gain and observed m.
    expected_sifted=float(np.sum(c['pulse_rate_hz']*c['frame_s']*profile['gain'][active]*c['basis_match_probability']))
    for mode,threshold in [('empirical',thresholds[policy]),('analytic',analytic)]:
        above=path>threshold if mode=='empirical' else path>=threshold
        anyhit=above.any(axis=1);first=np.where(anyhit,above.argmax(axis=1),-1)
        inside=(first>=0)&active[np.maximum(first,0)]
        pre=(first>=0)&(trace['time_s'][np.maximum(first,0)]<onset)
        for r in range(reps):
            records.append(dict(phase=phase,label=label,policy=policy,threshold_mode=mode,replicate=r,onset_s=onset,duration_ms=round(duration*1000),fraction=c['burst_fraction'],event_detection=bool(inside[r]),alert=bool(anyhit[r]),pre_event_alert=bool(pre[r]),attack_executed=bool(tot['intercepted_total'][r]>0),burst_monitor_rounds=int(bn[r]),intercepted_total=int(tot['intercepted_total'][r]),revealed_fraction=float((tot['audit_n'][r]+tot['monitor_n'][r])/tot['sifted'][r]),monitor_probability=float(tot['monitor_n'][r]/max(tot['sifted'][r]-tot['audit_n'][r],1)),delay_s=float((first[r]+1)*c['frame_s']-onset) if inside[r] else np.nan))
        k=int(inside.sum());lo,hi=ci(k,reps)
        sn.append(dict(phase=phase,label=label,policy=policy,threshold_mode=mode,onset_s=onset,duration_ms=round(duration*1000),fraction=c['burst_fraction'],n=reps,event_detection_rate=k/reps,ci_low=lo,ci_high=hi,alert_rate=float(anyhit.mean()),pre_event_alert_rate=float(pre.mean()),attack_execution_rate=float(np.mean(tot['intercepted_total']>0)),mean_burst_monitor_rounds=float(bn.mean()),expected_burst_sifted=expected_sifted,mean_intercepted_total=float(tot['intercepted_total'].mean()),mean_revealed_fraction=float(np.mean((tot['audit_n']+tot['monitor_n'])/tot['sifted'])),mean_monitor_probability=float(np.mean(tot['monitor_n']/np.maximum(tot['sifted']-tot['audit_n'],1))),median_delay_s=float(np.median((first[inside]+1)*c['frame_s']-onset)) if k else np.nan))
    summary.extend(sn)
    return sn,rows

# Re-evaluate the exact old streams with their original cost-matched fixed policies.
old_scenarios=['burst_200ms','burst_50ms','weak_burst','benign_excursion']
original=['stationary','drift_replay','null_boundary','benign_excursion','burst_200ms','burst_50ms','weak_burst','policy_aware_aggressive','policy_aware_subthreshold']
replay=[]
for name in old_scenarios:
    fp=float(pil.loc[pil.scenario==name,'fixed_probability'].iloc[0])
    fth=float(cal.loc[(cal.benchmark==name)&(cal.method=='fixed_mixture'),'threshold'].max())
    for policy in ['fixed','adaptive']:
        m,e,tot,prof,_=simulate_frames(name,cfg,trace,400,cfg['seed']+400000+original.index(name),adaptive=policy=='adaptive',fixed_p=fp)
        _,paths=statistics(m,e,cfg,with_paths=True);path=paths[:,:,4]
        for mode,th in [('empirical',adth if policy=='adaptive' else fth),('analytic',analytic)]:
            hit=path>th if mode=='empirical' else path>=th
            first=np.where(hit.any(axis=1),hit.argmax(axis=1),-1)
            inside=(first>=0)&prof['active'][np.maximum(first,0)]
            k=int(inside.sum());lo,hi=ci(k,400)
            replay.append(dict(scenario=name,policy=policy,threshold_mode=mode,log_threshold=th,n=400,k=k,event_detection_rate=k/400,ci_low=lo,ci_high=hi))
pd.DataFrame(replay).to_csv(out/'analytic_empirical_replay.csv',index=False)
print('Threshold replay complete',flush=True)

for di,d in enumerate(plan['durations_ms']):
    for fi,f in enumerate(plan['fractions']):
        settings=dict(burst_duration_s=d/1000,burst_fraction=f,burst_onset_s=plan['map_onset_s'])
        for policy in ['fixed','adaptive']:
            batch(f'd{d}_f{f}',policy,settings,plan['map_replicates'],plan['seed']+10000+100*di+fi)
    print(f'Map duration {d} ms complete',flush=True)

# A strategic attacker optimizes a committed burst onset independently against
# each defender. No sampling-probability pause trigger; no interception budget cap.
# Training and evaluation randomness are disjoint. Attack duration and fraction
# are identical for both defenders; timing-induced photon dose is reported.
training=[];chosen={};traces=[]
for policy in ['fixed','adaptive']:
    candidates=[]
    for oi,onset in enumerate(plan['attacker_onsets_s']):
        settings=dict(burst_duration_s=plan['attacker_duration_ms']/1000,burst_fraction=plan['attacker_fraction'],burst_onset_s=onset)
        rows,_=batch(f'train_{onset}',policy,settings,plan['attacker_training_replicates'],plan['seed']+20000+oi,phase='attacker_training')
        r=rows[0];candidates.append(r);training.append(r)
    # Lexicographic objective: empirical in-event alert rate, then testing mass,
    # then onset. This is a finite candidate optimizer, not a global optimum.
    best=min(candidates,key=lambda r:(r['event_detection_rate'],r['mean_burst_monitor_rounds'],r['onset_s']))
    chosen[policy]=best['onset_s']
# Cross-evaluate both selected onsets against both policies; this exposes ambient
# timing confounding instead of silently comparing different histories.
for oi,onset in enumerate(sorted(set(chosen.values()))):
    for policy in ['fixed','adaptive']:
        settings=dict(burst_duration_s=plan['attacker_duration_ms']/1000,burst_fraction=plan['attacker_fraction'],burst_onset_s=onset)
        _,rows=batch(f'selected_{onset}',policy,settings,plan['attacker_evaluation_replicates'],plan['seed']+30000+oi,phase='attacker_holdout')
        pd.DataFrame(rows).to_csv(out/f'attacker_{policy}_onset{onset}_trace.csv',index=False)
pd.DataFrame(training).to_csv(out/'attacker_training.csv',index=False)
pd.DataFrame(records).to_csv(out/'trials.csv',index=False)
pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
pd.DataFrame([dict(policy=p,empirical_log_threshold=t,analytic_log_threshold=analytic,e_threshold=np.exp(analytic),alpha=alpha) for p,t in thresholds.items()]).to_csv(out/'thresholds.csv',index=False)
metadata=dict(plan=plan,fixed_monitor_probability=fixed_p,thresholds=thresholds,analytic_log_threshold=analytic,per_method_alpha=alpha,attacker_selected_onsets=chosen,elapsed_seconds=time.perf_counter()-start,qiskit_max_discrepancy=channel_kernel_check(),trace_provenance=trace['provenance'],security_status='NO QKD SECRECY PROOF; classical conditional error-bound monitoring only; certified_secret_key_bits is null',family_note='0.01/6 per method; empirical and analytic are separate decision modes and are not OR combined',map_note='Fixed 3.3216% monitoring reference across all cells; adaptive reveal cost varies and is reported. Map onset fixed at 3 s; one synthetic drift realization.',source_calibration='Frozen calibration maxima in calibration/ (derived from the version 2 study); unchanged policy and null models. No attack outcomes fit thresholds.')
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2),flush=True)
