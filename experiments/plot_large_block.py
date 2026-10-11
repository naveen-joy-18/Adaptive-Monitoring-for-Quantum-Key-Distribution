#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',default='results_v2');a=p.parse_args();out=Path(a.results)
s=pd.read_csv(out/'summary.csv');tr=pd.read_csv(out/'synthetic_drift_trace.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','figure.dpi':200})
fig,ax=plt.subplots(3,1,figsize=(8,6.3),sharex=True,constrained_layout=True)
ax[0].plot(tr.time_s,tr.theta_rad,color='#295f7a');ax[0].set_ylabel('Residual angle (rad)')
ax[1].plot(tr.time_s,100*tr.total_gain,color='#497351');ax[1].set_ylabel('Detection gain (%)')
ax[2].plot(tr.time_s,100*tr.honest_qber,color='#986245');ax[2].axhline(3.5,color='black',ls='--',label='Null bound 3.5%');ax[2].set_ylabel('Expected QBER (%)');ax[2].set_xlabel('Time (s)');ax[2].legend(fontsize=9)
fig.savefig(out/'figure1_drift.png');fig.savefig(out/'figure1_drift.svg');plt.close(fig)
methods=['end_block','disjoint','sliding_multiscale','cusum','fixed_mixture','adaptive_mixture']
labels=['End block','Disjoint','Multiscale','CUSUM','Mixture','Adaptive']
fig,axes=plt.subplots(2,2,figsize=(9,6.5),constrained_layout=True)
for ax,name in zip(axes.flat,['burst_200ms','burst_50ms','weak_burst','benign_excursion']):
 g=s[s.scenario==name].set_index('method').loc[methods];v=g.event_detection_rate.to_numpy()*100
 err=np.stack([v-g.event_detection_ci_low.to_numpy()*100,g.event_detection_ci_high.to_numpy()*100-v])
 ax.bar(range(6),v,color=['#a2afb8']*4+['#497b98','#ac6545']);ax.errorbar(range(6),v,yerr=err,fmt='none',color='black',capsize=3,lw=1)
 ax.set_xticks(range(6),labels,rotation=35,ha='right',fontsize=9);ax.set_ylim(0,103);ax.set_ylabel('In-event alerts (%)');ax.set_title(name.replace('_',' '))
fig.savefig(out/'figure2_detection.png');fig.savefig(out/'figure2_detection.svg');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(9,6.2),sharex=True,constrained_layout=True)
for i,name in enumerate(['policy_aware_aggressive','policy_aware_subthreshold']):
 for policy,col in [('fixed','#497b98'),('adaptive','#ac6545')]:
  t=pd.read_csv(out/(name+'_'+policy+'_trace.csv'))
  axes[i,0].plot(t.time_s,t.test_probability,label=policy,color=col,lw=1)
  axes[i,1].plot(t.time_s,t.attack_fraction,label=policy,color=col,lw=1)
 axes[i,0].set_ylabel('Test probability');axes[i,1].set_ylabel('Intercept fraction')
 axes[i,0].set_title(name.replace('policy_aware_',''));axes[i,1].set_title('Policy-aware attacker actions')
 for ax in axes[i]:ax.axvspan(3,6,color='#777777',alpha=.08);ax.legend(fontsize=8)
for ax in axes[-1]:ax.set_xlabel('Time (s)')
fig.savefig(out/'figure3_attacker.png');fig.savefig(out/'figure3_attacker.svg');plt.close(fig)
h=pd.read_csv(out/'null_holdout.csv');h=h[(h.method=='family_any')&(h.null_stratum=='null_boundary')]
fig,ax=plt.subplots(figsize=(8,3.7),constrained_layout=True);v=100*h.rate.to_numpy()
ax.errorbar(np.arange(len(h)),v,yerr=np.stack([v-100*h.ci_low,100*h.ci_high-v]),fmt='o',capsize=4,color='#295f7a')
ax.axhline(1,color='black',ls='--',label='Per-block family budget 1%');ax.set_xticks(range(len(h)),h.benchmark.str.replace('_',' '),rotation=35,ha='right',fontsize=9);ax.set_ylabel('Any-method null alerts (%)');ax.legend(fontsize=8)
fig.savefig(out/'figure4_null.png');fig.savefig(out/'figure4_null.svg');plt.close(fig)
