#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results_v3'
s=pd.read_csv(out/'summary.csv');meta=json.loads((out/'metadata.json').read_text());m=s[s.phase=='map']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
ds=sorted(m.duration_ms.unique());fs=sorted(m.fraction.unique())
def mat(policy,mode,column):
 return m[(m.policy==policy)&(m.threshold_mode==mode)].pivot(index='fraction',columns='duration_ms',values=column).loc[fs,ds].to_numpy()
def save(fig,name):
 fig.savefig(out/(name+'.png'),dpi=250,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for row,mode in enumerate(['empirical','analytic']):
 for col,policy in enumerate(['fixed','adaptive']):
  ax=axes[row,col];z=100*mat(policy,mode,'event_detection_rate')
  im=ax.imshow(z,origin='lower',vmin=0,vmax=100,cmap='viridis',aspect='auto')
  ax.set_xticks(range(len(ds)),ds);ax.set_yticks(range(len(fs)),[f'{100*f:g}' for f in fs])
  for i in range(len(fs)):
   for j in range(len(ds)):ax.text(j,i,f'{z[i,j]:.0f}',ha='center',va='center',fontsize=8,color='black' if z[i,j]>65 else 'white')
  ax.set_title(f'{policy.capitalize()} sampling  {mode} threshold')
  ax.set_xlabel('Burst duration (ms)');ax.set_ylabel('Intercepted fraction (%)')
fig.colorbar(im,ax=axes,label='In-event detection (%)',shrink=.85)
save(fig,'figure1_detection_map')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,policy in zip(axes,['fixed','adaptive']):
 gap=100*(mat(policy,'empirical','event_detection_rate')-mat(policy,'analytic','event_detection_rate'))
 im=ax.imshow(gap,origin='lower',vmin=0,vmax=20,cmap='magma',aspect='auto')
 ax.set_xticks(range(len(ds)),ds);ax.set_yticks(range(len(fs)),[f'{100*f:g}' for f in fs]);ax.set_title(policy.capitalize());ax.set_xlabel('Burst duration (ms)');ax.set_ylabel('Intercepted fraction (%)')
 for i in range(len(fs)):
  for j in range(len(ds)):ax.text(j,i,f'{gap[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white')
fig.colorbar(im,ax=axes,label='Empirical minus analytic detection (percentage points)')
save(fig,'figure2_guarantee_price')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
cost=100*(mat('adaptive','empirical','mean_revealed_fraction')-mat('fixed','empirical','mean_revealed_fraction'))
im=axes[0].imshow(cost,origin='lower',cmap='coolwarm',aspect='auto',vmin=-1,vmax=1)
axes[0].set_xticks(range(len(ds)),ds);axes[0].set_yticks(range(len(fs)),[f'{100*f:g}' for f in fs]);axes[0].set_xlabel('Burst duration (ms)');axes[0].set_ylabel('Intercepted fraction (%)');axes[0].set_title('Adaptive minus fixed reveal cost')
fig.colorbar(im,ax=axes[0],label='Percentage points')
for pol,style in [('fixed','--'),('adaptive','-')]:
 for mode,color in [('empirical','#1d6982'),('analytic','#a45232')]:
  x=m[(m.policy==pol)&(m.threshold_mode==mode)&(m.fraction==.25)].sort_values('duration_ms')
  axes[1].plot(x.duration_ms,100*x.event_detection_rate,style+'o',color=color,label=f'{pol} {mode}')
axes[1].set_xlabel('Burst duration (ms)');axes[1].set_ylabel('In-event detection (%)');axes[1].set_title('25% interception cross section');axes[1].legend(fontsize=8)
save(fig,'figure3_cost_and_transition')
h=s[(s.phase=='attacker_holdout')&(s.threshold_mode=='empirical')];fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
onsets=sorted(h.onset_s.unique());x=np.arange(len(onsets));w=.33
for k,policy in enumerate(['fixed','adaptive']):
 a=h[h.policy==policy].sort_values('onset_s');y=100*a.event_detection_rate.to_numpy();err=np.vstack([y-100*a.ci_low.to_numpy(),100*a.ci_high.to_numpy()-y]);ax.bar(x+(k-.5)*w,y,w,yerr=err,capsize=3,label=policy)
ax.set_xticks(x,[f'{t:g} s' for t in onsets]);ax.set_xlabel('Selected onset cross-evaluated against both policies');ax.set_ylabel('In-event detection (%)');ax.set_ylim(0,100);ax.legend();save(fig,'figure4_attacker_holdout')
print('Four revision figures created')
