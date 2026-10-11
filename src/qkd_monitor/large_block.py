"""Frame-aggregated Qiskit monitoring benchmarks with finite calibration ranks.

A frame is a fixed batch with constant channel parameters and test probability.
Conditional binomial counts exactly marginalize the ideal-qubit model within
that frame. No random coin, audit tag, or current outcome is visible to Eve.
"""
import json
import numpy as np
from scipy.stats import beta
from .quantum import exact_p1

METHODS = ['end_block','disjoint','sliding_multiscale','cusum','fixed_mixture','adaptive_mixture']
SCENARIOS = ['stationary','drift_replay','null_boundary','benign_excursion',
             'burst_200ms','burst_50ms','weak_burst','policy_aware_aggressive',
             'policy_aware_subthreshold']


def build_trace(cfg, seed=7701):
    rng=np.random.default_rng(seed)
    frames=int(round(cfg['duration_s']/cfg['frame_s']))
    t=np.arange(frames)*cfg['frame_s']
    theta=np.zeros(frames); loss=np.zeros(frames)
    residual=.0; log_loss=0.
    for j in range(frames):
        residual=.995*residual+rng.normal(0,.0015)
        log_loss=.998*log_loss+rng.normal(0,.008)
        theta[j]=np.clip(.085+.045*np.sin(2*np.pi*t[j]/6)+residual,0,.15)
        loss[j]=np.clip(cfg['signal_gain']*np.exp(log_loss),.0008,.0032)
    # Physical rotation error, then intrinsic channel and readout error.
    rotation=np.sin(theta)**2
    intrinsic=.006
    signal_q=intrinsic+(1-2*intrinsic)*rotation
    signal_q=cfg['readout_flip']+(1-2*cfg['readout_flip'])*signal_q
    q=(loss*signal_q+.5*cfg['dark_gain'])/(loss+cfg['dark_gain'])
    return dict(time_s=t,theta_rad=theta,signal_gain=loss,honest_qber=q,
                total_gain=loss+cfg['dark_gain'],provenance='Synthetic OU residual polarization and log-gain trace; no experimental fit')


def scenario_profile(name,cfg,trace):
    frames=len(trace['time_s']);t=trace['time_s']
    q=np.asarray(trace['honest_qber']).copy();gain=np.asarray(trace['total_gain']).copy()
    active=np.zeros(frames,dtype=bool);fraction=np.zeros(frames)
    kind='null'
    if name=='stationary':
        q[:]=.015;gain[:]=cfg['signal_gain']+cfg['dark_gain']
    elif name=='null_boundary': q[:]=cfg['q0']
    elif name=='benign_excursion':
        active=(t>=3)&(t<3.2);q[active]=.08;kind='benign_anomaly'
    elif name.startswith('burst_') or name=='weak_burst':
        duration=cfg.get('burst_duration_s',.05 if name=='burst_50ms' else .2)
        onset=cfg.get('burst_onset_s',3.)
        active=(t>=onset)&(t<onset+duration-1e-10)
        fraction[active]=cfg.get('burst_fraction',.06 if name=='weak_burst' else .25)
        kind='attack'
    elif name.startswith('policy_aware'):
        active=(t>=3)&(t<6);kind='attack'
    elif name!='drift_replay':raise ValueError(name)
    return dict(q=q,gain=gain,active=active,fraction=fraction,kind=kind)


def channel_kernel_check():
    # Qiskit verifies closed-form intercept-resend marginal before aggregation.
    errors=[]
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix
    for theta in (0.,.085,.15):
        for bit in (0,1):
            for basis in (0,1):
                qc=QuantumCircuit(1)
                if bit:qc.x(0)
                if basis:qc.h(0)
                qc.ry(2*theta,0)
                if basis:qc.h(0)
                p1=float(DensityMatrix.from_instruction(qc).probabilities()[1])
                errors.append(abs(abs(p1-bit)-np.sin(theta)**2))
    for q in (.005,.02,.08):
        physical=np.mean([abs(exact_p1(bit,basis,basis,e,True,q)-bit)
                          for bit in (0,1) for basis in (0,1) for e in (0,1)])
        errors.append(abs(physical-(.25+.5*q)))
    if max(errors)>1e-12:raise AssertionError('Qiskit channel marginal mismatch')
    return max(errors)


def simulate_frames(name,cfg,trace,replicates,seed,adaptive=False,fixed_p=None,capture=False):
    rng=np.random.default_rng(seed)
    profile=scenario_profile(name,cfg,trace)
    frames=len(trace['time_s']);r=replicates
    monitor_n=np.zeros((r,frames),dtype=np.int32)
    monitor_e=np.zeros((r,frames),dtype=np.int32)
    totals={k:np.zeros(r,dtype=np.float64) for k in ['sifted','audit_n','audit_e','monitor_n','monitor_e','remaining_n','remaining_e','intercepted_remaining','intercepted_total']}
    ewma=np.full(r,.015)
    queue=np.full((r,cfg['feedback_delay_frames']),.015)
    budget=np.full(r,.015*cfg['pulse_rate_hz']*cfg['duration_s']*cfg['signal_gain']*cfg['basis_match_probability'])
    trace_rows=[]
    pulses=int(round(cfg['pulse_rate_hz']*cfg['frame_s']))
    pbase=cfg['monitor_low'] if fixed_p is None else fixed_p
    for j in range(frames):
        pi=np.where(ewma>cfg['suspicion_qber'],cfg['monitor_high'],cfg['monitor_low']) if adaptive else np.full(r,pbase)
        qh=profile['q'][j]
        f=np.full(r,profile['fraction'][j])
        if name.startswith('policy_aware') and profile['active'][j]:
            # Knowledge of the current predictable public test probability and
            # archived ambient trace; no access to current private coins/outcomes.
            target=.08 if name=='policy_aware_aggressive' else .032
            nominal=np.clip((target-qh)/(.25-.5*qh),0,.35)
            f=np.where(pi<=.04,nominal,0.)
            expected=pulses*profile['gain'][j]*cfg['basis_match_probability']
            f=np.minimum(f,np.maximum(budget,0)/max(expected,1))
            budget=np.maximum(0,budget-f*expected)
        # Subtracting Eve's unobserved outcomes gives the Qiskit marginal q.
        qe=qh+f*(.25-.5*qh)
        n=rng.binomial(pulses,profile['gain'][j]*cfg['basis_match_probability'],r)
        a=rng.binomial(n,cfg['audit_probability'])
        m=rng.binomial(n-a,pi)
        rem=n-a-m
        ia=rng.binomial(a,f);im=rng.binomial(m,f);ir=rng.binomial(rem,f)
        qa=.25+.5*qh
        ae=rng.binomial(ia,qa)+rng.binomial(a-ia,qh)
        me=rng.binomial(im,qa)+rng.binomial(m-im,qh)
        re=rng.binomial(ir,qa)+rng.binomial(rem-ir,qh)
        # Preserve correlation between intercepted-round exposure and errors.
        it=ia+im+ir
        monitor_n[:,j]=m;monitor_e[:,j]=me
        for key,value in [('sifted',n),('audit_n',a),('audit_e',ae),('monitor_n',m),('monitor_e',me),('remaining_n',rem),('remaining_e',re),('intercepted_remaining',ir),('intercepted_total',it)]:
            totals[key]+=value
        observed=np.where(m>0,me/np.maximum(m,1),ewma)
        # The EWMA state update itself uses delayed announcements, not a hidden
        # instantaneous EWMA. Queue holds raw rates from past completed frames.
        delayed=queue[:,0].copy()
        queue[:,:-1]=queue[:,1:];queue[:,-1]=observed
        ewma=(1-cfg['ewma_weight'])*ewma+cfg['ewma_weight']*delayed
        if capture:
            trace_rows.append(dict(time_s=float(trace['time_s'][j]),sifted=int(n[0]),monitor_n=int(m[0]),monitor_e=int(me[0]),
                                   test_probability=float(pi[0]),attack_fraction=float(f[0]),honest_qber=float(qh),effective_qber=float(qe[0]),
                                   audit_n=int(a[0]),remaining_n=int(rem[0])))
    return monitor_n,monitor_e,totals,profile,trace_rows


def kl_score(n,e,q0):
    # One-sided binomial likelihood ratio (a statistic, not an exact p value).
    p=np.divide(e,n,out=np.zeros_like(e,dtype=float),where=n>0)
    clipped=np.clip(p,1e-15,1-1e-15)
    val=e*np.log(clipped/q0)+(n-e)*np.log((1-clipped)/(1-q0))
    return np.where((p>q0)&(n>0),val,0.)


def statistics(m,e,cfg,with_paths=False):
    r,frames=m.shape
    n_c=np.cumsum(m,axis=1,dtype=np.float64)
    e_c=np.cumsum(e,axis=1,dtype=np.float64)
    maxima=np.zeros((r,5));paths=None
    if with_paths:paths=np.zeros((r,frames,5))
    end=kl_score(n_c[:,-1],e_c[:,-1],cfg['q0']);maxima[:,0]=end
    if with_paths:paths[:,-1,0]=end
    # Common deterministic frame grid; whole-run max absorbs every look/scale.
    block=cfg['disjoint_frames']
    for j in range(block-1,frames,block):
        start=j-block
        nn=n_c[:,j]-(n_c[:,start] if start>=0 else 0)
        ee=e_c[:,j]-(e_c[:,start] if start>=0 else 0)
        score=kl_score(nn,ee,cfg['q0']);maxima[:,1]=np.maximum(maxima[:,1],score)
        if with_paths:paths[:,j,1]=score
    if frames%block:
        start=(frames//block)*block-1
        score=kl_score(n_c[:,-1]-n_c[:,start],e_c[:,-1]-e_c[:,start],cfg['q0'])
        maxima[:,1]=np.maximum(maxima[:,1],score)
        if with_paths:paths[:,-1,1]=score
    for window in cfg['sliding_frames']:
        if window>frames:continue
        nn=n_c[:,window-1:]-np.column_stack([np.zeros(r),n_c[:,:-window]])
        ee=e_c[:,window-1:]-np.column_stack([np.zeros(r),e_c[:,:-window]])
        score=kl_score(nn,ee,cfg['q0'])
        maxima[:,2]=np.maximum(maxima[:,2],np.max(score,axis=1))
        if with_paths:paths[:,window-1:,2]=np.maximum(paths[:,window-1:,2],score)
    alts=np.array(cfg['alternative_qbers'])
    l1=np.log(alts/cfg['q0']);l0=np.log((1-alts)/(1-cfg['q0']))
    cusum=np.zeros((r,len(alts)));logsum=np.full((r,len(alts)),-np.inf)
    for j in range(frames):
        llr=e[:,j,None]*l1+(m[:,j,None]-e[:,j,None])*l0
        cusum=np.maximum(0,cusum+llr)
        c=np.max(cusum,axis=1)
        logsum=llr+np.logaddexp(logsum,0.)
        maxalt=np.max(logsum,axis=1)
        active=maxalt+np.log(np.exp(logsum-maxalt[:,None]).sum(axis=1))-np.log(frames*len(alts))
        dormant=np.log((frames-j-1)/frames) if j<frames-1 else -np.inf
        loge=np.logaddexp(dormant,active)
        maxima[:,3]=np.maximum(maxima[:,3],c)
        maxima[:,4]=np.maximum(maxima[:,4],loge)
        if with_paths:paths[:,j,3]=c;paths[:,j,4]=loge
    return maxima,paths


def rank_pvalues(values,calibration):
    s=np.sort(np.asarray(calibration))
    # Inclusive ties give conservative Monte Carlo p values.
    return (1+len(s)-np.searchsorted(s,values,side='left'))/(len(s)+1)


def calibration_threshold(values,alpha):
    # Reject with strict > threshold; handle discrete ties conservatively.
    s=np.sort(np.asarray(values));allowed=int(np.floor(alpha*(len(s)+1)))-1
    if allowed<0:return float('inf')
    return float(s[len(s)-allowed-1])


def classical_audit(totals,eps,audit_probability=.05):
    # A current-round audit coin is fresh and hidden. Unlike static Serfling,
    # this martingale bound permits future outcomes to react to past transcripts.
    # D_i=X_i(1-A_i/a) has E[D_i|past,X_i]=0, D_i<=1,
    # variance <=(1-a)/a. Union of fixed-n Freedman bounds with
    # delta_n=eps/(n*(n+1)) gives coverage at random sifted n.
    n=totals['sifted'];a=totals['audit_n'];ae=totals['audit_e']
    logterm=np.log(np.maximum(n,1)*(np.maximum(n,1)+1)/eps)
    variance=np.maximum(n,1)*(1-audit_probability)/audit_probability
    penalty=np.sqrt(2*variance*logterm)+(2/3)*logterm
    count_upper=np.minimum(n,ae/audit_probability+penalty)
    rem=totals['remaining_n']
    upper=np.clip((count_upper-ae-totals['monitor_e'])/np.maximum(rem,1),0,1)
    actual=totals['remaining_e']/np.maximum(rem,1)
    return upper,actual


def ci(k,n):
    return (0. if not k else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k)))
