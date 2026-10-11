"""Elementary sampling-coverage bounds; these are not secrecy theorems."""
import numpy as np

def worst_subset_mass(probabilities,k):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=1 or len(p)==0 or not np.all((p>=0)&(p<=1)):
        raise ValueError('A nonempty probability vector is required')
    if not 1<=k<=len(p):raise ValueError('Invalid attack cardinality')
    return float(np.partition(p,k-1)[:k].sum())

def cyclic_window_masses(probabilities,duration):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=1 or not 1<=duration<=len(p):raise ValueError('Invalid window')
    doubled=np.r_[p,p[:duration-1]]
    return np.convolve(doubled,np.ones(duration),mode='valid')

def analytic_log_threshold(alpha):
    if not 0<alpha<1:raise ValueError('alpha must be in (0,1)')
    return float(np.log(1/alpha))
