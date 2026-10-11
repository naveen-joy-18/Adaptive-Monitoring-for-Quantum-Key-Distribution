import unittest,sys,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qkd_monitor.schedule_bounds import *
class ScheduleBoundsTests(unittest.TestCase):
    def test_cardinality_bound_exhaustive(self):
        for vals in itertools.product([0,.02,.12,.4],repeat=5):
            for k in range(1,6):
                brute=min(sum(vals[i] for i in A) for A in itertools.combinations(range(5),k))
                self.assertAlmostEqual(worst_subset_mass(vals,k),brute)
                self.assertLessEqual(brute,k*np.mean(vals)+1e-12)
    def test_cyclic_window_bound_and_edges(self):
        p=[0,1,0]
        for d in [1,2,3]:
            w=cyclic_window_masses(p,d)
            self.assertAlmostEqual(w.mean(),d*np.mean(p))
            self.assertLessEqual(w.min(),d*np.mean(p))
        self.assertGreater(min(np.convolve(p,np.ones(2),'valid')),2*np.mean(p))
    def test_conditional_null_factor(self):
        q0=.035
        for v in [.05,.09,.18]:
            for q in np.linspace(0,q0,21):
                for r in [0,.019,.033,.114,1]:
                    expectation=1+r*(q*v/q0+(1-q)*(1-v)/(1-q0)-1)
                    self.assertLessEqual(expectation,1+1e-12)
        self.assertAlmostEqual(analytic_log_threshold(.01/6),np.log(600))
    def test_count_bound_not_detection_probability_order(self):
        self.assertLess(worst_subset_mass([0,.9,.9],2),2*.6)
        self.assertLess((1-0)*(1-.9),(1-.6)**2)
if __name__=='__main__':unittest.main()
