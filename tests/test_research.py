import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from qkd_monitor.quantum import QiskitKernel, exact_p1
from qkd_monitor.detectors import StartMixture
from qkd_monitor.statistics import audit_remainder_bound
from qkd_monitor.policies import monitor

class ResearchTests(unittest.TestCase):
    def test_bb84_physics(self):
        for bit in (0,1):
            for basis in (0,1):
                self.assertAlmostEqual(exact_p1(bit,basis,basis,0,False,0),bit)
                p = exact_p1(bit,basis,1-basis,0,False,0)
                self.assertAlmostEqual(p,.5)
                for q in (0,.02,.3):
                    error = np.mean([abs(exact_p1(bit,basis,basis,e,True,q)-bit) for e in (0,1)])
                    self.assertAlmostEqual(error,.25+.5*q)

    def test_kernel_interpolation(self):
        k=QiskitKernel()
        rng=np.random.default_rng(123)
        size=100
        bit,a,b,e,attack=(rng.integers(0,2,size) for _ in range(5))
        q=rng.uniform(0,.5,size)
        p=k.probabilities(bit,a,b,e,attack.astype(bool),q)
        for j in range(size):
            self.assertAlmostEqual(p[j],exact_p1(bit[j],a[j],b[j],e[j],attack[j],q[j]))

    def test_mixture_single_step_supermartingale(self):
        # Exact conditional expectation, including start activations.
        for history in ([],[0],[1],[0,1,0],[1,1,1,0]):
            d=StartMixture(100,stride=2)
            for i,x in enumerate(history): d.update(x,i)
            import copy
            d0,d1=copy.deepcopy(d),copy.deepcopy(d)
            v0=d0.update(0,len(history)); v1=d1.update(1,len(history))
            self.assertAlmostEqual(.95*v0+.05*v1,d.value)
            self.assertLessEqual(.98*v0+.02*v1,d.value+1e-12)

    def test_no_lookahead_policy(self):
        import json
        cfg=json.loads((Path(__file__).resolve().parents[1]/"configs/default.json").read_text())
        a=dict(error=np.tile([0,0,1],100),time=np.arange(300),n_emitted=300,event_start=100,event_end=200)
        b=dict(a);b["error"]=a["error"].copy();b["error"][150:]=1
        _,ta=monitor(a,cfg,12,adaptive=True,capture=True)
        _,tb=monitor(b,cfg,12,adaptive=True,capture=True)
        np.testing.assert_array_equal(ta["monitor_probability"][:151],tb["monitor_probability"][:151])

    def test_audit_edge_cases(self):
        self.assertEqual(audit_remainder_bound(10,0,0,0,0),1)
        self.assertEqual(audit_remainder_bound(10,10,1,0,0),0)
        b=audit_remainder_bound(10000,3000,30,1000,10)
        self.assertTrue(0 < b < .1)

if __name__=="__main__": unittest.main()
