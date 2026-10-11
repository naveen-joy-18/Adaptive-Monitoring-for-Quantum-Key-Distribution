import unittest,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from qkd_monitor.large_block import *

class LargeBlockTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/large_block.json').read_text())
    def test_qiskit_channel_and_trace(self):
        self.assertLess(channel_kernel_check(),1e-12)
        tr=build_trace(self.cfg)
        self.assertLess(np.max(tr['honest_qber']),self.cfg['q0'])
        self.assertTrue(np.all(tr['total_gain']>0))
    def test_rank_threshold_and_ties(self):
        vals=np.array([0,0,0,1,2,3,4,5,6])
        alpha=.2
        threshold=calibration_threshold(vals,alpha)
        trial=np.array([0,4,5,5.5,6,7])
        np.testing.assert_array_equal(trial>threshold,rank_pvalues(trial,vals)<=alpha)
        self.assertTrue(np.isinf(calibration_threshold(vals,.001)))
    def test_mixture_matches_enumeration(self):
        cfg=dict(self.cfg);cfg['sliding_frames']=[2];cfg['disjoint_frames']=2
        m=np.array([[8,10,9,12,7]]);e=np.array([[0,3,0,4,0]])
        _,paths=statistics(m,e,cfg,with_paths=True)
        for t in range(5):
            total=0
            for alt in cfg['alternative_qbers']:
                for start in range(5):
                    lr=1.
                    for j in range(start,t+1):
                        lr*=(alt/cfg['q0'])**e[0,j]*((1-alt)/(1-cfg['q0']))**(m[0,j]-e[0,j])
                    total+=lr
            self.assertAlmostEqual(paths[0,t,4],np.log(total/(5*3)))
    def test_budget_and_partition(self):
        cfg=dict(self.cfg);cfg['duration_s']=7
        tr=build_trace(cfg)
        m,e,tot,profile,rows=simulate_frames('policy_aware_subthreshold',cfg,tr,20,123,adaptive=True,capture=True)
        np.testing.assert_allclose(tot['audit_n']+tot['monitor_n']+tot['remaining_n'],tot['sifted'])
        self.assertTrue(np.all(e<=m))
        self.assertTrue(np.all(tot['intercepted_remaining']<=tot['remaining_n']))
        for row in rows:
            if row['test_probability']>.04:self.assertEqual(row['attack_fraction'],0.)
            self.assertLessEqual(row['effective_qber'],cfg['q0'])

if __name__=='__main__':unittest.main()
