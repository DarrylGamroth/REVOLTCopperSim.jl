import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location("metrics",Path(__file__).parents[2]/"benchmarks/analyze_convergence.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Measurements(unittest.TestCase):
    def test_mask_and_quadrants(self):
        self.assertEqual(len(m.OFFSETS),716)
        base=np.ones((64,64));self.assertTrue(np.all(m.slopes(base,4)==0))
        for (r,c),expected in zip(m.ORIGINS,((.25,.25),(-.25,.25),(-.25,-.25),(.25,-.25))):
            image=base.copy();image[c:c+30,r:r+30]+=1
            s=m.slopes(image,4).reshape((716,2),order="F")
            self.assertTrue(np.all(s[:,0]==expected[0]));self.assertTrue(np.all(s[:,1]==expected[1]))
    def test_relative(self):
        self.assertEqual(m.relative(np.zeros(2),np.zeros(2)),0)
        self.assertEqual(m.relative(np.ones(2),np.zeros(2)),float("inf"))
        self.assertEqual(m.relative(np.array([2.]),np.array([1.])),1)
    def test_calibration_gain(self):
        rate=np.ones((64,64,13));rate[10,10,2:]=2
        reference=dict(rates=rate,interaction=np.eye(4,dtype=float))
        actual=dict(rates=rate,interaction=2*np.eye(4,dtype=float))
        r=m.compare(actual,reference,4)
        self.assertEqual(r['calibration_relative_l2'],1)
        self.assertEqual(r['calibration_gain'],2)
        self.assertEqual(r['rate_max_relative_l2'],0)
        self.assertEqual(r['slope_response_max_relative_l2'],0)
if __name__=='__main__':unittest.main()
