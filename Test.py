import unittest
import numpy as np
from TL_v2 import TL

class UT(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up the TL class and do the training-evaluation loops.
        """
        self.process = TL(gravity_list=gravity_list, seeds=seeds)
        TL.train_eval(self.process)

    def test_case1(self):
        """
        Test if the shape of results is equal to (num_gravities, num_seeds, num_eval).
        """
        result = np.array(self.process.results_list)
        assert result.shape == (len(gravity_list), len(seeds), int(self.process.sum_episodes/self.process.eval_freq))

if __name__ == "__main__":
    gravity_list = [10.0, 20.0]
    seeds = [2, 6]
    test = UT()
    UT.setUp(test)
    UT.test_case1(test)