import unittest

import numpy as np

from TLearning import TL

class UT(unittest.TestCase):

    def setUp(self) -> None:
        self.process = TL(gravity_list=gravity_list, seeds=seeds)
        TL.train_eval(self.process)

    def test_case1(self):
        # tb.execute_cell([1, 4])
        result = np.array(self.process.results_list)
        assert result.shape == (len(gravity_list), len(seeds), int(self.process.sum_episodes/self.process.eval_freq))
        # X = tb.ref('X')
        # y = tb.ref('y')
        #
        # assert X.shape == (199, 7)
        # assert y.shape == (199,)
        #
        # # Make sure array is shuffled
        # assert X.tolist()[0][0] == pytest.approx(12.72, 0)

if __name__ == "__main__":
    gravity_list = [10.0, 20.0]
    seeds = [2, 6]
    test = UT()
    UT.setUp(test)
    UT.test_case1(test)