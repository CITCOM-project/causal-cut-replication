import unittest
import numpy as np
import water_g


class TestWaterG(unittest.TestCase):
    def setUp(self):
        self.assigned = [["MV101", 1], ["P101", 0], ["P102", 0]]
        self.followed_1 = [["MV101", 1], ["P101", 1], ["P102", 1]]
        self.followed_2 = [["MV101", 1], ["P101", 0], ["P102", 1]]

    def test_setup_xo_t_do_eq(self):
        xo_t_do = water_g.setup_xo_t_do(self.assigned, self.assigned)
        xo_t_do_fast = water_g.setup_xo_t_do_fast(self.assigned, self.assigned)
        np.testing.assert_array_equal(xo_t_do, xo_t_do_fast)

    def test_setup_xo_t_do_sensor_1(self):
        xo_t_do = water_g.setup_xo_t_do(self.assigned, self.followed_1)
        xo_t_do_fast = water_g.setup_xo_t_do_fast(self.assigned, self.followed_1)
        print(xo_t_do, xo_t_do_fast)
        np.testing.assert_array_equal(xo_t_do, xo_t_do_fast)

    def test_setup_xo_t_do_sensor_2(self):
        xo_t_do = water_g.setup_xo_t_do(self.assigned, self.followed_2)
        xo_t_do_fast = water_g.setup_xo_t_do_fast(self.assigned, self.followed_2)
        np.testing.assert_array_equal(xo_t_do, xo_t_do_fast)
