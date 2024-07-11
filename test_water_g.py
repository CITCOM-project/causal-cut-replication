import unittest
import numpy as np
import water_g
import pandas as pd


class TestWaterG(unittest.TestCase):
    def setUp(self):
        self.assigned = [["MV101", 1], ["P101", 0], ["P102", 0]]
        self.followed_1 = [["MV101", 1], ["P101", 1], ["P102", 1]]
        self.followed_2 = [["MV101", 1], ["P101", 0], ["P102", 1]]

    def test_setup_xo_t_do_eq(self):
        xo_t_do = water_g.setup_xo_t_do(self.assigned, self.assigned)
        np.testing.assert_array_equal(xo_t_do, [0, 0, 0, 0, 0])

    def test_setup_xo_t_do_sensor_1(self):
        xo_t_do = water_g.setup_xo_t_do(self.assigned, self.followed_1)
        np.testing.assert_array_equal(xo_t_do, [0, 0, 1, None, None])

    def test_setup_xo_t_do_sensor_2(self):
        xo_t_do = water_g.setup_xo_t_do(self.assigned, self.followed_2)
        np.testing.assert_array_equal(xo_t_do, [0, 0, 0, 1, None])

    def test_setup_fault_time_fast(self):
        min, max = 1, 2
        df = pd.DataFrame({"values": [1, 1, 1, 0], "time": [15, 16, 17, 18]})
        df["within_safe_range"] = df["values"].between(min, max)
        df["fault_time"] = water_g.setup_fault_time_fast(df)
        np.testing.assert_array_equal(
            df["fault_time"], [17.999, 17.999, 17.999, 17.999]
        )

    def test_setup_fault_time_fast_broadcast(self):
        min, max = 1, 2
        df = pd.DataFrame(
            {"id": [1, 1, 2, 2], "values": [1, 1, 1, 0], "time": [15, 16, 17, 18]}
        )
        df["within_safe_range"] = df["values"].between(min, max)
        df["fault_time"] = df.groupby("id").apply(water_g.setup_fault_time_fast).values

        np.testing.assert_array_equal(
            df["fault_time"], [30.999, 30.999, 17.999, 17.999]
        )

    def test_setup_fault_time_fast_no_fault(self):
        min, max = 1, 2
        df = pd.DataFrame({"values": [1, 1, 1], "time": [15, 16, 17]})
        df["within_safe_range"] = df["values"].between(min, max)
        df["fault_time"] = water_g.setup_fault_time_fast(df)
        np.testing.assert_array_equal(df["fault_time"], [31.999, 31.999, 31.999])

    def test_setup_fault_time_fast_fault_first(self):
        min, max = 1, 2
        df = pd.DataFrame({"values": [0, 1, 1], "time": [15, 16, 17]})
        df["within_safe_range"] = df["values"].between(min, max)
        df["fault_time"] = water_g.setup_fault_time_fast(df)
        np.testing.assert_array_equal(df["fault_time"], [14.999, 14.999, 14.999])
