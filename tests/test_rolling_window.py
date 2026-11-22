import unittest

import numpy as np

from preprocess.acf import rolling_window


class TestRollingWindow(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1, 10], [2, 20], [3, 30]])

    def test_unit_window_keeps_every_observation(self):
        expected = self.x[np.newaxis, :, :]
        np.testing.assert_array_equal(rolling_window(self.x, 1), expected)

    def test_complete_windows_keep_lag_and_sample_axes(self):
        expected = np.stack([self.x[:-1], self.x[1:]])
        np.testing.assert_array_equal(rolling_window(self.x, 2), expected)
        np.testing.assert_array_equal(rolling_window(self.x, 3), self.x[:, None, :])

    def test_oversized_window(self):
        self.assertEqual(rolling_window(self.x, 5).shape, (5, 0, 2))
        padded = rolling_window(self.x, 5, sparse=False)
        expected = np.full((5, 3, 2), np.nan)
        expected[0] = self.x
        expected[1, :2] = self.x[1:]
        expected[2, :1] = self.x[2:]
        np.testing.assert_array_equal(padded, expected)

    def test_empty_series(self):
        for sparse in (True, False):
            with self.subTest(sparse=sparse):
                self.assertEqual(rolling_window(self.x[:0], 3, sparse).shape, (3, 0, 2))

    def test_nonpositive_window(self):
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                rolling_window(self.x, size)
