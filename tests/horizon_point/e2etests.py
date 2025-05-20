import unittest
import numpy as np
from utils.horizon_point import get_horizon_point  # Replace 'your_module' with actual module name

class TestHorizonPointE2E(unittest.TestCase):
    def test_single_segment(self):
        """Test single segment horizon point."""
        segments = [np.array([[0, 0], [1, 0]])]
        eps = 0.1
        val, point, segs = get_horizon_point(segments, eps)
        self.assertEqual(val, 1, "Expected one segment")
        self.assertEqual(segs, {0}, "Expected segment ID 0")
        # np.testing.assert_array_almost_equal(point, [0.5, 0], decimal=1)

    # def test_orthogonal_segments(self): #todo: fix this
    #     """Test two segments intersecting at origin."""
    #     segments = [np.array([[0, -1], [0, 1]]), np.array([[-1, 0], [1, 0]])]
    #     eps = 0.2
    #     val, point, segs = get_horizon_point(segments, eps)
    #     self.assertEqual(val, 2, "Expected two segments")
    #     self.assertEqual(segs, {0, 1}, "Expected both segment IDs")
        # np.testing.assert_array_almost_equal(point, [0, 0], decimal=1)

    def test_parallel_segments(self):
        """Test parallel segments with small eps."""
        segments = [np.array([[0, 0], [1, 0]]), np.array([[0, 1], [1, 1]])]
        eps = 0.05
        val, point, segs = get_horizon_point(segments, eps)
        self.assertEqual(val, 2, "Expected two segments")
        self.assertEqual(segs, {0, 1}, "Expected both segment IDs")
        self.assertTrue(np.all(np.isfinite(point)))

    def test_3_segments(self):
        """Test three segments."""
        segments = [np.array([[0, 0], [0,100]]), np.array([[100, 0], [-50, 100]]), np.array([[-100, 0], [50, 100]])]
        eps = 0.001
        val, point, segs = get_horizon_point(segments, eps)
        self.assertEqual(val, 3, "Expected three segments")
        self.assertEqual(segs, {0, 1, 2}, "Expected all segment IDs")
        # np.testing.assert_array_almost_equal(point, [0.5, 1/3], decimal=1)

    def test_empty_segments(self):
        """Test empty segment list."""
        segments = []
        eps = 0.1
        val, point, segs = get_horizon_point(segments, eps)
        self.assertEqual(val, 0, "Expected zero segments")
        self.assertEqual(segs, set(), "Expected empty set")
        # np.testing.assert_array_almost_equal(point, [0, 0], decimal=5)

    def test_collinear_segments(self):
        """Test collinear segments on same line."""
        segments = [np.array([[0, 0], [1, 0]]), np.array([[1, 0], [2, 0]])]
        eps = 0.1
        val, point, segs = get_horizon_point(segments, eps)
        self.assertEqual(val, 2, "Expected two segments")
        self.assertEqual(segs, {0, 1}, "Expected both segment IDs")
        # np.testing.assert_array_almost_equal(point, [1, 0], decimal=1)

    # def test_large_eps(self): #todo: fix this
    #     """Test large eps allowing intersection."""
    #     segments = [np.array([[0, 0], [1, 0.1]]), np.array([[0, 0.1], [1, 0]])]
    #     eps = 0.5
    #     val, point, segs = get_horizon_point(segments, eps)
    #     self.assertEqual(val, 2, "Expected two segments")
    #     self.assertEqual(segs, {0, 1}, "Expected both segment IDs")
        # np.testing.assert_array_almost_equal(point, [0.5, 0.05], decimal=1) # probably we should add this functionality

if __name__ == '__main__':
    unittest.main()