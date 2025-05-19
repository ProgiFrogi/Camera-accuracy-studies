import unittest
import numpy as np
from unittest.mock import patch
# import os
# path_to_root = os.path.join(os.path.dirname(__file__), '..')
# path_to_root = os.path.join(path_to_root, '..')
# path_to_root = os.path.join(path_to_root, 'utils')
# import sys
# sys.path.append(sys)
# import sys
# sys.path.append('../../') #export PYTHONPATH = $PYTHONPATH:~/Desktop/Camera-accuracy-studies  
from utils.horizon_point import line, intersect, get_horizon_point, cross2d  

class TestLineClass(unittest.TestCase):
    def setUp(self):
        # Initialize a default line for reuse in tests
        self.center = np.array([0, 0])
        self.direction = np.array([1, 0])
        self.line = line(center=self.center, direction=self.direction, calc_rot=1, internal_id=0)

    def test_init_normalizes_direction(self):
        # Test that direction is normalized
        direction = np.array([3, 4])
        l = line(center=self.center, direction=direction)
        self.assertAlmostEqual(np.linalg.norm(l.direction), 1.0)
        expected_direction = direction / np.linalg.norm(direction)
        np.testing.assert_array_almost_equal(l.direction, expected_direction)

    def test_add_point(self):
        # Test adding a point and checking projection
        point = np.array([2, 1])
        self.line.add_point(point, line_id=1)
        self.assertEqual(len(self.line.points), 1)
        self.assertEqual(self.line.points[0][0], np.dot(self.line.direction, point - self.center))
        self.assertEqual(self.line.points[0][1], 1)

    def test_sort_points(self):
        # Test sorting points and updating other_lines_ids
        points = [np.array([1, 0]), np.array([-1, 0]), np.array([2, 0])]
        for i, p in enumerate(points):
            self.line.add_point(p, line_id=i)
        self.line.sort_points()
        expected_points = [-1, 1, 2]
        expected_ids = [1, 0, 2]
        self.assertEqual(self.line.points, expected_points)
        self.assertEqual(self.line.other_lines_ids, expected_ids)

    def test_get_point_id(self):
        # Test finding the nearest point ID
        self.line.add_point(np.array([1, 0]), line_id=0)
        self.line.add_point(np.array([2, 0]), line_id=1)
        self.line.sort_points()
        point = np.array([1.6, 0])
        near_id = self.line.get_point_id(point)
        self.assertEqual(near_id, 1)  # Should pick point at x=2 as closest

    def test_get_point(self):
        # Test retrieving points by ID, including edge cases
        self.line.add_point(np.array([1, 0]), line_id=0)
        self.line.sort_points()
        point = self.line.get_point(0)
        np.testing.assert_array_almost_equal(point, np.array([1, 0]))
        # Test boundary points
        point_neg = self.line.get_point(-1)
        np.testing.assert_array_almost_equal(point_neg, np.array([-9999, 0]))
        point_end = self.line.get_point(1)
        np.testing.assert_array_almost_equal(point_end, np.array([10001, 0]))
        # Test invalid index
        with self.assertRaises(IndexError):
            self.line.get_point(-2)

    def test_should_add_point(self):
        # Test should_add_point logic
        point = np.array([1, 1])
        result = self.line.should_add_point(point)
        self.assertTrue(result)  # Should be true based on calc_rot and geometry
        point = np.array([1, -1])
        result = self.line.should_add_point(point)
        self.assertFalse(result)

    def test_next_point_rot(self):
        # Test next_point_rot for choosing correct direction
        self.line.add_point(np.array([0, 0]), line_id=0)
        self.line.add_point(np.array([1, 0]), line_id=1)
        self.line.add_point(np.array([2, 0]), line_id=2)
        self.line.sort_points()
        prev = np.array([-1, 1])
        next_id = self.line.next_point_rot(point_id=1, prev=prev)
        self.assertEqual(next_id, 2)  # Should choose next point based on rotation


class TestIntersectFunction(unittest.TestCase):
    def test_intersect_perpendicular_lines(self):
        # Test intersection of two perpendicular lines
        line1 = line(center=np.array([0, 0]), direction=np.array([1, 0]))
        line2 = line(center=np.array([0, 0]), direction=np.array([0, 1]))
        intersection = intersect(line1, line2)
        np.testing.assert_array_almost_equal(intersection, np.array([0, 0]))

    def test_intersect_parallel_lines(self):
        # Test intersection of nearly parallel lines (should handle gracefully)
        line1 = line(center=np.array([0, 0]), direction=np.array([1, 0]))
        line2 = line(center=np.array([0, 1]), direction=np.array([1, 0.0001]))
        intersection = intersect(line1, line2)
        # Since lines are nearly parallel, intersection may be far or unstable
        self.assertTrue(np.all(np.isfinite(intersection)))

    def test_intersect_offset_lines(self):
        # Test intersection of lines with offset centers
        line1 = line(center=np.array([0, 0]), direction=np.array([1, 1]))
        line2 = line(center=np.array([1, 0]), direction=np.array([1, -1]))
        intersection = intersect(line1, line2)
        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(intersection, expected, decimal=5)


class TestGetHorizonPoint(unittest.TestCase):
    def test_single_segment(self):
        # Test with a single segment
        segments = [np.array([[0, 0], [1, 0]])]
        eps = 0.1
        val, point, segs = get_horizon_point(segments, eps)
        self.assertEqual(val, 1)  # Only one segment
        self.assertEqual(len(segs), 1)
        self.assertTrue(np.all(np.isfinite(point)))

    def test_two_intersecting_segments(self):
        # Test with two segments that can intersect within eps
        segments = [
            np.array([[0, 0], [1, 0]]),
            np.array([[0, 0], [0, 1]])
        ]
        eps = 0.1
        val, point, segs = get_horizon_point(segments, eps)
        self.assertGreaterEqual(val, 2)  # Should find both segments
        self.assertTrue(np.all(np.isfinite(point)))
        np.testing.assert_array_almost_equal(point, np.array([0, 0]), decimal=1)

    def test_parallel_segments(self):
        # Test with parallel segments
        segments = [
            np.array([[0, 0], [1, 0]]),
            np.array([[0, 1], [1, 1]])
        ]
        eps = 0.1
        val, point, segs = get_horizon_point(segments, eps)
        self.assertEqual(val, 2)  # Should handle parallel case
        self.assertTrue(np.all(np.isfinite(point)))

    def test_empty_segments(self):
        # Test with empty segment list
        segments = []
        eps = 0.1
        val, point, segs = get_horizon_point(segments, eps)
        self.assertEqual(val, 0)
        self.assertEqual(len(segs), 0)
        np.testing.assert_array_almost_equal(point, np.array([0, 0]))


class TestCross2D(unittest.TestCase):
    def test_cross2d_orthogonal_vectors(self):
        # Test cross product of orthogonal vectors
        x = np.array([1, 0])
        y = np.array([0, 1])
        result = cross2d(x, y)
        self.assertEqual(result, 1)

    def test_cross2d_parallel_vectors(self):
        # Test cross product of parallel vectors
        x = np.array([1, 0])
        y = np.array([2, 0])
        result = cross2d(x, y)
        self.assertEqual(result, 0)

    def test_cross2d_negative_result(self):
        # Test cross product yielding negative result
        x = np.array([0, 1])
        y = np.array([1, 0])
        result = cross2d(x, y)
        self.assertEqual(result, -1)


if __name__ == '__main__':
    unittest.main()