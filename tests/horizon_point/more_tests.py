import unittest
import numpy as np
from unittest.mock import patch
from utils.horizon_point import (cluster_segments, horizon_point_optimization_loss,
                        horizon_point_optimization_loss_v2, optimize_point,
                        segment_clusterization_loss, optimize_eps)

class TestSegmentClusteringE2E(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        self.segments = [
            np.array([[0, 40], [0, 20]]),  # Vertical
            np.array([[40, 0], [20, 0]]),  # Horizontal
            np.array([[0, 0], [10, 10]])    # Diagonal
        ]
        self.eps = 0.2

    def test_cluster_segments_single_cluster(self):
        """Test clustering with segments forming one cluster."""
        segments = [np.array([[10,0], [9,10]]), np.array([[-10,0], [-9,10]])]
        clusters = cluster_segments(segments, self.eps)
        self.assertEqual(len(clusters), 1, "Expected one cluster")
        self.assertEqual(clusters[0][0], {0, 1}, "Expected both segment IDs")
        np.testing.assert_array_almost_equal(clusters[0][1], [0,100], decimal=-1)

    def test_cluster_segments_disjoint_clusters(self):
        """Test clustering with disjoint segment groups."""
        segments = [
            np.array([[0, 0], [1, 0]]),      # Cluster 1
            np.array([[0, 0.1], [1, 0.1]]),
            np.array([[2, 1], [2, 5]])   # Cluster 2
        ]
        clusters = cluster_segments(segments, 0.005)
        self.assertGreaterEqual(len(clusters), 2, "Expected at least two clusters")
        cluster1_ids = clusters[0][0]
        cluster2_ids = clusters[1][0]
        self.assertTrue(cluster1_ids.isdisjoint(cluster2_ids), "Clusters should be disjoint")
        self.assertTrue({0, 1}.issubset(cluster1_ids) or {0, 1}.issubset(cluster2_ids), "Expected segments 0, 1 together")
        self.assertTrue(2 in cluster1_ids or 2 in cluster2_ids, "Expected segment 2 in a cluster")

    def test_cluster_segments_empty_input(self):
        """Test clustering with empty segments."""
        clusters = cluster_segments([], self.eps)
        self.assertEqual(len(clusters), 0, "Expected no clusters")

    # def test_horizon_point_optimization_loss(self):
    #     """Test loss function with known point and segments."""
    #     point = np.array([0, 0])
    #     segments = [np.array([[0, -1], [0, 1]]), np.array([[-1, 0], [1, 0]])]
    #     loss = horizon_point_optimization_loss(point, segments)
    #     expected_loss = 0  # Perpendicular segments at origin
    #     self.assertAlmostEqual(loss, expected_loss, places=5)

    # def test_horizon_point_optimization_loss_v2(self):
    #     """Test v2 loss function with angled segments."""
    #     point = np.array([0, 0])
    #     segments = [np.array([[0, 0], [1, 1]])]  # 45-degree segment
    #     loss = horizon_point_optimization_loss_v2(point, segments)
    #     expected_loss = np.tan(np.pi/4) * np.sqrt(2)  # tan(45°) * segment length
    #     self.assertAlmostEqual(loss, expected_loss, places=5)

    @patch('scipy.optimize.minimize')
    def test_optimize_point(self, mock_minimize):
        """Test point optimization with mocked minimize."""
        point = np.array([0, 0])
        segments = [np.array([[0, -1], [0, 1]]), np.array([[-1, 0], [1, 0]])]
        mock_minimize.return_value = type('obj', (), {'x': np.array([0.1, 0.1])})()
        optimized_point = optimize_point(point, segments)
        np.testing.assert_array_equal(optimized_point, [0.1, 0.1])
        mock_minimize.assert_called_once()

    def test_segment_clusterization_loss_insufficient_clusters(self):
        """Test clusterization loss with fewer than two clusters."""
        segments = [np.array([[0, -1], [0, 1]])]  # One segment
        loss = segment_clusterization_loss(self.eps, segments)
        self.assertEqual(loss, 1e9, "Expected large loss for insufficient clusters")

    @patch('utils.horizon_point.optimize_point')
    def test_segment_clusterization_loss_valid_clusters(self, mock_optimize):
        """Test clusterization loss with valid clusters."""
        segments = self.segments
        mock_optimize.side_effect = lambda p, s: p  # Return input point
        with patch('utils.horizon_point.cluster_segments') as mock_cluster:
            mock_cluster.return_value = [
                ({0, 1}, np.array([0, 0])),
                ({2}, np.array([0.5, 0.5]))
            ]
            loss = segment_clusterization_loss(self.eps, segments)
            self.assertLess(loss, 1e7, "Expected finite loss")
            self.assertGreater(loss, 0, "Expected positive loss")

    # @patch('scipy.optimize.minimize')
    def test_optimize_eps(self):
        """Test eps optimization."""
        segments = self.segments
        # mock_minimize.return_value = type('obj', (), {'x': np.array([0.15])})()
        optimized_eps = optimize_eps(segments)
        np.testing.assert_array_almost_equal(optimized_eps, 0,decimal=1)
        # mock_minimize.assert_called_once()

if __name__ == '__main__':
    unittest.main()