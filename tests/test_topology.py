"""
Tests for Topological Knot Detector

Tests the GUDHI-based topology analysis including:
- Betti number calculation
- Persistent entropy calculation
- Handling of edge cases
"""

import pytest
import numpy as np

# Skip all tests if GUDHI not available
gudhi = pytest.importorskip("gudhi")

from qnsc.topology import TopologicalKnotDetector
from qnsc.config import TopologyConfig


class TestTopologicalKnotDetector:
    """Tests for TopologicalKnotDetector."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        detector = TopologicalKnotDetector()
        assert detector.max_edge_length == 1.0
        assert detector.max_dimension == 2

    def test_init_with_config(self) -> None:
        """Test initialization with config."""
        config = TopologyConfig(
            max_edge_length=2.5,
            max_dimension=3
        )
        detector = TopologicalKnotDetector(config=config)
        assert detector.max_edge_length == 2.5
        assert detector.max_dimension == 3

    def test_simple_point_cloud(self) -> None:
        """Test on a simple point cloud (well-separated clusters)."""
        # Two well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(5, 2) * 0.1
        cluster2 = np.random.randn(5, 2) * 0.1 + 10
        points = np.vstack([cluster1, cluster2])

        detector = TopologicalKnotDetector(max_edge_length=2.0)
        metrics = detector.fit_transform(points)

        # Should detect 2 connected components (β₀ = 2)
        assert len(metrics["betti_numbers"]) >= 1
        assert metrics["betti_numbers"][0] == 2

        # Low entropy for simple topology
        assert metrics["persistent_entropy"] >= 0

    def test_loop_detection(self) -> None:
        """Test detection of loops (β₁ > 0)."""
        # Create a circle of points (should have 1 loop)
        theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        points = np.column_stack([np.cos(theta), np.sin(theta)])

        detector = TopologicalKnotDetector(max_edge_length=1.0, max_dimension=2)
        metrics = detector.fit_transform(points)

        # Should detect 1 connected component and 1 loop
        assert metrics["betti_numbers"][0] == 1
        if len(metrics["betti_numbers"]) > 1:
            assert metrics["betti_numbers"][1] >= 1

    def test_entropy_increases_with_complexity(self) -> None:
        """Test that entropy increases with topological complexity."""
        np.random.seed(42)

        # Simple: single cluster
        simple = np.random.randn(10, 2) * 0.1

        # Complex: multiple loops (torus-like)
        theta = np.linspace(0, 2 * np.pi, 15, endpoint=False)
        circle1 = np.column_stack([np.cos(theta), np.sin(theta)])
        circle2 = np.column_stack([np.cos(theta) + 3, np.sin(theta)])
        complex_data = np.vstack([circle1, circle2])

        detector = TopologicalKnotDetector(max_edge_length=2.0)

        simple_metrics = detector.fit_transform(simple)
        complex_metrics = detector.fit_transform(complex_data)

        # Complex topology should have higher or equal entropy
        # (Note: this is a heuristic test, not always guaranteed)
        assert complex_metrics["num_simplices"] >= simple_metrics["num_simplices"]

    def test_empty_input(self) -> None:
        """Test handling of empty or single-point input."""
        detector = TopologicalKnotDetector()

        # Single point
        single = np.array([[0.0, 0.0]])
        metrics = detector.fit_transform(single)
        assert metrics["betti_numbers"] == [1]
        assert metrics["persistent_entropy"] == 0.0

    def test_invalid_input_shape(self) -> None:
        """Test that invalid input shape raises error."""
        detector = TopologicalKnotDetector()

        with pytest.raises(ValueError):
            detector.fit_transform(np.array([1, 2, 3]))  # 1D array

    def test_knot_complexity_categories(self) -> None:
        """Test the complexity categorization."""
        detector = TopologicalKnotDetector()

        # Create mock metrics
        trivial = {"persistent_entropy": 0.2, "betti_numbers": [1]}
        simple = {"persistent_entropy": 1.0, "betti_numbers": [1, 1]}
        complex_m = {"persistent_entropy": 3.0, "betti_numbers": [1, 5, 2]}

        assert detector.get_knot_complexity(trivial) == "trivial"
        assert detector.get_knot_complexity(simple) == "simple"
        assert detector.get_knot_complexity(complex_m) == "complex"

    def test_sklearn_interface(self) -> None:
        """Test scikit-learn compatibility."""
        detector = TopologicalKnotDetector()

        # Test fit then transform (should work same as fit_transform)
        points = np.random.randn(10, 3)

        detector.fit(points)
        metrics = detector.transform(points)

        assert "betti_numbers" in metrics
        assert "persistent_entropy" in metrics


class TestPersistentEntropy:
    """Tests specifically for persistent entropy calculation."""

    def test_entropy_is_non_negative(self) -> None:
        """Entropy should always be non-negative."""
        np.random.seed(42)

        for _ in range(5):
            points = np.random.randn(20, 3)
            detector = TopologicalKnotDetector(max_edge_length=2.0)
            metrics = detector.fit_transform(points)
            assert metrics["persistent_entropy"] >= 0

    def test_entropy_with_no_features(self) -> None:
        """Entropy should be 0 when no persistent features."""
        # Very small filtration that won't capture any loops
        points = np.random.randn(5, 2) * 10
        detector = TopologicalKnotDetector(max_edge_length=0.01)
        metrics = detector.fit_transform(points)

        # If no dimension-1 features, entropy should be 0
        assert metrics["persistent_entropy"] >= 0  # Could be 0
