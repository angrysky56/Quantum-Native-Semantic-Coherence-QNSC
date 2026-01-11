"""
Tests for Quantum Semantic Projector

Tests the Qiskit-based quantum feature mapping including:
- Circuit construction
- Kernel matrix computation
- Dynamic tuning based on topology
"""

import pytest
import numpy as np

# Skip all tests if Qiskit not available
pytest.importorskip("qiskit")

from qnsc.quantum import QuantumSemanticProjector
from qnsc.config import QuantumConfig
from qnsc.types import TopologyMetrics


class TestQuantumSemanticProjector:
    """Tests for QuantumSemanticProjector."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        projector = QuantumSemanticProjector(feature_dimension=4)
        assert projector.feature_dimension == 4
        assert projector.reps == 2  # Default
        assert projector.entanglement == "linear"  # Default

    def test_init_with_config(self) -> None:
        """Test initialization with config."""
        config = QuantumConfig(
            default_reps=3,
            default_entanglement="full"
        )
        projector = QuantumSemanticProjector(
            feature_dimension=4,
            config=config
        )
        assert projector.reps == 3
        assert projector.entanglement == "full"

    def test_tuning_with_high_entropy(self) -> None:
        """Test circuit tuning with high entropy topology."""
        metrics: TopologyMetrics = {
            "betti_numbers": [1, 2],
            "persistent_entropy": 3.5,  # High entropy
            "simplex_tree_dim": 2,
            "num_simplices": 100
        }

        projector = QuantumSemanticProjector(
            feature_dimension=4,
            topology_metrics=metrics
        )

        # High entropy should increase reps
        assert projector.reps == 3  # high_entropy_reps

    def test_tuning_with_high_betti1(self) -> None:
        """Test circuit tuning with high Betti-1 (loops)."""
        metrics: TopologyMetrics = {
            "betti_numbers": [1, 10],  # Many loops
            "persistent_entropy": 1.0,
            "simplex_tree_dim": 2,
            "num_simplices": 100
        }

        projector = QuantumSemanticProjector(
            feature_dimension=4,
            topology_metrics=metrics
        )

        # High β₁ should trigger full entanglement
        assert projector.entanglement == "full"

    def test_kernel_matrix_shape(self) -> None:
        """Test kernel matrix has correct shape."""
        projector = QuantumSemanticProjector(feature_dimension=2)

        # 5 samples, 2 features
        X = np.random.randn(5, 2)
        result = projector.compute_kernel_matrix(X)

        assert result["kernel_matrix"].shape == (5, 5)

    def test_kernel_matrix_symmetric(self) -> None:
        """Test kernel matrix is symmetric."""
        projector = QuantumSemanticProjector(feature_dimension=2)

        X = np.random.randn(3, 2)
        result = projector.compute_kernel_matrix(X)

        matrix = result["kernel_matrix"]
        np.testing.assert_array_almost_equal(matrix, matrix.T, decimal=5)

    def test_kernel_matrix_psd(self) -> None:
        """Test kernel matrix is positive semi-definite."""
        projector = QuantumSemanticProjector(feature_dimension=2)

        X = np.random.randn(4, 2)
        result = projector.compute_kernel_matrix(X)

        # All eigenvalues should be >= 0 (with small tolerance)
        eigenvalues = np.linalg.eigvalsh(result["kernel_matrix"])
        assert np.all(eigenvalues >= -1e-6)

    def test_kernel_diagonal_is_one(self) -> None:
        """Test kernel diagonal entries are 1 (self-similarity)."""
        projector = QuantumSemanticProjector(feature_dimension=2)

        X = np.random.randn(3, 2)
        result = projector.compute_kernel_matrix(X)

        # Diagonal should be close to 1
        diagonal = np.diag(result["kernel_matrix"])
        np.testing.assert_array_almost_equal(diagonal, np.ones(3), decimal=2)

    def test_kernel_between_two_sets(self) -> None:
        """Test kernel computation between two different sets."""
        projector = QuantumSemanticProjector(feature_dimension=2)

        X1 = np.random.randn(3, 2)
        X2 = np.random.randn(4, 2)

        result = projector.compute_kernel_matrix(X1, X2)

        assert result["kernel_matrix"].shape == (3, 4)

    def test_dimension_mismatch_error(self) -> None:
        """Test error on dimension mismatch."""
        projector = QuantumSemanticProjector(feature_dimension=4)

        X = np.random.randn(3, 2)  # Wrong dimension

        with pytest.raises(ValueError):
            projector.compute_kernel_matrix(X)

    def test_update_topology(self) -> None:
        """Test updating topology metrics."""
        projector = QuantumSemanticProjector(feature_dimension=4)

        initial_reps = projector.reps

        # Update with high entropy
        new_metrics: TopologyMetrics = {
            "betti_numbers": [1, 10],
            "persistent_entropy": 5.0,
            "simplex_tree_dim": 2,
            "num_simplices": 100
        }

        projector.update_topology(new_metrics)

        # Should have changed
        assert projector.reps != initial_reps or projector.entanglement == "full"

    def test_circuit_info(self) -> None:
        """Test circuit info property."""
        projector = QuantumSemanticProjector(feature_dimension=4)

        info = projector.circuit_info

        assert "feature_dimension" in info
        assert "reps" in info
        assert "entanglement" in info
        assert info["feature_dimension"] == 4


class TestQuantumKernelValues:
    """Tests for kernel value behavior."""

    def test_identical_vectors(self) -> None:
        """Test kernel of identical vectors is 1."""
        projector = QuantumSemanticProjector(feature_dimension=2)

        x = np.array([[0.5, 0.5]])
        result = projector.compute_kernel_matrix(x, x)

        np.testing.assert_almost_equal(result["kernel_matrix"][0, 0], 1.0, decimal=2)

    def test_single_vector(self) -> None:
        """Test kernel with single vector."""
        projector = QuantumSemanticProjector(feature_dimension=2)

        x = np.array([0.5, 0.5])  # 1D input
        result = projector.compute_kernel_matrix(x)

        assert result["kernel_matrix"].shape == (1, 1)
