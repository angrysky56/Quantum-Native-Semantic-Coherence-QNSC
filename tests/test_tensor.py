"""
Tests for Superfluid Vacuum (Tensor Networks)

Tests the Quimb-based MPS operations including:
- State initialization
- Compression
- Coherence measurement
"""

import pytest
import numpy as np

# Skip all tests if Quimb not available
pytest.importorskip("quimb")

from qnsc.tensor import SuperfluidVacuum
from qnsc.config import TensorConfig


class TestSuperfluidVacuum:
    """Tests for SuperfluidVacuum."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        vacuum = SuperfluidVacuum(num_sites=4)
        assert vacuum.num_sites == 4
        assert vacuum.max_bond == 32
        assert vacuum.cutoff == 1e-6
        assert not vacuum.is_initialized

    def test_init_with_config(self) -> None:
        """Test initialization with config."""
        config = TensorConfig(
            max_bond_dimension=16,
            cutoff=1e-4
        )
        vacuum = SuperfluidVacuum(num_sites=4, config=config)
        assert vacuum.max_bond == 16
        assert vacuum.cutoff == 1e-4

    def test_initialize_from_dense(self) -> None:
        """Test MPS creation from dense vector."""
        vacuum = SuperfluidVacuum(num_sites=3)  # 2^3 = 8 dim

        # Random normalized state
        np.random.seed(42)
        state = np.random.randn(8) + 1j * np.random.randn(8)
        state = state / np.linalg.norm(state)

        vacuum.initialize_from_dense(state)

        assert vacuum.is_initialized
        assert len(vacuum.get_bond_dimensions()) == 2  # n-1 bonds

    def test_initialize_wrong_dimension(self) -> None:
        """Test error on wrong dimension."""
        vacuum = SuperfluidVacuum(num_sites=3)  # Expects 8-dim

        state = np.random.randn(10)  # Wrong size

        with pytest.raises(ValueError):
            vacuum.initialize_from_dense(state)

    def test_initialize_random(self) -> None:
        """Test random MPS initialization."""
        vacuum = SuperfluidVacuum(num_sites=4)

        vacuum.initialize_random(bond_dim=4)

        assert vacuum.is_initialized

    def test_compression_reduces_bond_dimension(self) -> None:
        """Test that compression limits bond dimension."""
        vacuum = SuperfluidVacuum(
            num_sites=4,
            max_bond_dimension=4
        )

        # Create state with potentially high entanglement
        np.random.seed(42)
        state = np.random.randn(16) + 1j * np.random.randn(16)

        vacuum.initialize_from_dense(state)

        # All bonds should be <= max_bond
        bond_dims = vacuum.get_bond_dimensions()
        assert all(d <= 4 for d in bond_dims)

    def test_coherence_measure(self) -> None:
        """Test entanglement entropy calculation."""
        vacuum = SuperfluidVacuum(num_sites=4)

        # Product state (no entanglement)
        product_state = np.zeros(16)
        product_state[0] = 1.0  # |0000⟩

        vacuum.initialize_from_dense(product_state)
        entropy = vacuum.get_coherence_measure()

        # Product state should have very low entropy
        assert entropy < 0.1

    def test_coherence_entangled_state(self) -> None:
        """Test entropy with entangled state."""
        vacuum = SuperfluidVacuum(num_sites=4)

        # GHZ-like state (maximally entangled)
        ghz = np.zeros(16)
        ghz[0] = 1 / np.sqrt(2)   # |0000⟩
        ghz[15] = 1 / np.sqrt(2)  # |1111⟩

        vacuum.initialize_from_dense(ghz)
        entropy = vacuum.get_coherence_measure()

        # GHZ state should have non-zero entropy
        assert entropy > 0

    def test_to_dense_roundtrip(self) -> None:
        """Test dense -> MPS -> dense roundtrip."""
        vacuum = SuperfluidVacuum(
            num_sites=3,
            max_bond_dimension=4  # High enough for exact
        )

        np.random.seed(42)
        original = np.random.randn(8)
        original = original / np.linalg.norm(original)

        vacuum.initialize_from_dense(original.astype(np.complex128))
        recovered = vacuum.to_dense()

        # Should recover original (possibly with global phase)
        # Check overlap instead of direct comparison
        overlap = np.abs(np.vdot(original, recovered))
        assert overlap > 0.99

    def test_to_vacuum_state(self) -> None:
        """Test export to VacuumState dataclass."""
        vacuum = SuperfluidVacuum(num_sites=3)

        state = np.random.randn(8) + 1j * np.random.randn(8)
        vacuum.initialize_from_dense(state)

        vs = vacuum.to_vacuum_state()

        assert vs.num_sites == 3
        assert vs.bond_dimension >= 1
        assert vs.entanglement_entropy >= 0
        assert len(vs.dense_vector) == 8

    def test_overlap(self) -> None:
        """Test overlap between two vacuum states."""
        v1 = SuperfluidVacuum(num_sites=3)
        v2 = SuperfluidVacuum(num_sites=3)

        # Same state should have overlap 1
        state = np.zeros(8)
        state[0] = 1.0

        v1.initialize_from_dense(state.astype(np.complex128))
        v2.initialize_from_dense(state.astype(np.complex128))

        overlap = v1.overlap(v2)
        np.testing.assert_almost_equal(overlap, 1.0, decimal=5)

    def test_overlap_orthogonal(self) -> None:
        """Test overlap of orthogonal states."""
        v1 = SuperfluidVacuum(num_sites=3)
        v2 = SuperfluidVacuum(num_sites=3)

        state1 = np.zeros(8)
        state1[0] = 1.0

        state2 = np.zeros(8)
        state2[1] = 1.0

        v1.initialize_from_dense(state1.astype(np.complex128))
        v2.initialize_from_dense(state2.astype(np.complex128))

        overlap = v1.overlap(v2)
        np.testing.assert_almost_equal(overlap, 0.0, decimal=5)

    def test_not_initialized_error(self) -> None:
        """Test error when accessing uninitialized vacuum."""
        vacuum = SuperfluidVacuum(num_sites=3)

        with pytest.raises(RuntimeError):
            vacuum.get_coherence_measure()

        with pytest.raises(RuntimeError):
            vacuum.to_dense()


class TestMPSCompression:
    """Tests specifically for MPS compression behavior."""

    def test_cutoff_removes_small_values(self) -> None:
        """Test that cutoff removes small singular values."""
        vacuum = SuperfluidVacuum(
            num_sites=4,
            max_bond_dimension=16,
            cutoff=0.1  # Aggressive cutoff
        )

        np.random.seed(42)
        state = np.random.randn(16)

        vacuum.initialize_from_dense(state.astype(np.complex128))

        # With aggressive cutoff, bond dims should be reduced
        bond_dims = vacuum.get_bond_dimensions()
        assert any(d < 4 for d in bond_dims)  # At least some reduction

    def test_normalization(self) -> None:
        """Test that MPS states are normalized."""
        vacuum = SuperfluidVacuum(num_sites=3)

        # Non-normalized input
        state = np.random.randn(8) * 100

        vacuum.initialize_from_dense(state.astype(np.complex128))
        recovered = vacuum.to_dense()

        norm = np.linalg.norm(recovered)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)
