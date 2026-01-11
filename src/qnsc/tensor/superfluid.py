"""
Superfluid Vacuum - Tensor Network State Manager

Implements the 'Superfluid Vacuum' metaphor using Matrix Product States (MPS).
Manages semantic states as tensor chains with dynamic SVD compression.

Mathematical Background:
    - Matrix Product States: Factorize high-dimensional state into tensor chain
    - Area Law: Entanglement scales with boundary, not volume
    - SVD Truncation: Removes noise while preserving essential correlations

The "superfluidity" (frictionless information flow) is achieved through
SVD truncation. By retaining only the largest singular values and discarding
the tail, we compress the state while preserving its essential correlations.

The bond dimension χ acts as the limit on semantic bandwidth:
    - Higher χ → More complex entanglement (thicker superfluid)
    - Lower χ → Stricter compression (thinner but cleaner flow)
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

try:
    import quimb.tensor as qtn
except ImportError:
    qtn = None

from qnsc.config import TensorConfig
from qnsc.types import VacuumState


class SuperfluidVacuum:
    """
    Implements the 'Superfluid Vacuum' metaphor using Matrix Product States.

    Manages semantic states as tensor chains with dynamic SVD compression.
    The compression removes "friction" (noise) while preserving coherent
    semantic structure.

    Attributes:
        num_sites: Number of semantic units/qubits in the chain
        max_bond: Maximum bond dimension (entanglement capacity)
        cutoff: SVD truncation threshold (noise removal)

    Example:
        >>> vacuum = SuperfluidVacuum(num_sites=8, max_bond_dimension=16)
        >>> vacuum.initialize_from_dense(quantum_projected_vector)
        >>> print(f"Vacuum Coherence: {vacuum.get_coherence_measure()}")
    """

    def __init__(
        self,
        num_sites: int,
        max_bond_dimension: int = 32,
        cutoff: float = 1e-6,
        config: TensorConfig | None = None
    ) -> None:
        """
        Initialize the Superfluid Vacuum.

        Args:
            num_sites: Number of semantic units in the chain.
            max_bond_dimension: The χ parameter - controls entanglement capacity.
            cutoff: Threshold for SVD truncation (removes frictional noise).
            config: Optional TensorConfig for parameter override.
        """
        if qtn is None:
            raise ImportError(
                "Quimb is required for tensor network operations. "
                "Install with: pip install quimb"
            )

        self.num_sites = num_sites

        if config is not None:
            self.max_bond = config.max_bond_dimension
            self.cutoff = config.cutoff
            self._compression_method: Literal["svd", "dm"] = config.compression_method
            self._canonical_form: Literal["left", "right", "mixed"] = config.canonical_form
        else:
            self.max_bond = max_bond_dimension
            self.cutoff = cutoff
            self._compression_method = "svd"
            self._canonical_form = "right"

        self._mps: qtn.MatrixProductState | None = None

    def initialize_from_dense(
        self,
        dense_vector: NDArray[np.complex128 | np.float64]
    ) -> "qtn.MatrixProductState":
        """
        Convert a raw high-dimensional vector into a Superfluid MPS.

        The dense vector represents the raw output from the Quantum Kernel
        or embedding layer. This method performs sequential SVDs to factorize
        the tensor into an MPS chain.

        Args:
            dense_vector: Raw state vector of dimension 2^num_sites.
                         Will be normalized to unit norm.

        Returns:
            The compressed MPS (also stored internally).

        Raises:
            ValueError: If vector dimension doesn't match 2^num_sites.
        """
        dense_vector = np.asarray(dense_vector)
        expected_dim = 2 ** self.num_sites

        if dense_vector.size != expected_dim:
            raise ValueError(
                f"Expected vector of size {expected_dim} (2^{self.num_sites}), "
                f"got {dense_vector.size}"
            )

        # Ensure normalization (Superfluid states must be normalized)
        norm = np.linalg.norm(dense_vector)
        if norm > 0:
            dense_vector = np.asarray(dense_vector / norm, dtype=dense_vector.dtype)

        # Reshape to tensor form
        dense_vector = dense_vector.reshape([2] * self.num_sites)

        # Create MPS from dense tensor
        # This performs sequential SVDs to factorize the tensor
        # dims=(2,)*num_sites assumes qubit-like local dimension (binary features)
        self._mps = qtn.MatrixProductState.from_dense(
            dense_vector,
            dims=(2,) * self.num_sites
        )

        # Apply initial compression immediately to enforce vacuum constraints
        self._compress()

        return self._mps

    def initialize_random(self, bond_dim: int | None = None) -> "qtn.MatrixProductState":
        """
        Initialize with a random MPS state.

        Useful for testing or as an initial state for variational algorithms.

        Args:
            bond_dim: Bond dimension for the random state.
                     Defaults to max_bond / 2.

        Returns:
            The random MPS.
        """
        bond_dim = bond_dim or (self.max_bond // 2)

        self._mps = qtn.MPS_rand_state(
            L=self.num_sites,
            bond_dim=bond_dim,
            phys_dim=2,
            normalize=True
        )

        return self._mps

    def _compress(self) -> None:
        """
        Apply SVD truncation to the tensor chain.

        This is the computational implementation of 'removing friction'.
        We retain only the bond dimension required to capture the coherent
        signal, discarding noise.

        The compression also creates a canonical form, enabling efficient
        computation of expectations and overlaps.
        """
        if self._mps is not None:
            # Compress using SVD with truncation
            self._mps.compress(
                max_bond=self.max_bond,
                cutoff=self.cutoff,
                method=self._compression_method,
                form=self._canonical_form
            )

    def evolve_state(self, operator_mpo: "qtn.MatrixProductOperator") -> None:
        """
        Evolve the vacuum state by applying a Matrix Product Operator.

        This models the 'evolution of thought' or semantic transformation.
        After evolution, the state is re-compressed to maintain superfluidity.

        Args:
            operator_mpo: The operator to apply (e.g., time evolution, gate).

        Raises:
            RuntimeError: If MPS not initialized.
        """
        if self._mps is None:
            raise RuntimeError("MPS not initialized. Call initialize_* first.")

        # Apply the operator (MPO) to the state (MPS) -> New MPS
        # This increases bond dimension (entanglement grows)
        self._mps = operator_mpo.apply(self._mps)

        # Re-compress to maintain superfluidity
        # Without this, bond dimension would grow exponentially
        self._compress()

    def get_coherence_measure(self) -> float:
        """
        Calculate Entanglement Entropy across the central bond.

        This serves as a proxy for the 'semantic coherence' of the vacuum.
        Higher entropy indicates stronger non-local semantic links.

        Returns:
            Von Neumann entropy at the central bipartition.

        Raises:
            RuntimeError: If MPS not initialized.
        """
        if self._mps is None:
            raise RuntimeError("MPS not initialized. Call initialize_* first.")

        # Split system in half to measure bipartition entanglement
        bipartition = self.num_sites // 2
        entropy = float(self._mps.entropy(bipartition))

        return entropy

    def get_bond_dimensions(self) -> list[int]:
        """
        Get the bond dimensions across all bonds.

        Returns:
            List of bond dimensions [χ₁, χ₂, ..., χ_{n-1}].
        """
        if self._mps is None:
            return []

        return list(self._mps.bond_sizes())

    def to_dense(self) -> NDArray[np.complex128]:
        """
        Convert MPS back to dense vector form.

        Warning: This is exponential in num_sites. Only use for small systems.

        Returns:
            Dense state vector.
        """
        if self._mps is None:
            raise RuntimeError("MPS not initialized.")

        return np.asarray(self._mps.to_dense()).flatten()

    def to_vacuum_state(self) -> VacuumState:
        """
        Export the current state as a VacuumState dataclass.

        Returns:
            VacuumState with dense vector and metadata.
        """
        if self._mps is None:
            raise RuntimeError("MPS not initialized.")

        dense = self.to_dense()
        bond_dims = self.get_bond_dimensions()

        # Enforce Canonical Global Phase
        # Find dominant element (largest magnitude) and rotate phase to make it Real+Positive
        # This prevents |ψ> and i|ψ> being orthogonal in [Re,Im] space.
        if dense.size > 0:
            idx = np.argmax(np.abs(dense))
            if np.abs(dense[idx]) > 1e-9:
                phase = np.angle(dense[idx])
                dense = dense * np.exp(-1j * phase)

        # Concatenate Real and Imaginary parts to preserve phase info in float storage
        # [Re(x0), Re(x1)..., Im(x0), Im(x1)...]
        flat_complex = np.concatenate([np.real(dense), np.imag(dense)]).astype(np.float64)

        return VacuumState(
            dense_vector=flat_complex,
            num_sites=self.num_sites,
            bond_dimension=max(bond_dims) if bond_dims else 1,
            entanglement_entropy=self.get_coherence_measure(),
            is_normalized=True
        )

    def overlap(self, other: "SuperfluidVacuum") -> float:
        """
        Compute the overlap (fidelity) with another vacuum state.

        Args:
            other: Another SuperfluidVacuum instance.

        Returns:
            |⟨ψ|φ⟩|² - the fidelity between states.
        """
        if self._mps is None or other._mps is None:
            raise RuntimeError("Both MPS must be initialized.")

        overlap_val = self._mps.H @ other._mps
        return float(abs(overlap_val) ** 2)

    @property
    def is_initialized(self) -> bool:
        """Check if the MPS has been initialized."""
        return self._mps is not None
