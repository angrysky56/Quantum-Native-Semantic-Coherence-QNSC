"""
Quantum Semantic Projector

Implements the 'Quantum Feature Map' aspect of the QNSC architecture.
Projects semantic data into Hilbert space using entangled feature maps,
tuned by topological metrics from the Knot Detector.

Mathematical Background:
    - Quantum Kernel: K(x,y) = |⟨φ(x)|φ(y)⟩|² (fidelity of quantum states)
    - ZZFeatureMap: Uses second-order Pauli-Z evolution for entanglement
    - The ZZ gates (e^{-iZjZk}) create entanglement between qubits,
      effectively "braiding" semantic features together

The key innovation is dynamic circuit parameterization based on topology:
    - High persistent entropy → Deeper circuit (more repetitions)
    - High Betti-1 (loops) → Full entanglement (all-to-all connectivity)
"""

from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

try:
    # Feature Maps
    from qiskit.circuit.library import ZZFeatureMap

    # Primitives (New V2 Interface)
    from qiskit.primitives import StatevectorSampler as Sampler
    from qiskit.quantum_info import Statevector
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    # Algorithms (Fidelity)
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.utils.algorithm_globals import algorithm_globals
    QISKIT_AVAILABLE = True
except ImportError:
    ZZFeatureMap = None
    FidelityQuantumKernel = None  # type: ignore
    ComputeUncompute = None  # type: ignore
    Sampler = None
    algorithm_globals = None # type: ignore
    Statevector = None
    QISKIT_AVAILABLE = False

from qnsc.config import QuantumConfig
from qnsc.types import QuantumKernelResult, TopologyMetrics


class QuantumSemanticProjector:
    """
    Implements the 'Quantum Feature Map' aspect of the architecture.

    Projects semantic data into Hilbert space using entangled feature maps,
    dynamically tuned by topological metrics.

    The ZZFeatureMap physically enacts the "knot" metaphor: the "braiding"
    of qubits via CNOT gates (ZjZk) creates the topological constraints
    of the semantic knot within the Hilbert space.

    Attributes:
        feature_dimension: Dimension of input semantic vectors
        reps: Circuit depth (repetitions)
        entanglement: Entanglement strategy ('linear', 'full', etc.)

    Example:
        >>> projector = QuantumSemanticProjector(
        ...     feature_dimension=4,
        ...     topology_metrics=metrics
        ... )
        >>> kernel_matrix = projector.compute_kernel_matrix(data_batch)
    """

    def __init__(
        self,
        feature_dimension: int,
        topology_metrics: TopologyMetrics | None = None,
        config: QuantumConfig | None = None
    ) -> None:
        """
        Initialize the Quantum Semantic Projector.

        Args:
            feature_dimension: Dimension of input semantic vectors.
                              This determines the number of qubits used.
            topology_metrics: Output from TopologicalKnotDetector.
                             Used to dynamically tune circuit parameters.
            config: Optional QuantumConfig for parameter override.
        """
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit is required for quantum feature mapping. "
                "Install with: pip install qiskit qiskit-machine-learning"
            )

        self.feature_dimension = feature_dimension
        self.config = config or QuantumConfig()

        # Dynamically tune circuit based on topological complexity
        self.reps, self.entanglement = self._tune_circuit(topology_metrics)

        # Build the kernel with current parameters
        self._kernel = self._build_kernel()

    def _tune_circuit(
        self,
        metrics: TopologyMetrics | None
    ) -> tuple[int, Literal["linear", "circular", "full", "sca"]]:
        """
        Dynamically adjust circuit depth and entanglement strategy.

        High entropy → Deeper circuit to capture complexity
        High Betti-1 (loops) → Full entanglement to capture cycles

        Args:
            metrics: Topological metrics from knot detection.

        Returns:
            Tuple of (reps, entanglement_type).
        """
        if metrics is None:
            return self.config.default_reps, self.config.default_entanglement

        entropy = metrics.get('persistent_entropy', 0)

        # Safely extract Betti-1 (index 1 if available)
        betti_numbers = metrics.get('betti_numbers', [])
        betti_1 = betti_numbers[1] if len(betti_numbers) > 1 else 0

        # Determine circuit depth based on entropy
        reps = self.config.default_reps
        if entropy > self.config.high_entropy_threshold:
            reps = self.config.high_entropy_reps
        elif entropy < self.config.low_entropy_threshold:
            reps = self.config.low_entropy_reps

        # Determine entanglement based on Betti-1 (loop count)
        entanglement: Literal["linear", "circular", "full", "sca"]
        if betti_1 > self.config.betti1_full_entanglement_threshold:
            entanglement = "full"
        else:
            entanglement = self.config.default_entanglement

        return reps, entanglement

    def _build_kernel(self) -> "FidelityQuantumKernel":
        """
        Construct the FidelityQuantumKernel with ZZFeatureMap.

        The ZZFeatureMap uses second-order Pauli-Z evolution:
            U_Φ(x) = ∏_d ( ∏_j e^{-iφ_j Z_j} ∏_{j<k} e^{-iφ_{jk} Z_j Z_k} )

        where the ZjZk terms create entanglement between qubits.

        Returns:
            Configured FidelityQuantumKernel instance.
        """
        # 1. Define Feature Map using the new function API (Qiskit 2.1+)
        # Returns a QuantumCircuit instead of deprecated BlueprintCircuit
        feature_map = ZZFeatureMap(
            feature_dimension=self.feature_dimension,
            reps=self.reps,
            entanglement=self.entanglement
        )

        # Store for direct access in get_state_vector
        self._feature_map = feature_map

        # 2. Define Fidelity Estimator
        # ComputeUncompute uses the Sampler primitive to estimate |⟨ψ|φ⟩|²
        sampler = Sampler()
        fidelity = ComputeUncompute(sampler=sampler)

        # 3. Create Kernel
        # enforce_psd=True ensures the kernel matrix is Positive Semi-Definite
        kernel = FidelityQuantumKernel(
            feature_map=feature_map,
            fidelity=fidelity,
            enforce_psd=self.config.enforce_psd
        )

        return kernel

    def compute_kernel_matrix(
        self,
        x_1: NDArray[np.float64],
        x_2: NDArray[np.float64] | None = None
    ) -> QuantumKernelResult:
        """
        Compute the Gram matrix (similarity matrix) for semantic vectors.

        The kernel value K(x,y) represents the quantum fidelity between
        states encoded from x and y - essentially how "similar" the
        semantic concepts are in the quantum Hilbert space.

        Args:
            x_1: First set of semantic vectors (N1, feature_dimension).
            x_2: Optional second set. If None, computes K(x_1, x_1).

        Returns:
            QuantumKernelResult with kernel matrix and circuit info.
        """
        # Validate input dimensions
        x_1 = np.asarray(x_1, dtype=np.float64)
        if x_1.ndim == 1:
            x_1 = x_1.reshape(1, -1)

        if x_1.shape[1] != self.feature_dimension:
            raise ValueError(
                f"Expected vectors of dimension {self.feature_dimension}, "
                f"got {x_1.shape[1]}"
            )

        if x_2 is not None:
            x_2 = np.asarray(x_2, dtype=np.float64)
            if x_2.ndim == 1:
                x_2 = x_2.reshape(1, -1)
            if x_2.shape[1] != self.feature_dimension:
                raise ValueError(
                    f"Expected vectors of dimension {self.feature_dimension}, "
                    f"got {x_2.shape[1]}"
                )

        # Compute the kernel matrix
        kernel_matrix = self._kernel.evaluate(x_1, x_2)

        return QuantumKernelResult(
            kernel_matrix=kernel_matrix,
            circuit_reps=self.reps,
            entanglement_type=self.entanglement
        )

    def get_state_vector(self, embedding: NDArray[np.float64]) -> NDArray[np.complex128]:
        """
        Get the explicit Statevector for a given embedding.

        Args:
            embedding: Single embedding vector (D,) or (1, D).

        Returns:
            Review complex state vector (2^feature_dimension,).
        """
        embedding = np.asarray(embedding, dtype=np.float64).flatten()
        if len(embedding) != self.feature_dimension:
             raise ValueError(f"Embedding dim {len(embedding)} != feature dim {self.feature_dimension}")

        # Bind parameters to the feature map circuit
        # ZZFeatureMap parameters are naturally ordered
        bound_circuit = self._feature_map.assign_parameters(embedding)
        return cast(NDArray[np.complex128], Statevector(bound_circuit).data)

    def update_topology(self, metrics: TopologyMetrics) -> None:
        """
        Update circuit parameters based on new topological metrics.

        This allows the projector to adapt to changing semantic complexity
        without full reinitialization.

        Args:
            metrics: New topological metrics from knot detection.
        """
        new_reps, new_entanglement = self._tune_circuit(metrics)

        if new_reps != self.reps or new_entanglement != self.entanglement:
            self.reps = new_reps
            self.entanglement = new_entanglement
            self._kernel = self._build_kernel()

    @property
    def circuit_info(self) -> dict[str, int | str]:
        """Get current circuit configuration."""
        return {
            "feature_dimension": self.feature_dimension,
            "reps": self.reps,
            "entanglement": self.entanglement,
        }
