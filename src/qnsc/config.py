"""
QNSC Configuration Management

Centralizes all tunable parameters for the QNSC architecture.
Supports environment variable overrides for deployment flexibility.
"""

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TopologyConfig:
    """Configuration for Topological Knot Detection (GUDHI)."""

    # Maximum edge length for Rips complex filtration
    max_edge_length: float = 1.0

    # Maximum homology dimension to compute (0=clusters, 1=loops, 2=voids)
    max_dimension: int = 2

    # Entropy threshold for "simple" vs "complex" topology
    low_entropy_threshold: float = 0.5
    high_entropy_threshold: float = 2.0


@dataclass
class QuantumConfig:
    """Configuration for Quantum Feature Mapping (Qiskit)."""

    # Feature dimension for quantum mapping (must be small for simulation)
    feature_dimension: int = 8

    # Default circuit repetitions (depth)
    default_reps: int = 2

    # Circuit reps for low/high entropy topologies
    low_entropy_reps: int = 1
    high_entropy_reps: int = 3

    # Entropy thresholds for circuit tuning
    low_entropy_threshold: float = 0.5
    high_entropy_threshold: float = 2.0

    # Entanglement structure: 'linear', 'circular', 'full', 'sca'
    default_entanglement: Literal["linear", "circular", "full", "sca"] = "linear"

    # Betti-1 threshold for switching to full entanglement
    betti1_full_entanglement_threshold: int = 5

    # Enforce PSD for kernel matrices
    enforce_psd: bool = True


@dataclass
class TensorConfig:
    """Configuration for Superfluid Vacuum (Quimb MPS)."""

    # Maximum bond dimension (entanglement capacity)
    max_bond_dimension: int = 32

    # SVD truncation cutoff (noise threshold)
    cutoff: float = 1e-6

    # Compression method: 'svd', 'dm' (density matrix)
    compression_method: Literal["svd", "dm"] = "svd"

    # Canonical form: 'left', 'right', 'mixed'
    canonical_form: Literal["left", "right", "mixed"] = "right"


@dataclass
class StorageConfig:
    """Configuration for Markov Blanket Store (Milvus)."""

    # Milvus connection URI (QNSC uses port 19531 to avoid conflicts)
    uri: str = field(default_factory=lambda: os.getenv("MILVUS_URI", "http://localhost:19531"))

    # Collection name for semantic states
    collection_name: str = "quantum_semantic_blanket"

    # Dense vector dimension
    dense_dim: int = 128

    # HNSW index parameters
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100

    # Default entropy threshold for search filtering
    default_entropy_threshold: float = 0.5

    # Number of results to return
    top_k: int = 5


@dataclass
class QNSCConfig:
    """
    Master configuration for the QNSC system.

    All component configurations are nested for organization.
    Can be instantiated with defaults or customized per-component.

    Example:
        config = QNSCConfig()
        config.topology.max_edge_length = 2.0
        config.quantum.default_reps = 3
    """

    topology: TopologyConfig = field(default_factory=TopologyConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    tensor: TensorConfig = field(default_factory=TensorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Global settings
    verbose: bool = field(
        default_factory=lambda: os.getenv("QNSC_VERBOSE", "false").lower() == "true"
    )

    @classmethod
    def from_env(cls) -> "QNSCConfig":
        """
        Create config from environment variables.

        Environment variables:
            MILVUS_URI: Milvus connection string
            QNSC_VERBOSE: Enable verbose logging
            QNSC_MAX_BOND_DIM: Override max bond dimension
        """
        config = cls()

        # Override from environment
        if max_bond := os.getenv("QNSC_MAX_BOND_DIM"):
            config.tensor.max_bond_dimension = int(max_bond)

        return config
