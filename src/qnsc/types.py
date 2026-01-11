"""
QNSC Type Definitions

Defines structured data types for inter-component communication.
Uses TypedDict for JSON-serializable dictionaries and dataclasses for rich objects.
"""

from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray


class TopologyMetrics(TypedDict):
    """
    Output from TopologicalKnotDetector.

    Attributes:
        betti_numbers: List of Betti numbers [β₀, β₁, ...] where
                       β₀ = connected components, β₁ = loops, β₂ = voids
        persistent_entropy: Shannon entropy of the persistence diagram
        simplex_tree_dim: Dimension of the computed simplex tree
        num_simplices: Total number of simplices in the complex
    """
    betti_numbers: list[int]
    persistent_entropy: float
    simplex_tree_dim: int
    num_simplices: int


class QuantumKernelResult(TypedDict):
    """
    Output from quantum kernel computation.

    Attributes:
        kernel_matrix: The Gram matrix (similarity matrix) for semantic vectors
        circuit_reps: Number of circuit repetitions used
        entanglement_type: Entanglement strategy used ('linear', 'full', etc.)
    """
    kernel_matrix: NDArray[np.float64]
    circuit_reps: int
    entanglement_type: str


@dataclass
class VacuumState:
    """
    Represents a Superfluid Vacuum state (MPS-compressed semantic state).

    Attributes:
        dense_vector: Flattened vector representation for storage
        num_sites: Number of semantic units in the MPS chain
        bond_dimension: Actual bond dimension after compression
        entanglement_entropy: Bipartite entanglement entropy (coherence measure)
        is_normalized: Whether the state is normalized
    """
    dense_vector: NDArray[np.float64]
    num_sites: int
    bond_dimension: int
    entanglement_entropy: float
    is_normalized: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dense_vector": self.dense_vector.tolist(),
            "num_sites": self.num_sites,
            "bond_dimension": self.bond_dimension,
            "entanglement_entropy": self.entanglement_entropy,
            "is_normalized": self.is_normalized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VacuumState":
        """Create from dictionary."""
        return cls(
            dense_vector=np.array(data["dense_vector"]),
            num_sites=data["num_sites"],
            bond_dimension=data["bond_dimension"],
            entanglement_entropy=data["entanglement_entropy"],
            is_normalized=data.get("is_normalized", True),
        )


@dataclass
class BlanketSearchResult:
    """
    Result from Markov Blanket search.

    Attributes:
        id: Primary key of the matched document
        score: Hybrid search score (combined dense + sparse)
        topic_tag: Metadata topic tag
        entropy_level: Stored entropy level of the matched state
        distance: Distance/similarity metric value
    """
    id: int
    score: float
    topic_tag: str
    entropy_level: float
    distance: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """
    Complete result from the QNSC pipeline.

    Attributes:
        topology: Topological metrics from knot detection
        vacuum_state: Compressed superfluid state
        stored: Whether the state was stored in the blanket
        metadata: Additional processing metadata
    """
    topology: TopologyMetrics
    vacuum_state: VacuumState | None = None
    stored: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
