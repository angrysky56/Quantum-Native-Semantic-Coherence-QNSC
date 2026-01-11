"""
Topological Knot Detector

Implements the 'Topological Knot' metaphor using GUDHI.
Calculates Betti numbers and Persistent Entropy to quantify semantic complexity.

The "Topological Knot" represents a stable, localized semantic structure that
resists deformation. In this architecture, semantic ambiguities (polysemy,
paradoxes) are treated as topological features—specifically, high-dimensional
holes or cycles in the data manifold.

Mathematical Background:
    - Persistent Homology: Tracks topological features across scales
    - Betti Numbers: β₀ = clusters, β₁ = loops, β₂ = voids
    - Persistent Entropy: Shannon entropy of persistence diagram lifetimes
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import gudhi
except ImportError:
    gudhi = None

from qnsc.config import TopologyConfig
from qnsc.types import TopologyMetrics


class TopologicalKnotDetector(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Implements the 'Topological Knot' metaphor using GUDHI.

    Calculates Betti numbers and Persistent Entropy to quantify semantic complexity.
    Compatible with scikit-learn Pipeline interface.

    Attributes:
        max_edge_length: The filtration threshold (epsilon) for Rips complex
        max_dimension: Maximum homology dimension to compute

    Example:
        >>> detector = TopologicalKnotDetector(max_edge_length=2.0)
        >>> metrics = detector.fit_transform(semantic_point_cloud)
        >>> print(f"Topological Entropy: {metrics['persistent_entropy']}")
    """

    def __init__(
        self,
        max_edge_length: float = 1.0,
        max_dimension: int = 2,
        config: TopologyConfig | None = None
    ) -> None:
        """
        Initialize the Topological Knot Detector.

        Args:
            max_edge_length: The filtration threshold (epsilon).
                             Larger values capture larger-scale features.
            max_dimension: The maximum homology dimension to compute.
                           0=clusters, 1=loops (semantic ambiguities), 2=voids.
            config: Optional TopologyConfig for parameter override.
        """
        if config is not None:
            self.max_edge_length = config.max_edge_length
            self.max_dimension = config.max_dimension
        else:
            self.max_edge_length = max_edge_length
            self.max_dimension = max_dimension

        if gudhi is None:
            raise ImportError(
                "GUDHI is required for topological analysis. "
                "Install with: pip install gudhi"
            )

    def _compute_persistence_entropy(
        self,
        persistence_intervals: list[tuple[float, float]]
    ) -> float:
        """
        Calculate Shannon entropy of the persistence diagram.

        H = -Σ(pᵢ * log₂(pᵢ)), where pᵢ is relative lifetime of feature i.

        A high entropy indicates a rich, complex topology with multiple
        significant features (a complex knot). Low entropy indicates
        a topologically simple space (a trivial knot).

        Args:
            persistence_intervals: List of (birth, death) tuples from GUDHI.

        Returns:
            Persistent entropy value (non-negative float).
        """
        if len(persistence_intervals) == 0:
            return 0.0

        # Calculate lifetimes: death - birth
        lifetimes: list[float] = []

        for birth, death in persistence_intervals:
            # Handle infinite death times (features that persist indefinitely)
            # In a Rips filtration, we cap at max_edge_length
            if death == float('inf'):
                death = self.max_edge_length

            lifetime = death - birth
            if lifetime > 0:
                lifetimes.append(lifetime)

        if len(lifetimes) == 0:
            return 0.0

        lifetimes_arr = np.array(lifetimes)
        total_lifetime = np.sum(lifetimes_arr)

        if total_lifetime == 0:
            return 0.0

        # Normalize to probability distribution
        probs = lifetimes_arr / total_lifetime

        # Calculate Shannon Entropy (base 2)
        # Use np.where to avoid log(0)
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        return entropy

    def fit(self, embeddings: NDArray[np.float64], y: Any = None) -> "TopologicalKnotDetector":
        """
        Fit method for sklearn compatibility.

        Args:
            X: Point cloud data (N_samples, N_features).
            y: Ignored (sklearn convention).

        Returns:
            Self for method chaining.
        """
        # No fitting required - this is a stateless transformer
        return self

    def transform(self, embeddings: NDArray[np.float64]) -> TopologyMetrics:
        """
        Compute topological features of the point cloud.

        Args:
            X: Point cloud data (N_samples, N_features).
               Could be word embeddings, sentence vectors, or attention states.

        Returns:
            TopologyMetrics dictionary containing:
                - betti_numbers: List of Betti numbers
                - persistent_entropy: Entropy of persistence diagram
                - simplex_tree_dim: Dimension of computed complex
                - num_simplices: Total simplices in complex
        """
        return self._compute_topology(embeddings)

    def fit_transform(self, embeddings: NDArray[np.float64], y: Any = None) -> TopologyMetrics:
        """
        Compute topological features (combined fit and transform).

        Args:
            X: Point cloud data (N_samples, N_features).
            y: Ignored (sklearn convention).

        Returns:
            TopologyMetrics dictionary.
        """
        return self._compute_topology(embeddings)

    def _compute_topology(self, embeddings: NDArray[np.float64]) -> TopologyMetrics:
        """
        Internal method to compute all topological features.

        Pipeline:
            1. Construct Rips Complex from point cloud
            2. Create Simplex Tree data structure
            3. Compute Persistence pairs
            4. Extract Betti numbers
            5. Calculate Persistent Entropy for dimension 1 (loops)
        """
        # Validate input
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got {embeddings.ndim}D")

        if embeddings.shape[0] < 2:
            # Not enough points for meaningful topology
            return TopologyMetrics(
                betti_numbers=[embeddings.shape[0]],
                persistent_entropy=0.0,
                simplex_tree_dim=0,
                num_simplices=embeddings.shape[0]
            )

        # 1. Construct Rips Complex
        # The RipsComplex builds the neighborhood graph based on distance
        rips_complex = gudhi.RipsComplex(
            points=embeddings.tolist(),
            max_edge_length=self.max_edge_length
        )

        # 2. Create Simplex Tree
        # The SimplexTree is the core data structure for filtration
        # We limit dimension to avoid combinatorial explosion
        simplex_tree = rips_complex.create_simplex_tree(
            max_dimension=self.max_dimension
        )

        # 3. Compute Persistence
        # This triggers the computation of the persistence pairs
        simplex_tree.compute_persistence()

        # 4. Extract Betti Numbers
        # Betti numbers at the end of filtration (max_edge_length)
        betti_numbers = simplex_tree.betti_numbers()

        # 5. Extract Persistence Intervals for Entropy
        # We focus on Dimension 1 (loops) as they represent semantic paradoxes/cycles
        persistence_intervals_dim1 = simplex_tree.persistence_intervals_in_dimension(1)

        # Convert to list of tuples for entropy calculation
        intervals_list = [
            (float(interval[0]), float(interval[1]))
            for interval in persistence_intervals_dim1
        ]

        entropy = self._compute_persistence_entropy(intervals_list)

        return TopologyMetrics(
            betti_numbers=list(betti_numbers),
            persistent_entropy=entropy,
            simplex_tree_dim=simplex_tree.dimension(),
            num_simplices=simplex_tree.num_simplices()
        )

    def get_knot_complexity(self, metrics: TopologyMetrics) -> str:
        """
        Interpret topological metrics as semantic knot complexity.

        Args:
            metrics: Output from fit_transform.

        Returns:
            Complexity category: 'trivial', 'simple', or 'complex'.
        """
        entropy = metrics["persistent_entropy"]

        if entropy < 0.5:
            return "trivial"
        elif entropy < 2.0:
            return "simple"
        else:
            return "complex"
