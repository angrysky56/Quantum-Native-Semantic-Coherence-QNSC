"""
QNSC Pipeline Coordinator

Orchestrates the complete QNSC data flow:
    1. Input: Raw semantic data (text) → embeddings
    2. Topological Preprocessing: GUDHI → Betti numbers, entropy
    3. Quantum Embedding: Qiskit → Hilbert space projection
    4. Superfluid Compression: Quimb → MPS state
    5. Markov Storage/Retrieval: Milvus → Hybrid search

The pipeline forms a closed loop where retrieved context can be
fed back into the Topological Detector as new input.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from qnsc.config import QNSCConfig
from qnsc.quantum import QuantumSemanticProjector
from qnsc.storage import MarkovBlanketStore
from qnsc.tensor import SuperfluidVacuum
from qnsc.topology import TopologicalKnotDetector
from qnsc.types import BlanketSearchResult, PipelineResult, TopologyMetrics, VacuumState

logger = logging.getLogger(__name__)


class QNSCPipeline:
    """
    Orchestrates the complete QNSC semantic processing pipeline.

    The pipeline is not linear but cyclic - output from the Markov Blanket
    (retrieved context) can be fed back into the Topological Detector.

    Data Flow:
        1. Input embeddings → TopologicalKnotDetector
        2. Topology metrics → QuantumSemanticProjector (tunes circuit)
        3. Quantum kernel → SuperfluidVacuum (MPS compression)
        4. Vacuum state → MarkovBlanketStore (storage/retrieval)

    Example:
        >>> pipeline = QNSCPipeline()
        >>> result = pipeline.process(semantic_embeddings)
        >>> print(f"Coherence: {result.vacuum_state.entanglement_entropy}")
        >>>
        >>> # Search for similar states
        >>> results = pipeline.search(query_embedding)
    """

    def __init__(
        self,
        config: QNSCConfig | None = None,
        connect_storage: bool = False
    ) -> None:
        """
        Initialize the QNSC Pipeline.

        Args:
            config: Configuration for all components.
            connect_storage: If True, connect to Milvus immediately.
        """
        self.config = config or QNSCConfig()

        # Initialize components (lazy loading for optional imports)
        self._knot_detector: TopologicalKnotDetector | None = None
        self._projector: QuantumSemanticProjector | None = None
        self._vacuum: SuperfluidVacuum | None = None
        self._store: MarkovBlanketStore | None = None

        # Cache for topology metrics (used to tune quantum circuit)
        self._last_topology: TopologyMetrics | None = None

        # Cache PCA model so search queries are projected into same manifold
        self._pca_model: PCA | None = None

        if connect_storage:
            self._init_storage()

    def _init_topology(self) -> TopologicalKnotDetector:
        """Lazy init for topology detector."""
        if self._knot_detector is None:
            self._knot_detector = TopologicalKnotDetector(
                config=self.config.topology
            )
        return self._knot_detector

    def _init_quantum(
        self,
        feature_dim: int,
        topology: TopologyMetrics | None = None
    ) -> QuantumSemanticProjector:
        """Lazy init for quantum projector."""
        if self._projector is None or self._projector.feature_dimension != feature_dim:
            self._projector = QuantumSemanticProjector(
                feature_dimension=feature_dim,
                topology_metrics=topology,
                config=self.config.quantum
            )
        elif topology is not None:
            self._projector.update_topology(topology)
        return self._projector

    def _init_vacuum(self, num_sites: int) -> SuperfluidVacuum:
        """Lazy init for superfluid vacuum."""
        if self._vacuum is None or self._vacuum.num_sites != num_sites:
            self._vacuum = SuperfluidVacuum(
                num_sites=num_sites,
                config=self.config.tensor
            )
        return self._vacuum

    def _init_storage(self) -> MarkovBlanketStore:
        """Lazy init for storage."""
        if self._store is None:
            self._store = MarkovBlanketStore(config=self.config.storage)
            self._store.connect()
            self._store.setup_schema()
        return self._store

    def analyze_topology(
        self,
        embeddings: NDArray[np.float64]
    ) -> TopologyMetrics:
        """
        Analyze the topological structure of semantic embeddings.

        Args:
            embeddings: Point cloud of semantic vectors (N, D).

        Returns:
            TopologyMetrics with Betti numbers and entropy.
        """
        detector = self._init_topology()
        metrics = detector.fit_transform(embeddings)
        self._last_topology = metrics

        if self.config.verbose:
            logger.info(
                f"Topology: β={metrics['betti_numbers']}, "
                f"H={metrics['persistent_entropy']:.4f}"
            )

        return metrics

    def compute_quantum_kernel(
        self,
        embeddings: NDArray[np.float64],
        topology: TopologyMetrics | None = None
    ) -> NDArray[np.float64]:
        """
        Compute quantum kernel matrix for embeddings.

        Args:
            embeddings: Semantic vectors (N, D).
            topology: Optional topology metrics to tune circuit.

        Returns:
            Kernel matrix (N, N) of quantum fidelities.
        """
        feature_dim = embeddings.shape[1]

        # Apply PCA reduction if input dimension exceeds quantum feature capacity
        # Quantum kernels are classically simulated here, so we must limit dimension
        # to avoid O(2^N) exponential complexity explosion.
        if feature_dim > self.config.quantum.feature_dimension:
            if self.config.verbose:
                logger.info(
                    f"Coupling classical embeddings ({feature_dim}d) to quantum "
                    f"manifold ({self.config.quantum.feature_dimension}d) via PCA"
                )

            # Simple PCA projection to feature dimension
            if self._pca_model is None:
                self._pca_model = PCA(n_components=self.config.quantum.feature_dimension)
                embeddings = self._pca_model.fit_transform(embeddings)
            else:
                embeddings = self._pca_model.transform(embeddings)

            feature_dim = self.config.quantum.feature_dimension

        projector = self._init_quantum(feature_dim, topology or self._last_topology)
        result = projector.compute_kernel_matrix(embeddings)

        if self.config.verbose:
            logger.info(
                f"Quantum kernel: reps={result['circuit_reps']}, "
                f"entanglement={result['entanglement_type']}"
            )

        return result["kernel_matrix"]

    def compress_to_vacuum(
        self,
        state_vector: NDArray[np.complex128 | np.float64],
        num_sites: int | None = None
    ) -> VacuumState:
        """
        Compress a state vector into MPS (superfluid) form.

        Args:
            state_vector: Raw state vector (must be 2^num_sites dim).
            num_sites: Number of sites (inferred from vector if None).

        Returns:
            VacuumState with compressed representation.
        """
        if num_sites is None:
            # Infer from vector size
            size = state_vector.size
            num_sites = int(np.log2(size))
            if 2 ** num_sites != size:
                raise ValueError(
                    f"State vector size {size} is not a power of 2"
                )

        vacuum = self._init_vacuum(num_sites)
        vacuum.initialize_from_dense(state_vector)

        if self.config.verbose:
            logger.info(
                f"Vacuum compressed: χ={vacuum.get_bond_dimensions()}, "
                f"S={vacuum.get_coherence_measure():.4f}"
            )

        return vacuum.to_vacuum_state()

    def process(
        self,
        embeddings: NDArray[np.float64],
        store: bool = False,
        metadata: list[dict[str, Any]] | dict[str, Any] | None = None
    ) -> PipelineResult:
        """
        Run the complete QNSC pipeline on semantic embeddings.

        Pipeline stages:
            1. Topology analysis (Betti numbers, entropy)
            2. Quantum kernel computation
            3. State vector extraction (principal eigenvector of kernel)
            4. MPS compression (superfluid vacuum)
            5. Optional storage in Markov Blanket

        Args:
            embeddings: Semantic vectors (N, D) as point cloud.
            store: If True, store result in Milvus.
            metadata: Optional metadata for storage. Can be a list for batch or single dict.
                      Must include 'topic_tag'.

        Returns:
            PipelineResult with topology metrics and vacuum state.
        """
        if self.config.verbose:
            logger.info("Processing batch of %d embeddings", len(embeddings))

        # Stage 1: Topology (Global context)
        # We still compute global topology to understand the "shape" of the dataset
        topology = self.analyze_topology(embeddings)

        # Stage 2: Quantum Kernel Projection (Classically simulated via PCA coupling if needed)
        # We transform the *individual* embeddings into the quantum feature space
        # Note: compute_quantum_kernel normally returns N x N kernel.
        # For batch processing, we want the projected feature vectors or state vectors.
        # But our Projector is kernel-based.

        # To get distinct states per document, we can treat each row of the kernel matrix
        # as a representation of that document's relation to the batch (Kernel PCA-like).
        # Compute kernel (side effect: fits PCA model if needed)
        _ = self.compute_quantum_kernel(embeddings, topology)

        # Stage 3: Superfluid Compression per Document
        # We treat each document as its own quantum state.

        vacuum_states = []
        stored_count = 0

        # Note: compute_quantum_kernel fits the PCA model internally.
        # We re-project embeddings below using self._pca_model.transform()

        # Refactor: We need to access the projected embeddings from compute_quantum_kernel
        # or do projection here.
        # Since compute_quantum_kernel is a "black box" returning a kernel matrix,
        # let's replicate the projection logic OR rely on the fact that if we call
        # projector.get_state_vector, we need 8-dim input.

        # Let's assume compute_quantum_kernel fitted the PCA.
        # We can re-transform here using self._pca_model

        embedding_dim_current = embeddings.shape[1]
        target_dim = self.config.quantum.feature_dimension

        x_projected = embeddings
        if embedding_dim_current > target_dim and self._pca_model is not None:
             x_projected = self._pca_model.transform(embeddings)

        # Initialize projector if not already (it should be from stage 2)
        projector = self._init_quantum(target_dim, topology)

        for i in range(len(embeddings)):
            # Get explicit state vector from the Projector
            # maps 8-dim feature -> 256-dim complex vector
            vec_input = x_projected[i]
            psi = projector.get_state_vector(vec_input)

            # Pad to power of 2 for MPS?
            # Statevector from 8 qubits is size 256 (already power of 2).
            # If feature_dim=8, size=256.

            # Compress to vacuum
            n_sites = target_dim # 8 sites for 8 qubits
            vacuum = self.compress_to_vacuum(psi, n_sites)
            vacuum_states.append(vacuum)

            if (i+1) % 5 == 0 or (i+1) == len(embeddings):
                 print(f"  Quantum Processing: {i+1}/{len(embeddings)} docs...", end="\r")

            # Stage 5: Storage (Per document)
            if store:
                # Prepare metadata for this specific document
                doc_meta = metadata[i] if metadata and isinstance(metadata, list) and i < len(metadata) else {}

                # Combine with topological insights
                state_data = vacuum.dense_vector # This is the Compressed Semantic State

                # Sparse features
                sparse = {
                    1: float(topology["betti_numbers"][0]) if topology["betti_numbers"] else 0.0,
                    2: vacuum.entanglement_entropy,
                }

                # Full Metadata
                full_meta = {
                    "entropy_level": vacuum.entanglement_entropy,
                    "global_entropy": topology["persistent_entropy"],
                    **doc_meta
                }

                if self._store_state_direct(state_data, sparse, full_meta):
                    stored_count += 1

        return PipelineResult(
            topology=topology,
            vacuum_state=vacuum_states[0] if vacuum_states else None,
            stored=stored_count > 0,
            metadata={
                "num_processed": len(embeddings),
                "num_stored": stored_count,
            }
        )

    def _store_state_direct(
        self,
        dense_vector: NDArray[np.float64],
        sparse_dict: dict[int, float],
        metadata: dict[str, Any]
    ) -> bool:
        """Helper to store a single prepared state directly."""
        try:
            store = self._init_storage()

            # Ensure dimension match
            dense = dense_vector
            if len(dense) > self.config.storage.dense_dim:
                dense = dense[:self.config.storage.dense_dim]
            elif len(dense) < self.config.storage.dense_dim:
                dense = np.pad(dense, (0, self.config.storage.dense_dim - len(dense)))

            store.insert_state(dense, sparse_dict, metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to store state: {e}")
            return False

    def _store_state(
        self,
        vacuum_state: VacuumState,
        topology: TopologyMetrics,
        metadata: dict[str, Any] | None
    ) -> bool:
        """Store vacuum state in Markov Blanket."""
        try:
            store = self._init_storage()

            # Truncate/pad dense vector to storage dimension
            dense = vacuum_state.dense_vector
            if len(dense) > self.config.storage.dense_dim:
                dense = dense[:self.config.storage.dense_dim]
            elif len(dense) < self.config.storage.dense_dim:
                dense = np.pad(
                    dense,
                    (0, self.config.storage.dense_dim - len(dense))
                )

            # Create sparse context from topology (use positive indices only)
            # Milvus requires sparse indices to be positive < 2^32-1
            sparse = {
                1: float(topology["betti_numbers"][0]) if topology["betti_numbers"] else 0.0,
                2: topology["persistent_entropy"],
            }

            # Prepare metadata
            meta = {
                "entropy_level": topology["persistent_entropy"],
                "topic_tag": metadata.get("topic_tag", "unknown") if metadata else "unknown",
            }

            store.insert_state(dense, sparse, meta)
            return True

        except Exception as e:
            logger.error(f"Failed to store state: {e}")
            return False

    def search(
        self,
        query_embedding: NDArray[np.float64],
        limit: int | None = None,
        entropy_threshold: float | None = None
    ) -> list[BlanketSearchResult]:
        """
        Search the Markov Blanket for similar semantic states.

        Args:
            query_embedding: Query vector.
            limit: Max results to return.
            entropy_threshold: Max entropy filter.

        Returns:
            List of matching BlanketSearchResult objects.
        """
        store = self._init_storage()

        # Ensure query is correct dimension
        query = np.asarray(query_embedding).flatten()

        # Apply PCA to query if model exists
        if self._pca_model is not None:
             query_reshaped = query.reshape(1, -1)
             if query_reshaped.shape[1] == self._pca_model.n_features_in_:
                 query = self._pca_model.transform(query_reshaped).flatten()

        # Project to Quantum State AND compress to Vacuum (matching storage pipeline)
        # Storage path: Embed -> PCA -> Quantum State -> MPS Compression -> dense_vector
        if self._projector:
            try:
                # Get complex state vector (256d for 8 qubits)
                psi = self._projector.get_state_vector(query)
                # Apply MPS compression (same as storage path)
                vacuum_state = self.compress_to_vacuum(psi, self.config.quantum.feature_dimension)
                # Use the same dense_vector format as stored documents
                query = vacuum_state.dense_vector

                # DEBUG: Check query vector stats
                if self.config.verbose:
                    print(f"DEBUG: Search Query Vector Norm: {np.linalg.norm(query)}")
                    print(f"DEBUG: Search Query First 5: {query[:5]}")

            except Exception as e:
                logger.warning(f"Failed to project query to quantum state: {e}. Using raw PCA vector.")

        if len(query) > self.config.storage.dense_dim:
            query = query[:self.config.storage.dense_dim]
        elif len(query) < self.config.storage.dense_dim:
            query = np.pad(query, (0, self.config.storage.dense_dim - len(query)))

        results = store.dense_search(
            query_dense=query,
            limit=limit,
            filter_expr=f"entropy_level < {entropy_threshold}" if entropy_threshold else None
        )

        # DEBUG: Check result scores
        if self.config.verbose and results:
             print(f"DEBUG: First Result Score: {results[0].score}")

        return results

    def close(self) -> None:
        """Close all connections."""
        if self._store is not None:
            self._store.close()
