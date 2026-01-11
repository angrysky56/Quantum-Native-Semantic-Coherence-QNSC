"""
Markov Blanket Store - Vector Database Interface

Implements the 'Markov Blanket' metaphor using Milvus.
Enforces statistical boundaries via Hybrid Search (Dense Quantum + Sparse Context).

Mathematical Background:
    - Markov Blanket: MB(X) - the boundary rendering X independent of the rest
    - Hybrid Search: Combines dense (quantum fidelity) + sparse (keyword) similarity
    - HNSW Index: Graph topology simulating Small World networks
    - Inner Product Metric: Aligns with quantum state overlap/fidelity

The Markov Blanket is formalized as the set of external vectors satisfying
specific geometric (IP similarity) and logical (metadata filter) criteria.
"""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

try:
    from pymilvus import (
        AnnSearchRequest,
        DataType,
        MilvusClient,
        RRFRanker,
    )
except ImportError:
    MilvusClient = None
    DataType = None
    AnnSearchRequest = None
    RRFRanker = None

from qnsc.config import StorageConfig
from qnsc.types import BlanketSearchResult


class MarkovBlanketStore:
    """
    Implements the 'Markov Blanket' metaphor using Milvus.

    Enforces statistical boundaries via Hybrid Search combining:
    - Dense vectors: Quantum state / superfluid embeddings (IP metric)
    - Sparse vectors: Contextual keywords (BM25/SPLADE style)
    - Metadata filters: Structural constraints (topic, entropy level)

    The HNSW index topology mirrors the Markov Blanket concept:
    search explores neighbors of neighbors (local graph structure)
    to minimize the global energy function (distance).

    Attributes:
        client: Milvus client instance
        collection_name: Name of the vector collection
        dim: Dense vector dimension

    Example:
        >>> store = MarkovBlanketStore(uri="http://localhost:19530", dim=128)
        >>> store.insert_state(dense_vec, sparse_dict, {"topic_tag": "physics"})
        >>> results = store.active_inference_search(query_dense, query_sparse)
    """

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        dim: int = 128,
        config: StorageConfig | None = None
    ) -> None:
        """
        Initialize the Markov Blanket Store.

        Args:
            uri: Milvus server connection URI.
            dim: Dimension of dense semantic vectors.
            config: Optional StorageConfig for parameter override.
        """
        if MilvusClient is None:
            raise ImportError(
                "pymilvus is required for vector storage. "
                "Install with: pip install pymilvus"
            )

        if config is not None:
            self._uri = config.uri
            self.collection_name = config.collection_name
            self.dim = config.dense_dim
            self._hnsw_m = config.hnsw_m
            self._hnsw_ef_construction = config.hnsw_ef_construction
            self._hnsw_ef_search = config.hnsw_ef_search
            self._default_entropy_threshold = config.default_entropy_threshold
            self._top_k = config.top_k
        else:
            self._uri = uri
            self.collection_name = "quantum_semantic_blanket"
            self.dim = dim
            self._hnsw_m = 16
            self._hnsw_ef_construction = 200
            self._hnsw_ef_search = 100
            self._default_entropy_threshold = 0.5
            self._top_k = 5

        self._client: MilvusClient | None = None
        self._collection_exists = False

    def connect(self) -> None:
        """Establish connection to Milvus server."""
        self._client = MilvusClient(uri=self._uri)

    def _ensure_connected(self) -> "MilvusClient":
        """Ensure client is connected."""
        if self._client is None:
            self.connect()
        assert self._client is not None
        return self._client

    def setup_schema(self, drop_existing: bool = False) -> None:
        """
        Define the schema for the Markov Blanket collection.

        Creates:
        - pk: Auto-incrementing primary key
        - quantum_state: Dense vector for superfluid state
        - sparse_context: Sparse vector for keywords
        - entropy_level: Float metadata for filtering
        - topic_tag: String metadata for categorization

        Args:
            drop_existing: If True, drops existing collection first.
        """
        client = self._ensure_connected()

        if client.has_collection(self.collection_name):
            if drop_existing:
                client.drop_collection(self.collection_name)
            else:
                self._collection_exists = True
                return

        # Create schema with dynamic fields enabled
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

        # Primary Key
        schema.add_field(
            field_name="pk",
            datatype=DataType.INT64,
            is_primary=True
        )

        # Dense Field: Stores the compressed Superfluid State (MPS-derived)
        schema.add_field(
            field_name="quantum_state",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dim
        )

        # Sparse Field: Stores contextual keywords (BM25/SPLADE style)
        schema.add_field(
            field_name="sparse_context",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # Metadata fields for Boolean Logic (The structural boundary)
        schema.add_field(
            field_name="entropy_level",
            datatype=DataType.FLOAT
        )
        schema.add_field(
            field_name="topic_tag",
            datatype=DataType.VARCHAR,
            max_length=64
        )

        # Create Indices to optimize the 'Blanket' retrieval
        index_params = client.prepare_index_params()

        # HNSW for Dense: Graph-based index simulating Small World topology
        # M and efConstruction control graph connectivity
        # Metric Type 'IP' (Inner Product) aligns with Quantum Fidelity
        index_params.add_index(
            field_name="quantum_state",
            index_type="HNSW",
            metric_type="IP",
            params={"M": self._hnsw_m, "efConstruction": self._hnsw_ef_construction}
        )

        # Sparse Inverted Index for keyword matching
        index_params.add_index(
            field_name="sparse_context",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # Create the collection with schema and indices
        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

        self._collection_exists = True

    def insert_state(
        self,
        dense_vec: NDArray[np.float64] | list[float],
        sparse_dict: dict[int, float],
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Insert a semantic state into the blanket.

        Args:
            dense_vec: Output from SuperfluidVacuum (flattened/projected).
            sparse_dict: Classical keyword dictionary {hash: weight}.
            metadata: Must include 'entropy_level' and 'topic_tag'.

        Returns:
            Insert result from Milvus.
        """
        client = self._ensure_connected()

        if not self._collection_exists:
            self.setup_schema()

        # Ensure dense_vec is a list
        if isinstance(dense_vec, np.ndarray):
            dense_vec = dense_vec.tolist()

        data = [{
            "quantum_state": dense_vec,
            "sparse_context": sparse_dict,
            **metadata
        }]

        result = client.insert(self.collection_name, data)
        # Flush to ensure data is immediately searchable
        client.flush(self.collection_name)
        return cast(dict[str, Any], result)

    def insert_batch(
        self,
        dense_vecs: list[NDArray[np.float64]] | NDArray[np.float64],
        sparse_dicts: list[dict[int, float]],
        metadata_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Batch insert multiple semantic states.

        Args:
            dense_vecs: List of dense vectors.
            sparse_dicts: List of sparse keyword dicts.
            metadata_list: List of metadata dicts.

        Returns:
            Insert result from Milvus.
        """
        client = self._ensure_connected()

        if not self._collection_exists:
            self.setup_schema()

        data = []
        for i, (dense, sparse, meta) in enumerate(
            zip(dense_vecs, sparse_dicts, metadata_list)
        ):
            if isinstance(dense, np.ndarray):
                dense = dense.tolist()
            data.append({
                "quantum_state": dense,
                "sparse_context": sparse,
                **meta
            })

        result = client.insert(self.collection_name, data)
        client.flush(self.collection_name)
        return cast(dict[str, Any], result)

    def active_inference_search(
        self,
        query_dense: NDArray[np.float64] | list[float],
        query_sparse: dict[int, float] | None = None,
        entropy_threshold: float | None = None,
        topic_filter: str | None = None,
        limit: int | None = None
    ) -> list[BlanketSearchResult]:
        """
        Perform the 'Active Inference' query defining the Markov Blanket.

        Combines quantum similarity (dense) with metadata constraints
        via Hybrid Search using Reciprocal Rank Fusion.

        The blanket boundary is defined by:
        - Geometric: Inner Product similarity (quantum fidelity)
        - Logical: entropy_level < threshold AND topic_tag == filter

        Args:
            query_dense: Query quantum state vector.
            query_sparse: Optional query sparse vector.
            entropy_threshold: Max entropy for filtering (default from config).
            topic_filter: Optional topic tag to filter by.
            limit: Max results to return (default from config).

        Returns:
            List of BlanketSearchResult objects.
        """
        client = self._ensure_connected()

        if not self._collection_exists:
            return []

        entropy_threshold = entropy_threshold or self._default_entropy_threshold
        limit = limit or self._top_k

        # Ensure query_dense is a list
        if isinstance(query_dense, np.ndarray):
            query_dense = query_dense.tolist()

        # Build filter expression
        filter_parts = [f"entropy_level < {entropy_threshold}"]
        if topic_filter:
            filter_parts.append(f'topic_tag == "{topic_filter}"')
        filter_expr = " and ".join(filter_parts)

        # If sparse query provided, use hybrid search
        if query_sparse is not None:
            # Request 1: Dense Quantum Search (Superfluid similarity)
            req_dense = AnnSearchRequest(
                data=[query_dense],
                anns_field="quantum_state",
                param={"metric_type": "IP", "params": {"ef": self._hnsw_ef_search}},
                limit=limit * 2,  # Fetch more for reranking
                expr=filter_expr
            )

            # Request 2: Sparse Context Search (Keyword relevance)
            req_sparse = AnnSearchRequest(
                data=[query_sparse],
                anns_field="sparse_context",
                param={"metric_type": "IP"},
                limit=limit * 2
            )

            # Rerank using Reciprocal Rank Fusion
            # RRF balances dense and sparse without weight tuning
            ranker = RRFRanker()

            results = client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[req_dense, req_sparse],
                ranker=ranker,
                limit=limit,
                output_fields=["*"]
            )
        else:
            # Dense-only search
            results = client.search(
                collection_name=self.collection_name,
                data=[query_dense],
                anns_field="quantum_state",
                search_params={"metric_type": "IP", "params": {"ef": self._hnsw_ef_search}},
                limit=limit,
                filter=filter_expr,
                output_fields=["*"]
            )

        # Parse results into BlanketSearchResult objects
        search_results = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                search_results.append(BlanketSearchResult(
                    id=hit.get("id", 0),
                    score=hit.get("score", 0.0),
                    topic_tag=entity.get("topic_tag", ""),
                    entropy_level=entity.get("entropy_level", 0.0),
                    distance=hit.get("distance", 0.0),
                    metadata=entity
                ))

        return search_results

    def dense_search(
        self,
        query_dense: NDArray[np.float64] | list[float],
        limit: int | None = None,
        filter_expr: str | None = None
    ) -> list[BlanketSearchResult]:
        """
        Simple dense-only search (no hybrid).

        Args:
            query_dense: Query vector.
            limit: Max results.
            filter_expr: Optional filter expression.

        Returns:
            List of search results.
        """
        client = self._ensure_connected()

        if not self._collection_exists:
            return []

        limit = limit or self._top_k

        if isinstance(query_dense, np.ndarray):
            query_dense = query_dense.tolist()

        results = client.search(
            collection_name=self.collection_name,
            data=[query_dense],
            anns_field="quantum_state",
            search_params={"metric_type": "IP", "params": {"ef": self._hnsw_ef_search}},
            limit=limit,
            filter=filter_expr,
            output_fields=["*"]
        )

        search_results = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                search_results.append(BlanketSearchResult(
                    id=hit.get("id", 0),
                    score=hit.get("score", 0.0),
                    topic_tag=entity.get("topic_tag", ""),
                    entropy_level=entity.get("entropy_level", 0.0),
                    distance=hit.get("distance", 0.0),
                    metadata=entity
                ))

        return search_results

    def get_collection_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        client = self._ensure_connected()

        if not self._collection_exists:
            return {"exists": False}

        stats = client.get_collection_stats(self.collection_name)
        return {"exists": True, **stats}

    def drop_collection(self) -> None:
        """Drop the collection."""
        client = self._ensure_connected()

        if client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)

        self._collection_exists = False

    def close(self) -> None:
        """Close the Milvus connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
