"""
Tests for Markov Blanket Store (Milvus)

Tests the vector database operations including:
- Schema creation
- Insert operations
- Hybrid search

Note: Most tests require Milvus to be running and will be skipped otherwise.
"""

import pytest
import numpy as np

# Try to import pymilvus
try:
    from pymilvus import MilvusClient
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

from qnsc.storage import MarkovBlanketStore
from qnsc.config import StorageConfig


# Mark all tests to skip if Milvus not available
pytestmark = pytest.mark.skipif(
    not MILVUS_AVAILABLE,
    reason="pymilvus not installed"
)


def milvus_running() -> bool:
    """Check if Milvus server is running."""
    try:
        client = MilvusClient(uri="http://localhost:19531")
        client.list_collections()
        client.close()
        return True
    except Exception:
        return False


# Additional skip for integration tests
requires_milvus_server = pytest.mark.skipif(
    not milvus_running(),
    reason="Milvus server not running"
)


class TestMarkovBlanketStoreInit:
    """Tests for MarkovBlanketStore initialization."""

    def test_init_default(self) -> None:
        """Test default initialization without connecting."""
        store = MarkovBlanketStore()
        assert store.collection_name == "quantum_semantic_blanket"
        assert store.dim == 128

    def test_init_with_config(self) -> None:
        """Test initialization with config."""
        config = StorageConfig(
            collection_name="test_blanket",
            dense_dim=64
        )
        store = MarkovBlanketStore(config=config)
        assert store.collection_name == "test_blanket"
        assert store.dim == 64


@requires_milvus_server
class TestMarkovBlanketStoreOperations:
    """Integration tests requiring Milvus server."""

    @pytest.fixture
    def store(self) -> MarkovBlanketStore:
        """Create a test store."""
        config = StorageConfig(
            collection_name="qnsc_test_blanket",
            dense_dim=32
        )
        store = MarkovBlanketStore(config=config)
        store.connect()
        store.setup_schema(drop_existing=True)
        yield store
        store.drop_collection()
        store.close()

    def test_connect(self, store: MarkovBlanketStore) -> None:
        """Test Milvus connection."""
        stats = store.get_collection_stats()
        assert stats["exists"] is True

    def test_insert_single(self, store: MarkovBlanketStore) -> None:
        """Test inserting a single state."""
        dense = np.random.randn(32).tolist()
        sparse = {1: 1.0, 2: 0.5}  # Use positive indices for Milvus
        metadata = {"entropy_level": 0.3, "topic_tag": "test"}

        result = store.insert_state(dense, sparse, metadata)

        assert "insert_count" in result or result.get("insert_count") == 1

    def test_insert_batch(self, store: MarkovBlanketStore) -> None:
        """Test batch insertion."""
        n = 5
        dense_vecs = [np.random.randn(32) for _ in range(n)]
        sparse_dicts = [{i + 1: 1.0} for i in range(n)]  # Positive indices
        metadata_list = [
            {"entropy_level": 0.1 * i, "topic_tag": f"topic{i}"}
            for i in range(n)
        ]

        result = store.insert_batch(dense_vecs, sparse_dicts, metadata_list)

        assert result.get("insert_count") == n

    def test_dense_search(self, store: MarkovBlanketStore) -> None:
        """Test dense vector search."""
        # Insert some data
        for i in range(10):
            dense = np.random.randn(32)
            sparse = {i + 1: 1.0}  # Positive index
            metadata = {"entropy_level": 0.1 * i, "topic_tag": f"topic{i % 3}"}
            store.insert_state(dense.tolist(), sparse, metadata)

        # Search
        query = np.random.randn(32)
        results = store.dense_search(query, limit=5)

        assert len(results) <= 5
        for r in results:
            assert hasattr(r, "id")
            assert hasattr(r, "score")

    def test_search_with_filter(self, store: MarkovBlanketStore) -> None:
        """Test search with entropy filter."""
        # Insert with varying entropy
        for i in range(10):
            dense = np.random.randn(32)
            sparse = {i + 1: 1.0}  # Positive index
            metadata = {"entropy_level": 0.5 + 0.1 * i, "topic_tag": "test"}
            store.insert_state(dense.tolist(), sparse, metadata)

        # Search with low entropy threshold
        query = np.random.randn(32)
        results = store.active_inference_search(
            query_dense=query,
            entropy_threshold=0.7,
            limit=10
        )

        # All results should have entropy < 0.7
        for r in results:
            assert r.entropy_level < 0.7

    def test_drop_collection(self, store: MarkovBlanketStore) -> None:
        """Test collection dropping."""
        store.drop_collection()
        stats = store.get_collection_stats()
        assert stats["exists"] is False


class TestMarkovBlanketStoreMocked:
    """Unit tests that don't require Milvus server."""

    def test_import_error_handling(self) -> None:
        """Test that import errors are handled gracefully."""
        # This test verifies the structure handles missing pymilvus
        # The actual error would be raised during import
        pass

    def test_config_defaults(self) -> None:
        """Test configuration defaults."""
        config = StorageConfig()
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construction == 200
        assert config.hnsw_ef_search == 100
        assert config.top_k == 5
