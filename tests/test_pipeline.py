"""Tests for QNSC Pipeline Coordinator."""
import pytest
import numpy as np

gudhi = pytest.importorskip("gudhi")
qiskit = pytest.importorskip("qiskit")
quimb = pytest.importorskip("quimb")

from qnsc.pipeline import QNSCPipeline
from qnsc.config import QNSCConfig


class TestQNSCPipeline:
    """Tests for pipeline."""

    def test_init(self) -> None:
        pipeline = QNSCPipeline()
        assert pipeline.config is not None

    def test_analyze_topology(self) -> None:
        pipeline = QNSCPipeline()
        np.random.seed(42)
        embeddings = np.random.randn(10, 4)
        metrics = pipeline.analyze_topology(embeddings)
        assert "betti_numbers" in metrics
        assert metrics["persistent_entropy"] >= 0

    def test_process(self) -> None:
        pipeline = QNSCPipeline()
        np.random.seed(42)
        embeddings = np.random.randn(5, 2)
        result = pipeline.process(embeddings, store=False)
        assert result.topology is not None
        assert result.vacuum_state is not None
