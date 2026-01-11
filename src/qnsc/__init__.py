"""
Quantum-Native Semantic Coherence (QNSC) System

A physics-based semantic processing architecture integrating:
- Topological Data Analysis (TDA) via GUDHI
- Quantum Machine Learning (QML) via Qiskit
- Tensor Networks (TN) via Quimb
- Vector Database Management via Milvus

The system operationalizes a model where meaning behaves as a dynamic,
entangled state governed by topological constraints and quantum mechanical evolution.
"""

__version__ = "0.1.0"
__author__ = "QNSC Team"

from qnsc.config import QNSCConfig
from qnsc.types import TopologyMetrics, VacuumState

__all__ = [
    "QNSCConfig",
    "TopologyMetrics",
    "VacuumState",
    "__version__",
]
