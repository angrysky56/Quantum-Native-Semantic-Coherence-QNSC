"""Pytest configuration for QNSC tests."""
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_milvus: mark test as requiring Milvus server"
    )
