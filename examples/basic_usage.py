"""
QNSC Basic Usage Example

Demonstrates the complete Quantum-Native Semantic Coherence pipeline:
1. Analyze topology of semantic embeddings
2. Compute quantum kernel
3. Compress to superfluid vacuum state
4. Store in Markov Blanket (if Milvus available)
"""

import numpy as np
from qnsc import QNSCConfig
from qnsc.pipeline import QNSCPipeline


def main() -> None:
    """Run the QNSC pipeline demo."""
    print("=" * 60)
    print("Quantum-Native Semantic Coherence (QNSC) Demo")
    print("=" * 60)

    # Create configuration
    config = QNSCConfig()
    config.verbose = True

    # Initialize pipeline (without storage for this demo)
    pipeline = QNSCPipeline(config=config)

    # Generate synthetic semantic embeddings
    # In practice, these would come from a language model
    np.random.seed(42)

    print("\n1. Creating synthetic semantic embeddings...")
    # Cluster 1: Related concepts
    cluster1 = np.random.randn(5, 4) * 0.3 + np.array([1, 0, 0, 0])
    # Cluster 2: Different topic
    cluster2 = np.random.randn(5, 4) * 0.3 + np.array([0, 1, 0, 0])
    # Mixed: Ambiguous concepts (creates loops)
    mixed = np.random.randn(3, 4) * 0.5

    embeddings = np.vstack([cluster1, cluster2, mixed])
    print(f"   Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Run the pipeline with storage enabled
    print("\n2. Running QNSC pipeline (with storage)...")
    result = pipeline.process(embeddings, store=True, metadata={"topic_tag": "demo_run"})

    # Display topology results
    print("\n3. Topology Analysis (Knot Detection):")
    print(f"   Betti numbers: {result.topology['betti_numbers']}")
    print(f"   β₀ (clusters): {result.topology['betti_numbers'][0]}")
    if len(result.topology['betti_numbers']) > 1:
        print(f"   β₁ (loops): {result.topology['betti_numbers'][1]}")
    print(f"   Persistent entropy: {result.topology['persistent_entropy']:.4f}")

    # Display vacuum state
    print("\n4. Superfluid Vacuum State:")
    print(f"   Number of sites: {result.vacuum_state.num_sites}")
    print(f"   Entanglement entropy: {result.vacuum_state.entanglement_entropy:.4f}")

    # Demonstrate Search
    print("\n5. Searching Markov Blanket:")
    # Use the first embedding as a query
    query = embeddings[0]
    search_results = pipeline.search(query, limit=3)

    for i, res in enumerate(search_results):
        print(f"   Result {i+1}: score={res.score:.4f}, entropy={res.entropy_level:.4f}, id={res.id}")

    # Summary
    print("\n6. Pipeline Summary:")
    print(f"   Embeddings processed: {result.metadata['num_embeddings']}")
    print(f"   Stored in DB: {result.stored}")

    # Cleanup
    pipeline.close()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
