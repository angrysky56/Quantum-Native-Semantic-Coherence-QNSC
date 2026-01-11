#!/usr/bin/env python3
"""
Research Reports Data Integration Demo

Ingests markdown research reports, generates embeddings via OpenRouter/OpenAI API,
and processes them through the QNSC pipeline to reveal topological structure.

Usage:
    1. Copy .env.example to .env and set OPENROUTER_API_KEY
    2. Run: python examples/research_demo.py
"""

import glob
import os
import re
import time
import warnings
from typing import Any

import numpy as np

# Suppress GUDHI syntax warnings (invalid escape sequence '\l' in docstrings)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="gudhi")
from dotenv import load_dotenv # noqa: E402, I001
from openai import OpenAI # noqa: E402, I001

# Load environment variables
load_dotenv()

# Add src to path
import sys # noqa: E402, I001
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# QNSC Imports
from qnsc.pipeline import QNSCPipeline  # noqa: E402, I001
from qnsc.config import QNSCConfig # noqa: E402, I001

# Configuration
REPORTS_DIR = os.getenv("REPORTS_DIR_PATH", "/home/ty/Repositories/deep-research-reports")
API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
CHUNK_SIZE = 100000  # Characters (Full paper context for Qwen-8B)

# Initialize OpenAI client
client = None
if API_KEY:
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def get_embedding(text: str) -> list[float]:
    """Fetch embedding from API."""
    if not client:
        # Fallback for testing without API key (random vector)
        # return np.random.randn(1536).tolist() # Standard OpenAI dimension
        raise ValueError("API Key not found. Please set OPENROUTER_API_KEY in .env")

    try:
        # Normalize text
        text = text.replace("\n", " ")

        response = client.embeddings.create(
            model=MODEL,
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"{Colors.RED}Error fetching embedding: {e}{Colors.END}")
        return []

# --- Chunking Logic ---
class MarkdownChunker:
    """Simple semantic splitter for Markdown."""

    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str, metadata: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
        chunks = []
        chunk_metas = []

        # Split by headers (level 2) to get sections
        sections = re.split(r'(^## .+$)', text, flags=re.MULTILINE)

        current_header = "Introduction"

        for i in range(len(sections)):
            section = sections[i].strip()
            if not section:
                continue

            # If line is header, update context
            if section.startswith("## "):
                current_header = section.replace("## ", "").strip()
                continue

            # Content block - split by paragraphs if too long
            if len(section) > self.chunk_size:
                sub_chunks = self._split_paragraphs(section)
                for j, sub in enumerate(sub_chunks):
                    chunks.append(sub)
                    chunk_metas.append({
                        **metadata,
                        "header": current_header,
                        "chunk_index": len(chunks),
                        "chunk_type": "paragraph"
                    })
            else:
                chunks.append(section)
                chunk_metas.append({
                    **metadata,
                    "header": current_header,
                    "chunk_index": len(chunks),
                    "chunk_type": "section"
                })

        return chunks, chunk_metas

    def _split_paragraphs(self, text: str) -> list[str]:
        paras = text.split("\n\n")
        chunks = []
        current_chunk: list[str] = []
        current_len = 0

        for p in paras:
            if current_len + len(p) > self.chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [p]
                current_len = len(p)
            else:
                current_chunk.append(p)
                current_len += len(p)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks

def load_documents(directory: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Load markdown files from directory and chunk them."""
    files = glob.glob(os.path.join(directory, "**/*.md"), recursive=True)

    all_chunks = []
    all_metas = []

    chunker = MarkdownChunker(chunk_size=1500, overlap=100)

    print(f"Found {len(files)} markdown files in {directory}")

    # Process limited number of files for demo speed
    demo_limit = 10
    if len(files) > demo_limit:
        print(f"{Colors.YELLOW}Limiting to {demo_limit} documents (but fully chunked) for demo speed.{Colors.END}")
        files = files[:demo_limit]

    for fpath in files:
        try:
            with open(fpath, encoding='utf-8') as f:
                content = f.read()

            if len(content.strip()) < 50:
                continue

            base_meta = {
                "filename": os.path.basename(fpath),
                "path": fpath,
                "topic_tag": "research_report"
            }

            chunks, metas = chunker.split_text(content, base_meta)

            all_chunks.extend(chunks)
            all_metas.extend(metas)

            print(f"  Loaded {os.path.basename(fpath)} -> {len(chunks)} chunks", end="\r")

        except Exception as e:
            print(f"{Colors.YELLOW}Skipping {fpath}: {e}{Colors.END}")

    print(f"\nTotal: Generated {len(all_chunks)} semantic chunks from {len(files)} files.")
    return all_chunks, all_metas

def main() -> None:
    print_header("QNSC Research Data Integration")

    # 0. Check API Key
    if not API_KEY:
        print(f"{Colors.RED}Error: OPENROUTER_API_KEY not found.{Colors.END}")
        print("Please copy '.env.example' to '.env' and add your API Key.")
        return

    # 1. Load Data
    print(f"{Colors.BOLD}1. Loading Research Reports...{Colors.END}")
    docs, metadatas = load_documents(REPORTS_DIR)

    if not docs:
        print(f"{Colors.RED}No documents found!{Colors.END}")
        return

    print(f"{Colors.GREEN}Loaded {len(docs)} semantic chunks.{Colors.END}")

    # 2. Generate Embeddings
    print(f"\n{Colors.BOLD}2. Generating Embeddings via {MODEL}...{Colors.END}")

    # Inject seq_id for reliable retrieval mapping
    for i, meta in enumerate(metadatas):
        meta['seq_id'] = i

    embeddings = []
    valid_metas = []

    # Cache Setup
    import hashlib
    import pickle
    cache_file = ".embedding_cache.pkl"
    cache = {}

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            print(f"{Colors.GREEN}Loaded {len(cache)} cached embeddings.{Colors.END}")
        except Exception:
            print(f"{Colors.YELLOW}Corrupt cache, starting fresh.{Colors.END}")

    start_time = time.time()
    new_count = 0

    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        print(f"  Processing {i+1}/{len(docs)}: {meta['filename']}...", end="\r")

        # Compute content hash
        doc_hash = hashlib.md5(doc.encode("utf-8")).hexdigest()

        if doc_hash in cache:
            emb = cache[doc_hash]
        else:
            emb = get_embedding(doc)
            if emb:
                cache[doc_hash] = emb
                new_count += 1
                # Rate limit only on API call
                time.sleep(0.2)

        if emb:
            embeddings.append(emb)
            valid_metas.append(meta)

    # Save Cache
    if new_count > 0:
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
        print(f"\n{Colors.GREEN}Updated cache with {new_count} new embeddings.{Colors.END}")

    print(f"\n{Colors.GREEN}Generated {len(embeddings)} embeddings in {time.time() - start_time:.2f}s{Colors.END}")

    if not embeddings:
        print(f"{Colors.RED}Failed to generate any embeddings.{Colors.END}")
        return

    # Convert to numpy
    x = np.array(embeddings)

    # 3. Initialize Pipeline
    print(f"\n{Colors.BOLD}3. Initializing QNSC Pipeline...{Colors.END}")

    os.environ["QNSC_VERBOSE"] = "true" # Enable debug logs
    config = QNSCConfig()
    config.storage.collection_name = "research_knowledge_graph"
    # Auto-detect dimension from API result
    embedding_dim = x.shape[1]
    config.storage.dense_dim = embedding_dim
    # Update feature map dimension for Qiskit (quantum simulation limits)
    # We must reduce dimension for quantum circuit simulation (classically simulated)
    # PCAg is handled automatically by the pipeline if input > feature_dimension
    config.quantum.feature_dimension = 8

    pipeline = QNSCPipeline(config=config, connect_storage=True)

    # Reset collection for demo
    print(f"{Colors.YELLOW}Resetting Knowledge Graph storage...{Colors.END}")
    try:
        if pipeline._store:
            pipeline._store.drop_collection()
            print(f"{Colors.GREEN}Collection cleared.{Colors.END}")
        else:
             print(f"{Colors.RED}Storage not connected.{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Failed to clear collection: {e}{Colors.END}")

    # 4. Process
    print(f"\n{Colors.BOLD}4. Processing through Quantum Pipeline...{Colors.END}")
    print("  (Topology Analysis -> Quantum Projection -> Vacuum Compression -> Storage)")

    result = pipeline.process(x, store=True, metadata=valid_metas)

    # Note: Pipeline process() handles batch storage accurately now.

    # 5. Analysis Results
    print_header("Topological Knowledge Structure")
    topo = result.topology
    print(f"Knowledge Clusters (β₀): {Colors.GREEN}{topo['betti_numbers'][0]}{Colors.END}")
    print(f"Knowledge Loops (β₁):    {Colors.GREEN}{topo['betti_numbers'][1] if len(topo['betti_numbers']) > 1 else 0}{Colors.END}")
    print(f"Semantic Entropy:        {Colors.GREEN}{topo['persistent_entropy']:.4f}{Colors.END}")

    if len(topo['betti_numbers']) > 1 and topo['betti_numbers'][1] > 0:
        print(f"{Colors.YELLOW}Detector found {topo['betti_numbers'][1]} non-trivial conceptual loops!{Colors.END}")
        print("A 'non-trivial loop' (H1 feature) in the semantic space indicates a circular chain of concepts.")
        print("This suggests that starting from one idea and moving to locally similar ones eventually returns to the start,")
        print("forming a logical cycle or feedback loop in the corpus.")
    else:
        print("Knowledge structure is currently hierarchical (tree-like), no logical loops detected.")

    print_header("Quantum-Native Semantic Search")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        query_text = input(f"\n{Colors.BOLD}Enter concept to search > {Colors.END}").strip()

        if query_text.lower() in ['exit', 'quit']:
            break

        if not query_text:
            continue

        q_emb = get_embedding(query_text)
        if not q_emb:
            continue

        # Search
        results = pipeline.search(q_emb, limit=5)

        print(f"\nTop 5 Coherent Matches for '{query_text}':")
        print("-" * 60)

        for i, res in enumerate(results):
            # Map index back to document
            try:
                # Use seq_id from metadata if available (reliable), else fallback to ID
                seq_id = res.metadata.get("seq_id")
                if seq_id is not None:
                    doc_idx = int(seq_id)
                else:
                    doc_idx = res.id # Unreliable fallback

                doc_content = docs[doc_idx]
                doc_meta = metadatas[doc_idx]

                # Snippet (first 300 chars or context window)
                snippet = doc_content[:300].replace("\n", " ") + "..."

                # QNSC Metrics from metadata
                # We inserted keys: 'entropy_level', 'global_entropy', 'topic_tag'
                entropy = res.metadata.get("entropy_level", 0.0)
                # Invert entropy for "Coherence Score" (Low entropy = High coherence)
                coherence = 1.0 / (1.0 + entropy)

                print(f"{Colors.BOLD}{i+1}. {doc_meta['filename']}{Colors.END}")
                print(f"   {Colors.BLUE}Snippet:{Colors.END} \"{snippet}\"")
                print(f"   {Colors.GREEN}Relevance:{Colors.END} {res.score:.4f} | {Colors.YELLOW}Quantum Coherence:{Colors.END} {coherence:.2f} (H={entropy:.2f})")
                print("-" * 60)

            except IndexError:
                print(f"{i+1}. [Unknown Index {res.id}]")

    pipeline.close()
    print(f"\n{Colors.GREEN}Session Ended.{Colors.END}")

if __name__ == "__main__":
    main()
