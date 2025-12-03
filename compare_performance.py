import time
import os
import shutil
import numpy as np
from src.config import settings
from src.pipeline.ingest import IngestionPipeline
from src.retrieval.search import HierarchicalRetriever
import faiss
import uuid
from typing import List, Dict, Any
from src.core.embedding import EmbeddingModel


class FlatRAG:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.index = None
        self.nodes = []
        self.dimension = 384  # all-MiniLM-L6-v2 dimension

    def ingest(self, documents: List[str]):
        # Chunking (Same logic as IngestionPipeline to be fair)
        chunks = []
        for doc in documents:
            chunk_size = 500
            overlap = 50
            if len(doc) <= chunk_size:
                chunks.append(doc)
            else:
                for i in range(0, len(doc) - overlap, chunk_size - overlap):
                    chunks.append(doc[i : i + chunk_size])

        # Encode (Float32 always)
        embeddings = self.embedder.encode(chunks, quantize=False)

        # Store nodes in memory (simulating a simple DB)
        self.nodes = []
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            self.nodes.append({"id": str(uuid.uuid4()), "text": text, "embedding": emb})

        # Build Faiss Index
        self.index = faiss.IndexFlatL2(self.dimension)
        if embeddings is not None and len(embeddings) > 0:
            matrix = np.array(embeddings).astype("float32")
            self.index.add(matrix)

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.index or self.index.ntotal == 0:
            return []

        query_emb = self.embedder.encode(query_text, quantize=False)
        query_vec = np.array([query_emb]).astype("float32")

        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                node = self.nodes[idx]
                results.append(
                    {"id": node["id"], "text": node["text"], "score": float(dist)}
                )
        return results


def get_dir_size(path):
    total = 0
    if os.path.exists(path):
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def main():
    # 1. Prepare Data
    print("Fetching 20 Newsgroups dataset (this may take a moment)...")
    from sklearn.datasets import fetch_20newsgroups

    # Load dataset, removing metadata to make it harder/more realistic
    newsgroups = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )

    # Filter out empty or very short documents
    full_docs = [text for text in newsgroups.data if len(text) > 100]

    # Limit to 5000 documents as requested for reasonable test duration
    target_count = 5000
    if len(full_docs) > target_count:
        full_docs = full_docs[:target_count]

    print(f"Dataset size: {len(full_docs)} real documents from 20 Newsgroups")

    # --- Baseline: Flat RAG ---
    print("\n=== Running Baseline: Flat RAG ===")
    flat_rag = FlatRAG()

    start_time = time.time()
    flat_rag.ingest(full_docs)
    flat_ingest_time = time.time() - start_time
    print(f"Flat RAG Ingestion Time: {flat_ingest_time:.4f}s")

    # Calculate Flat Storage (Memory approximation)
    # 384 floats * 4 bytes * num_chunks
    num_chunks = flat_rag.index.ntotal
    flat_storage_bytes = num_chunks * 384 * 4
    print(f"Flat RAG Index Size (Approx RAM): {flat_storage_bytes / 1024:.2f} KB")

    start_time = time.time()
    flat_results = flat_rag.query("machine learning types", k=5)
    print(f"Flat RAG result: {flat_results}")
    flat_query_time = time.time() - start_time
    print(f"Flat RAG Query Time: {flat_query_time:.4f}s")

    # --- Proposed: HiRAG (Variable Quantization) ---
    print("\n=== Running Proposed: HiRAG ===")
    # Clean up
    if os.path.exists(settings.DATA_DIR):
        for f in os.listdir(settings.DATA_DIR):
            if f != "embedding_cache":
                path = settings.DATA_DIR / f
                if path.is_file():
                    os.remove(path)
    if os.path.exists(settings.INDEX_DIR):
        shutil.rmtree(settings.INDEX_DIR)
        os.makedirs(settings.INDEX_DIR)

    hirag_pipeline = IngestionPipeline()

    start_time = time.time()
    hirag_pipeline.ingest(full_docs)
    hirag_ingest_time = time.time() - start_time
    print(f"HiRAG Ingestion Time: {hirag_ingest_time:.4f}s")

    # Re-initialize Retriever to load the newly created indices
    hirag_retriever = HierarchicalRetriever()

    # Calculate HiRAG Storage (Disk)
    # We measure the size of the 'indices' folder + 'data' folder (metadata)
    # Note: This includes metadata overhead which FlatRAG didn't count fully,
    # but we are interested in the Vector Index size mainly.
    # Let's look at the .index files specifically for a fair comparison of "Index Size"
    hirag_index_size = 0
    for f in os.listdir(settings.INDEX_DIR):
        if f.endswith(".index"):
            hirag_index_size += (settings.INDEX_DIR / f).stat().st_size

    print(f"HiRAG Vector Index Size (Disk): {hirag_index_size / 1024:.2f} KB")

    start_time = time.time()
    hirag_results = hirag_retriever.query("machine learning types", top_k=5)
    print(f"HiRAG result: {hirag_results}")
    hirag_query_time = time.time() - start_time
    print(f"HiRAG Query Time: {hirag_query_time:.4f}s")

    # --- Comparison Summary ---
    print("\n=== Comparison Summary ===")
    print(f"{'Metric':<20} | {'Flat RAG':<15} | {'HiRAG':<15} | {'Improvement':<15}")
    print("-" * 70)
    print(
        f"{'Ingestion Time':<20} | {flat_ingest_time:.4f}s       | {hirag_ingest_time:.4f}s       | {flat_ingest_time / hirag_ingest_time:.2f}x (Slower is expected)"
    )
    print(
        f"{'Index Size':<20} | {flat_storage_bytes / 1024:.2f} KB       | {hirag_index_size / 1024:.2f} KB       | {flat_storage_bytes / hirag_index_size:.2f}x Smaller"
    )
    print(
        f"{'Query Time':<20} | {flat_query_time:.4f}s       | {hirag_query_time:.4f}s       | {flat_query_time / hirag_query_time:.2f}x Faster"
    )


if __name__ == "__main__":
    main()
