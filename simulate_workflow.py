import sys
import os
import shutil
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import settings
from src.pipeline.ingest import IngestionPipeline
from src.retrieval.search import HierarchicalRetriever
from src.storage.metadata_store import MetadataStore
from src.core.schema import QuantizedEmbedding


class WorkflowSimulator:
    def __init__(self):
        self.pipeline = None
        self.retriever = None

    def setup(self, clean: bool = False):
        print("\n[1] System Setup")
        if clean:
            print("    Cleaning existing data...")
            if os.path.exists(settings.DATA_DIR):
                # Don't delete cache if we want to test caching, but for full clean we might.
                # Let's keep cache, delete metadata/indices
                for f in os.listdir(settings.DATA_DIR):
                    if f != "embedding_cache":
                        path = settings.DATA_DIR / f
                        if path.is_file():
                            os.remove(path)
            if os.path.exists(settings.INDEX_DIR):
                shutil.rmtree(settings.INDEX_DIR)
                os.makedirs(settings.INDEX_DIR)
        else:
            print("    Using existing data.")

        # Initialize components AFTER cleanup to ensure fresh state
        self.pipeline = IngestionPipeline()
        self.retriever = HierarchicalRetriever()

    def run_ingestion(self, documents):
        print("\n[2] Running Ingestion Pipeline")
        start_time = time.time()
        self.pipeline.ingest(documents)
        print(f"    Ingestion took {time.time() - start_time:.2f}s")

    def inspect_hierarchy(self):
        print("\n[3] Inspecting Hierarchy & Quantization")

        for layer in range(settings.HIERARCHY_LEVELS):
            store = MetadataStore(layer_id=layer)
            nodes = store.get_all_nodes()
            print(f"    Layer {layer}: {len(nodes)} nodes")

            if nodes:
                sample = nodes[0]
                print(f"      Sample Node ID: {sample.id}")
                print(
                    f"      Text Preview: {sample.text[:50] if sample.text else 'None'}..."
                )

                # Check Quantization
                if isinstance(sample.embedding, QuantizedEmbedding):
                    size_bytes = len(sample.embedding.data)
                    bits = sample.embedding.bits
                    print(
                        f"      Embedding: Quantized ({bits}-bit). Size: {size_bytes} bytes"
                    )
                    print(
                        f"      Scale: {sample.embedding.scale:.4f}, Min: {sample.embedding.min_val:.4f}"
                    )
                else:
                    # Assume float32
                    if hasattr(sample.embedding, "nbytes"):
                        print(
                            f"      Embedding: Float32 (Not Quantized). Size: {sample.embedding.nbytes} bytes"
                        )
                    else:
                        print("      Embedding: Float32 (Not Quantized).")

    def run_retrieval_test(self, query):
        print(f"\n[4] Testing Retrieval for: '{query}'")

        # We want to trace the path. The retriever doesn't expose it by default,
        # so we will infer it by looking at the results and their parents.

        start_time = time.time()
        results = self.retriever.query(query, top_k=3)
        duration = time.time() - start_time

        print(f"    Retrieval took {duration:.4f}s")
        print(f"    Found {len(results)} results.")

        for i, res in enumerate(results):
            print(f"    Result #{i + 1}:")
            print(f"      ID: {res['id']}")
            print(f"      Text: {res['text'][:100]}...")

            # Trace back
            node_id = res["id"]
            path = []

            # Level 0
            store_l0 = MetadataStore(layer_id=0)
            node = store_l0.get_node(node_id)
            if node:
                path.append(f"L0: {node.text[:20]}...")
                parent_id = node.parent_id

                # Level 1
                if parent_id:
                    store_l1 = MetadataStore(layer_id=1)
                    parent = store_l1.get_node(parent_id)
                    if parent:
                        path.append(f"L1: {parent.text[:20]}...")
                        grandparent_id = parent.parent_id

                        # Level 2
                        if grandparent_id:
                            store_l2 = MetadataStore(layer_id=2)
                            grandparent = store_l2.get_node(grandparent_id)
                            if grandparent:
                                path.append(f"L2: {grandparent.text[:20]}...")

            print(f"      Path: {' -> '.join(reversed(path))}")


def main():
    sim = WorkflowSimulator()

    # 1. Setup (Clean start)
    sim.setup(clean=True)

    # 2. Data
    # Technical docs simulation
    docs = [
        "The CPU is the central processing unit of the computer. It executes instructions.",
        "RAM is random access memory. It stores data temporarily for quick access by the CPU.",
        "The GPU is the graphics processing unit. It handles rendering of images and video.",
        "SSD stands for Solid State Drive. It is a storage device that uses flash memory.",
        "A motherboard connects all the components of a computer together.",
        "Python is a high-level programming language known for its readability.",
        "Java is a class-based, object-oriented programming language.",
        "C++ is an extension of the C programming language.",
        "Machine learning is a field of study in artificial intelligence.",
        "Deep learning is part of a broader family of machine learning methods.",
        "Neural networks are computing systems inspired by the biological neural networks.",
        "Supervised learning is the machine learning task of learning a function that maps an input to an output.",
        "Unsupervised learning is a type of machine learning that looks for previously undetected patterns.",
        "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions.",
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.",
        "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images.",
        "Robotics is an interdisciplinary branch of computer science and engineering.",
        "The Internet of Things describes physical objects with sensors, processing ability, software, and other technologies.",
        "Cloud computing is the on-demand availability of computer system resources.",
        "Big data is a field that treats ways to analyze, systematically extract information from, or otherwise deal with data sets that are too large or complex.",
        "Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks.",
        "Blockchain is a growing list of records, called blocks, that are linked using cryptography.",
        "Quantum computing is the use of quantum phenomena such as superposition and entanglement to perform computation.",
        "Virtual reality is a simulated experience that can be similar to or completely different from the real world.",
        "Augmented reality is an interactive experience of a real-world environment where the objects that reside in the real world are enhanced by computer-generated perceptual information.",
    ]

    # Duplicate to ensure enough volume for clustering
    full_docs = []
    for d in docs:
        full_docs.extend([d] * 5)  # 5 copies of each

    # 3. Run Ingestion
    sim.run_ingestion(full_docs)

    # 4. Inspect
    sim.inspect_hierarchy()

    # 5. Query
    sim.run_retrieval_test("machine learning types")
    sim.run_retrieval_test("computer hardware components")

    print("\n=== Simulation Complete ===")


if __name__ == "__main__":
    main()
