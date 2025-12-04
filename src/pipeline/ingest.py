from typing import List
import uuid
from src.core.schema import Node
from src.core.embedding import EmbeddingModel
from src.storage.vector_store import VectorStore
from src.storage.metadata_store import MetadataStore
from src.hierarchy.builder import HierarchyBuilder
from src.config import settings


class IngestionPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.builder = HierarchyBuilder()
        self.vector_stores = {}
        self.metadata_stores = {}

        for i in range(settings.HIERARCHY_LEVELS):
            self.vector_stores[i] = VectorStore(layer_id=i)
            self.metadata_stores[i] = MetadataStore(layer_id=i)

    def ingest(self, documents: List[str]):
        """
        Ingests a list of documents, chunks them, and builds the hierarchy.
        """
        # 1. Create Level 0 Nodes (Chunks)
        # Simple splitting by newline or fixed size for now
        chunks = []
        for doc in documents:
            # Simple chunking: split by 500 chars overlap 50
            chunk_size = 500
            overlap = 50
            if len(doc) <= chunk_size:
                chunks.append(doc)
            else:
                for i in range(0, len(doc) - overlap, chunk_size - overlap):
                    chunks.append(doc[i : i + chunk_size])

        level_0_nodes = []
        print(f"Encoding {len(chunks)} chunks...")

        # Determine precision for Level 0
        precision = settings.LAYER_PRECISION.get(0, "float32")
        print(
            f"DEBUG: Level 0 Precision: {precision}, Settings: {settings.LAYER_PRECISION}"
        )
        quantize = precision != "float32"
        bits = 4
        if precision == "int8":
            bits = 8
        elif precision == "int4":
            bits = 4
        elif precision == "float16":
            bits = 16

        embeddings = self.embedder.encode(chunks, quantize=quantize, bits=bits)

        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            node = Node(
                id=str(uuid.uuid4()),
                level=0,
                text=text,
                embedding=emb,
                metadata={"source_idx": i},
            )
            level_0_nodes.append(node)

        # Save Level 0
        print("Saving Level 0...")
        self.vector_stores[0].add(level_0_nodes)
        for node in level_0_nodes:
            self.metadata_stores[0].add_node(node)
        self.vector_stores[0].save()
        self.metadata_stores[0].save()

        # 2. Build Hierarchy
        current_nodes = level_0_nodes
        for level in range(settings.HIERARCHY_LEVELS - 1):
            print(f"Building level {level + 1} from {len(current_nodes)} nodes...")
            next_level_nodes = self.builder.build_next_level(
                current_nodes, current_level=level
            )

            if not next_level_nodes:
                print(f"Warning: No nodes generated for level {level + 1}")
                break

            # Save next level
            self.vector_stores[level + 1].add(next_level_nodes)
            for node in next_level_nodes:
                self.metadata_stores[level + 1].add_node(node)

            self.vector_stores[level + 1].save()
            self.metadata_stores[level + 1].save()

            # Save current level again because parent_ids were updated
            self.metadata_stores[level].save()

            current_nodes = next_level_nodes

        print("Ingestion complete.")
