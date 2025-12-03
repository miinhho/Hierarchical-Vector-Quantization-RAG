from typing import List, Dict
import numpy as np
from src.core.embedding import EmbeddingModel
from src.storage.vector_store import VectorStore
from src.storage.metadata_store import MetadataStore
from src.config import settings
from src.core.quantization import Quantizer
from src.core.schema import QuantizedEmbedding


class HierarchicalRetriever:
    def __init__(self, levels: int = settings.HIERARCHY_LEVELS):
        self.levels = levels
        self.embedder = EmbeddingModel()
        self.vector_stores: Dict[int, VectorStore] = {}
        self.metadata_stores: Dict[int, MetadataStore] = {}

        # Initialize stores for all levels
        for i in range(levels):
            self.vector_stores[i] = VectorStore(layer_id=i)
            self.metadata_stores[i] = MetadataStore(layer_id=i)

    def query(self, text: str, top_k: int = settings.TOP_K_LEAF) -> List[Dict]:
        """
        Performs a hierarchical search.
        1. Search top layer.
        2. For each result, get children.
        3. Search within children (using exact scan since subset is small).
        4. Repeat until leaf layer.
        """
        # 1. Encode query (float32)
        query_emb = self.embedder.encode(text, quantize=False)
        if isinstance(query_emb, list):  # Should be single vector
            query_emb = query_emb[0]

        # Ensure query_emb is float32 numpy array
        query_emb = np.array(query_emb).astype(np.float32)

        # 2. Start from top layer
        current_candidates = []  # List of node_ids

        # Top layer search
        top_layer = self.levels - 1
        print(
            f"DEBUG: Top layer is {top_layer}. Store ntotal: {self.vector_stores[top_layer].index.ntotal}"
        )

        # We search for more candidates at the top to ensure coverage
        top_results = self.vector_stores[top_layer].search(
            query_emb, k=settings.TOP_K_TOP
        )
        print(f"DEBUG: Top layer search returned {len(top_results)} results")
        current_candidates = [res[0] for res in top_results]

        # 3. Drill down
        for layer in range(self.levels - 2, -1, -1):
            print(
                f"DEBUG: Drilling down to layer {layer}. Candidates: {len(current_candidates)}"
            )
            if not current_candidates:
                break

            # Get children of current candidates
            children_ids = set()
            for parent_id in current_candidates:
                parent_node = self.metadata_stores[layer + 1].get_node(parent_id)
                if parent_node:
                    children_ids.update(parent_node.children_ids)

            print(f"DEBUG: Found {len(children_ids)} children")

            if not children_ids:
                break

            # Retrieve child nodes
            child_nodes = []
            for cid in children_ids:
                node = self.metadata_stores[layer].get_node(cid)
                if node:
                    child_nodes.append(node)

            print(f"DEBUG: Retrieved {len(child_nodes)} child nodes")

            if not child_nodes:
                continue

            # Score children
            scores = []
            for node in child_nodes:
                vec = node.get_embedding_vector()
                if vec is None:
                    if isinstance(node.embedding, QuantizedEmbedding):
                        vec = Quantizer.dequantize(node.embedding)
                    else:
                        print(f"DEBUG: Node {node.id} has no valid embedding")
                        continue

                # L2 Distance
                dist = np.linalg.norm(vec - query_emb)
                scores.append((node.id, dist))

            print(f"DEBUG: Scored {len(scores)} children")

            # Sort by distance (asc)
            scores.sort(key=lambda x: x[1])

            # Select top k for next layer
            k_next = settings.TOP_K_MID if layer > 0 else top_k
            current_candidates = [s[0] for s in scores[:k_next]]
            print(
                f"DEBUG: Selected {len(current_candidates)} candidates for next layer"
            )

        # 4. Return final results (Leaf nodes)
        results = []
        for cid in current_candidates:
            node = self.metadata_stores[0].get_node(cid)
            if node:
                results.append(
                    {"id": node.id, "text": node.text, "metadata": node.metadata}
                )

        return results
