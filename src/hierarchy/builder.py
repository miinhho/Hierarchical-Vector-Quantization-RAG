from typing import List, Dict
from sklearn.cluster import KMeans
import numpy as np
import uuid
from src.core.schema import Node, QuantizedEmbedding
from src.core.embedding import EmbeddingModel
from src.core.quantization import Quantizer
from src.config import settings


class HierarchyBuilder:
    def __init__(self):
        self.embedder = EmbeddingModel()

    def build_next_level(self, nodes: List[Node], current_level: int) -> List[Node]:
        """
        Takes a list of nodes at current_level, clusters them, selects a representative node,
        and returns a list of new nodes at current_level + 1.
        """
        if not nodes:
            return []

        # Extract embeddings
        embeddings = []
        valid_nodes = []
        for node in nodes:
            vec = node.get_embedding_vector()
            if vec is None and isinstance(node.embedding, QuantizedEmbedding):
                vec = Quantizer.dequantize(node.embedding)

            if vec is not None:
                embeddings.append(vec)
                valid_nodes.append(node)

        if not embeddings:
            return []

        # Map node_id to embedding for quick lookup later
        node_embedding_map = {n.id: vec for n, vec in zip(valid_nodes, embeddings)}

        X = np.array(embeddings)

        # Determine number of clusters
        # Heuristic: sqrt(N) or N/5, but bounded
        # We want to reduce the number of nodes significantly
        n_clusters = max(1, int(len(X) / 5))

        if len(X) < 5:
            n_clusters = 1

        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        labels = kmeans.fit_predict(X)

        clusters: Dict[int, List[Node]] = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_nodes[i])

        new_nodes = []
        for label, members in clusters.items():
            # Establish horizontal relationships (siblings)
            member_ids = [n.id for n in members]
            for member in members:
                # Link to other members in the same cluster (siblings)
                member.related_ids = [mid for mid in member_ids if mid != member.id]

            # Select Representative Node (Centroid-based)
            # Instead of summarizing with LLM, we pick the node closest to the cluster center.
            center_vec = kmeans.cluster_centers_[label]

            closest_member = None
            min_dist = float("inf")

            for member in members:
                vec = node_embedding_map.get(member.id)
                if vec is None:
                    continue

                dist = np.linalg.norm(vec - center_vec)
                if dist < min_dist:
                    min_dist = dist
                    closest_member = member

            summary = closest_member.text if closest_member else ""
            # Optional: Add a prefix to indicate it's a representative
            # summary = f"[Representative] {summary}"

            # Create embedding for summary
            # Determine precision for next level
            next_level = current_level + 1
            precision = settings.LAYER_PRECISION.get(next_level, "int8")
            quantize = precision != "float32"
            bits = 4
            if precision == "int8":
                bits = 8
            elif precision == "int4":
                bits = 4
            elif precision == "float16":
                bits = 16

            emb = self.embedder.encode(summary, quantize=quantize, bits=bits)

            new_node = Node(
                id=str(uuid.uuid4()),
                level=current_level + 1,
                text=summary,
                embedding=emb,
                children_ids=[n.id for n in members],
                metadata={
                    "cluster_size": len(members),
                    "representative_source_id": closest_member.id
                    if closest_member
                    else None,
                },
            )

            # Update parent pointers in children (Note: this modifies the input nodes)
            for member in members:
                member.parent_id = new_node.id

            new_nodes.append(new_node)

        return new_nodes
