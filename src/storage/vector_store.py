import faiss
import numpy as np
import pickle
from typing import List, Tuple
from src.config import settings
from src.core.quantization import Quantizer
from src.core.schema import Node, QuantizedEmbedding


class VectorStore:
    def __init__(self, layer_id: int, dim: int = settings.VECTOR_DIM):
        self.layer_id = layer_id
        self.dim = dim

        # Determine precision based on layer
        precision = settings.LAYER_PRECISION.get(layer_id, "int4")  # Default to int4

        self.is_quantized = False

        if precision == "float32":
            self.index = faiss.IndexFlatL2(dim)
        elif precision == "float16":
            try:
                self.index = faiss.IndexScalarQuantizer(
                    dim, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2
                )
                self.is_quantized = True
            except AttributeError:
                print(
                    f"Warning: QT_fp16 not supported for layer {layer_id}, falling back to Flat"
                )
                self.index = faiss.IndexFlatL2(dim)
        elif precision == "int8":
            try:
                self.index = faiss.IndexScalarQuantizer(
                    dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
                )
                self.is_quantized = True
            except AttributeError:
                print(
                    f"Warning: QT_8bit not supported for layer {layer_id}, falling back to Flat"
                )
                self.index = faiss.IndexFlatL2(dim)
        elif precision == "int4":
            try:
                self.index = faiss.IndexScalarQuantizer(
                    dim, faiss.ScalarQuantizer.QT_4bit, faiss.METRIC_L2
                )
                self.is_quantized = True
            except AttributeError:
                print(
                    f"Warning: QT_4bit not supported for layer {layer_id}, falling back to Flat"
                )
                self.index = faiss.IndexFlatL2(dim)
        else:
            # Fallback
            self.index = faiss.IndexFlatL2(dim)

        self.id_map = {}  # internal_id -> node_id
        self.reverse_id_map = {}  # node_id -> internal_id
        self.load()

    def add(self, nodes: List[Node]):
        if not nodes:
            return

        vectors = []
        valid_nodes = []

        for i, node in enumerate(nodes):
            vec = node.get_embedding_vector()
            if vec is None:
                # If it's quantized, dequantize it
                if isinstance(node.embedding, QuantizedEmbedding):
                    vec = Quantizer.dequantize(node.embedding)
                else:
                    continue  # Should not happen

            vectors.append(vec)
            valid_nodes.append(node)

        if vectors:
            data = np.array(vectors).astype("float32")

            # Train if necessary
            if self.is_quantized and not self.index.is_trained:
                # Faiss requires training for ScalarQuantizer
                # If we don't have enough data points for training, it might fail or be suboptimal
                # But for ScalarQuantizer, it usually just needs min/max, so small data is fine.
                try:
                    self.index.train(data)
                except Exception as e:
                    print(f"Error training index for layer {self.layer_id}: {e}")
                    # Fallback to Flat if training fails (e.g. too few points)
                    self.index = faiss.IndexFlatL2(self.dim)
                    self.is_quantized = False

            start_idx = self.index.ntotal
            self.index.add(data)

            # Verify addition
            if self.index.ntotal == start_idx:
                print(
                    f"Warning: Index add failed for layer {self.layer_id}. ntotal did not increase."
                )

            # Update maps
            for i, node in enumerate(valid_nodes):
                internal_id = start_idx + i
                self.id_map[internal_id] = node.id
                self.reverse_id_map[node.id] = internal_id

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0:
            return []

        distances, indices = self.index.search(
            query_vector.reshape(1, -1).astype("float32"), k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                node_id = self.id_map.get(idx)
                if node_id:
                    results.append((node_id, float(dist)))

        return results

    def save(self):
        faiss.write_index(
            self.index, str(settings.INDEX_DIR / f"layer_{self.layer_id}.index")
        )
        # Save maps
        with open(settings.INDEX_DIR / f"layer_{self.layer_id}_map.pkl", "wb") as f:
            pickle.dump((self.id_map, self.reverse_id_map), f)

    def load(self):
        path = settings.INDEX_DIR / f"layer_{self.layer_id}.index"
        if path.exists():
            self.index = faiss.read_index(str(path))
            map_path = settings.INDEX_DIR / f"layer_{self.layer_id}_map.pkl"
            if map_path.exists():
                with open(map_path, "rb") as f:
                    self.id_map, self.reverse_id_map = pickle.load(f)
