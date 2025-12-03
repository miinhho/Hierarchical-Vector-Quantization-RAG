from typing import Dict, Optional, List
import pickle
from src.core.schema import Node
from src.config import settings

class MetadataStore:
    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.nodes: Dict[str, Node] = {}
        self.file_path = settings.DATA_DIR / f"layer_{layer_id}_metadata.pkl"
        self.load()

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_all_nodes(self) -> List[Node]:
        return list(self.nodes.values())

    def save(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.nodes, f)

    def load(self):
        if self.file_path.exists():
            with open(self.file_path, 'rb') as f:
                self.nodes = pickle.load(f)
