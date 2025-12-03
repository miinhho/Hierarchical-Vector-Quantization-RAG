from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
import numpy as np


class QuantizedEmbedding(BaseModel):
    """
    Represents a quantized embedding vector.
    """

    data: bytes  # Packed binary data
    scale: float
    min_val: float
    original_dtype: str = "float32"
    bits: int = 4
    shape: List[int]

    class Config:
        arbitrary_types_allowed = True


class Node(BaseModel):
    """
    Represents a node in the hierarchical RAG system.
    """

    id: str
    level: int
    text: Optional[str] = (
        None  # Text is auxiliary, might be None for higher levels if only summary exists
    )

    # Embedding can be raw float list or quantized object
    # We use Any here to avoid complex validation issues with numpy arrays,
    # but in practice it will be np.ndarray or QuantizedEmbedding
    embedding: Any = None

    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    related_ids: List[str] = Field(default_factory=list)  # Horizontal relationships
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_embedding_vector(self) -> np.ndarray:
        """Helper to get the float vector, dequantizing if necessary."""
        if isinstance(self.embedding, np.ndarray):
            return self.embedding
        elif isinstance(self.embedding, list):
            return np.array(self.embedding, dtype=np.float32)
        elif isinstance(self.embedding, QuantizedEmbedding):
            # This would require the dequantization logic which we will implement in core/quantization.py
            # For now, we return None to let the caller handle it using Quantizer
            return None
        return None


class Cluster(BaseModel):
    """
    Represents a cluster of nodes used for building the next level of hierarchy.
    """

    id: str
    level: int
    center: List[float]
    member_ids: List[str]
    summary: Optional[str] = None
