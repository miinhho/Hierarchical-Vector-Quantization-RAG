from typing import List, Union
import numpy as np
import os
import pickle
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.core.quantization import Quantizer
from src.core.schema import QuantizedEmbedding


class EmbeddingModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            cls._instance.cache_dir = settings.DATA_DIR / "embedding_cache"
            os.makedirs(cls._instance.cache_dir, exist_ok=True)
        return cls._instance

    def _get_cache_path(self, text: str) -> Path:
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.pkl"

    def _load_from_cache(self, text: str) -> Union[np.ndarray, None]:
        path = self._get_cache_path(text)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except OSError:
                return None
        return None

    def _save_to_cache(self, text: str, embedding: np.ndarray):
        path = self._get_cache_path(text)
        with open(path, "wb") as f:
            pickle.dump(embedding, f)

    def encode(
        self, texts: Union[str, List[str]], quantize: bool = True, bits: int = 4
    ) -> Union[np.ndarray, List[QuantizedEmbedding], QuantizedEmbedding]:
        """
        Generates embeddings for the given text(s).
        If quantize is True, returns QuantizedEmbedding objects.
        Otherwise, returns numpy arrays.
        """
        if isinstance(texts, str):
            # Check cache
            cached = self._load_from_cache(texts)
            if cached is not None:
                embedding = cached
            else:
                embedding = self.model.encode(texts)
                self._save_to_cache(texts, embedding)

            if quantize:
                return Quantizer.quantize(embedding, bits=bits)
            return embedding

        # List of embeddings
        # We handle list caching item by item to be safe and simple
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []

        # 1. Try to load from cache
        for i, text in enumerate(texts):
            cached = self._load_from_cache(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)  # Placeholder
                texts_to_compute.append(text)
                indices_to_compute.append(i)

        # 2. Compute missing
        if texts_to_compute:
            computed = self.model.encode(texts_to_compute)
            for i, idx in enumerate(indices_to_compute):
                emb = computed[i]
                embeddings[idx] = emb
                self._save_to_cache(texts_to_compute[i], emb)

        # 3. Quantize if needed
        final_embeddings = np.array(embeddings)
        if quantize:
            return [Quantizer.quantize(emb, bits=bits) for emb in final_embeddings]
        return final_embeddings
