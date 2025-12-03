import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_DIM: int = 384

    # Hierarchy Configuration
    HIERARCHY_LEVELS: int = 3

    # Layer-wise Precision Configuration
    # Level 0 (Leaf): High Precision (Float32)
    # Level 1 (Mid): Mid Precision (8-bit)
    # Level 2 (Top): Low Precision (4-bit)
    LAYER_PRECISION: dict = {0: "float32", 1: "int8", 2: "int4"}

    # Quantization Configuration
    QUANTIZATION_BITS: int = 4  # Storage Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    INDEX_DIR: Path = PROJECT_ROOT / "indices"

    # Retrieval Configuration
    TOP_K_LEAF: int = 5
    TOP_K_MID: int = 5
    TOP_K_TOP: int = 10

    # Summarization
    GEMINI_API_KEY: str = ""
    SUMMARIZATION_MODEL: str = "gemini-2.5-flash-lite"

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.INDEX_DIR, exist_ok=True)
