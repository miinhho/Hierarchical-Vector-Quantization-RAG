[1] System Setup
    Cleaning existing data...

[2] Running Ingestion Pipeline
Encoding 125 chunks...
DEBUG: Level 0 Precision: float32, Settings: {0: 'float32', 1: 'int8', 2: 'int4'}
Saving Level 0...
Building level 1 from 125 nodes...
Building level 2 from 25 nodes...
Ingestion complete.
    Ingestion took 1.85s

[3] Inspecting Hierarchy & Quantization
    Layer 0: 125 nodes
      Sample Node ID: f3ac51c9-8dbc-485f-a140-51c67cf0ce92
      Text Preview: The CPU is the central processing unit of the comp...
      Embedding: Float32 (Not Quantized). Size: 1536 bytes
    Layer 1: 25 nodes
      Sample Node ID: 6e8f2b59-250a-4dfb-86ca-1b81be8e2277
      Text Preview: The CPU is the central processing unit of the comp...
      Embedding: Quantized (8-bit). Size: 384 bytes
      Scale: 0.0011, Min: -0.1380
    Layer 2: 5 nodes
      Sample Node ID: 3819c3da-c57d-4734-a0a0-618c79c0e0b5
      Text Preview: The CPU is the central processing unit of the comp...
      Embedding: Quantized (4-bit). Size: 192 bytes
      Scale: 0.0189, Min: -0.1380

[4] Testing Retrieval for: 'machine learning types'
DEBUG: Drilling down to layer 1. Candidates: 0
    Retrieval took 0.0004s
    Found 0 results.

[4] Testing Retrieval for: 'computer hardware components'
DEBUG: Drilling down to layer 1. Candidates: 0
    Retrieval took 0.0004s
    Found 0 results.

=== Simulation Complete ===