Fetching 20 Newsgroups dataset (this may take a moment)...
Dataset size: 5000 real documents from 20 Newsgroups

=== Running Baseline: Flat RAG ===
Flat RAG Ingestion Time: 432.9816s
Flat RAG Index Size (Approx RAM): 25296.00 KB
Flat RAG Query Time: 0.0071s

=== Running Proposed: HiRAG ===
Encoding 16864 chunks...
DEBUG: Level 0 Precision: float32, Settings: {0: 'float32', 1: 'int8', 2: 'int4'}
Saving Level 0...
Building level 1 from 16864 nodes...
Building level 2 from 3372 nodes...
Ingestion complete.
HiRAG Ingestion Time: 870.0009s
HiRAG Vector Index Size (Disk): 26693.08 KB
DEBUG: Top layer is 2. Store ntotal: 674
DEBUG: Top layer search returned 10 results
DEBUG: Drilling down to layer 1. Candidates: 10
DEBUG: Found 54 children
DEBUG: Retrieved 54 child nodes
DEBUG: Scored 54 children
DEBUG: Selected 5 candidates for next layer
DEBUG: Drilling down to layer 0. Candidates: 5
DEBUG: Found 31 children
DEBUG: Retrieved 31 child nodes
DEBUG: Scored 31 children
DEBUG: Selected 5 candidates for next layer
HiRAG Query Time: 0.0030s

=== Comparison Summary ===
Metric               | Flat RAG        | HiRAG           | Improvement
----------------------------------------------------------------------
Ingestion Time       | 432.9816s       | 870.0009s       | 0.50x (Slower is expected)
Index Size           | 25296.00 KB       | 26693.08 KB       | 0.95x Smaller
Query Time           | 0.0071s       | 0.0030s       | 2.40x Faster