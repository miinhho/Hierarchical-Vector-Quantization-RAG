# Hierarchical Vector Quantization RAG

## Overview
Unlike traditional HiRAG systems (like RAPTOR) that rely on slow and expensive LLM summarization, Hierarchical Vector Quantization RAG uses a **Vector-Centric Representative Selection** strategy combined with **Variable Quantization**.

## Key Architecture

### 1. Hierarchy Construction (Representative Selection)
Instead of generating new summaries for each cluster using an LLM, Hierarchical Vector Quantization RAG selects the **Representative Node** (the node closest to the cluster centroid) to represent the cluster at the upper level.
- **Pros**: Zero API cost, >100x faster ingestion, zero hallucinations (preserves original data).
- **Cons**: Lacks the abstractive synthesis capability of generative summaries.

### 2. Variable Quantization (Mixed-Precision)
The system applies different quantization levels based on the hierarchy depth to optimize the trade-off between memory/speed and accuracy.

| Layer | Role | Precision | Purpose |
| :--- | :--- | :--- | :--- |
| **L2 (Top)** | Global Search | **Int4** (4-bit) | Extremely fast, broad filtering. Acts as a "low-res thumbnail" to quickly identify relevant clusters. |
| **L1 (Mid)** | Intermediate | **Int8** (8-bit) | Balanced precision for narrowing down candidates. |
| **L0 (Leaf)** | Exact Match | **Float32** | High precision for final ranking and retrieving the exact context. |

### 3. The Role of Quantization: "Blurring" as a Feature
Does lowering resolution via quantization actually help? Yes, by acting as a **Low-pass Filter**.
- **Abstraction Source**: The true semantic abstraction comes from **Centroid Selection** (averaging the topic), not just quantization.
- **Blurring Effect**: Int4 quantization removes "noise" (fine-grained vector details), forcing the search to focus on the **general direction (topic)** rather than exact keyword matches.
- **Efficiency**: This allows the top layers to act as a high-speed "coarse filter," discarding irrelevant sections massively before the expensive Float32 comparison happens at the leaf level.

## Performance Comparison
(Based on 5,000 document test using the 20 Newsgroups dataset)

| Metric | Flat RAG (Baseline) | MVGA (Proposed) | Improvement |
| :--- | :--- | :--- | :--- |
| **Query Time** | 0.0071s | **0.0030s** | **~2.4x Faster** |
| **Ingestion Cost** | Low | **Zero (No LLM)** | **100% Cheaper** vs Standard HiRAG |

## Getting Started

### Prerequisites
- Python 3.10+
- `uv` or `pip`

### Installation
```bash
pip install -r requirements.txt
```

### Running Simulation
To see the architecture in action with a small synthetic dataset:
```bash
uv run simulate_workflow.py
```

### Running Performance Benchmark
To compare Hierarchical Vector Quantization RAG against a standard Flat RAG baseline using real-world data:
```bash
uv run compare_performance.py
```