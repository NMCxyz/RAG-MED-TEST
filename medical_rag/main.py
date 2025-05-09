"""
Medical RAG System Demo
-----------------------
Mô phỏng toàn bộ luồng xử lý của hệ thống RAG y khoa từ câu hỏi đến câu trả lời
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any

# Import các module của hệ thống
from chunker.semantic_chunker import semantic_chunk_pdf
from kg_extract.triplet_extractor import extract_triplets_from_chunk
from query_processing.complexity_classifier import classify_complexity
from query_processing.param_selector import select_params
from query_processing.query_rewriter import rewrite_query
from retrieval.dense_retriever import retrieve_with_bge_m3
from retrieval.sparse_retriever import retrieve_with_splade
from retrieval.fusion import reciprocal_rank_fusion
from graph.seed_select import select_seed_nodes
from graph.mixpr_router import get_subgraph_with_mixpr
from graph.gfm_encoder import encode_graph_with_gfm
from reasoning.graph_cot import reasoning_with_graph_cot
from reasoning.qwen_med import generate_answer_with_qwen
from verification.refind import verify_answer_with_refind
from verification.formatter import format_verified_answer

# Demo dữ liệu
DEMO_PDF_PATH = "data/sample_medical_text.pdf"
DEMO_QUERY = "new osteoporosis therapy"

def run_full_pipeline(query: str, pdf_path: Optional[str] = None) -> Dict[str, Any]:
    """Chạy toàn bộ pipeline RAG y khoa và trả về kết quả mỗi bước"""
    
    results = {"query": query}
    
    # 0a. Semantic chunking (không cần chạy nếu đã có chunks)
    if pdf_path:
        print("\n--- Bước 0a: Semantic Chunking ---")
        chunks = semantic_chunk_pdf(pdf_path)
        print(f"Output: {len(chunks)} chunks được tạo, ví dụ: {chunks[0]['id']} - {chunks[0]['text'][:50]}...")
        results["chunks"] = chunks
    
    # 0b. Triplet Extract
    print("\n--- Bước 0b: Triplet Extraction ---")
    sample_chunk = {
        "id": "C42", 
        "text": "Bisphosphonates like Alendronate are used in treating osteoporosis by inhibiting bone resorption."
    }
    triplets = extract_triplets_from_chunk(sample_chunk)
    print(f"Output: {len(triplets)} triplets extracted, ví dụ: {triplets[0]}")
    results["triplets"] = triplets
    
    # 1a. Complexity Classification
    print("\n--- Bước 1a: Complexity Classification ---")
    complexity_tag = classify_complexity(query)
    print(f"Output: {complexity_tag}")
    results["complexity_tag"] = complexity_tag
    
    # 1b. Parameter Selection
    print("\n--- Bước 1b: Parameter Selection ---")
    params = select_params(complexity_tag)
    print(f"Output: {params}")
    results["params"] = params
    
    # 1c. Query Rewriting
    print("\n--- Bước 1c: Query Rewriting (DMQR) ---")
    rewritten_queries = rewrite_query(query, params["num_rewrites"])
    print(f"Output: {rewritten_queries}")
    results["rewritten_queries"] = rewritten_queries
    
    # 2a. Dense Retrieval with BGE-M3
    print("\n--- Bước 2a: Dense Retrieval (BGE-M3) ---")
    q1 = rewritten_queries[0]
    dense_vector, dense_results = retrieve_with_bge_m3(q1)
    print(f"Output: dense_vector (shape: {dense_vector.shape}), top chunk: {dense_results[0]}")
    results["dense_vector"] = dense_vector
    results["dense_results"] = dense_results
    
    # 2b. Sparse Retrieval with SPLADE
    print("\n--- Bước 2b: Sparse Retrieval (SPLADE) ---")
    sparse_vector, sparse_results = retrieve_with_splade(q1)
    print(f"Output: sparse_vector (keys: {list(sparse_vector.keys())[:3]}...), top chunk: {sparse_results[0]}")
    results["sparse_vector"] = sparse_vector
    results["sparse_results"] = sparse_results
    
    # 2c. Fusion with RRF
    print("\n--- Bước 2c: Fusion (RRF) ---")
    fused_results = reciprocal_rank_fusion([dense_results, sparse_results])
    print(f"Output: top fused result: {fused_results[0]}")
    results["fused_results"] = fused_results
    
    # 3a. Seed Selection
    print("\n--- Bước 3a: Seed Selection ---")
    seed_nodes = select_seed_nodes(dense_vector)
    print(f"Output: {len(seed_nodes)} seed nodes: {seed_nodes[:3]}...")
    results["seed_nodes"] = seed_nodes
    
    # 3b. MixPR Router
    print("\n--- Bước 3b: MixPR Router ---")
    subgraph_nodes, subgraph_edges = get_subgraph_with_mixpr(seed_nodes, params["k"], params["topK"])
    print(f"Output: {len(subgraph_nodes)} nodes, {len(subgraph_edges)} edges")
    results["subgraph_nodes"] = subgraph_nodes
    results["subgraph_edges"] = subgraph_edges
    
    # 4. GFM Encoder
    print("\n--- Bước 4: GFM Encoder ---")
    node_embeddings, graph_embedding = encode_graph_with_gfm(subgraph_nodes, subgraph_edges)
    print(f"Output: node_embeddings (shape: {node_embeddings.shape}), graph_embedding (shape: {graph_embedding.shape})")
    results["node_embeddings"] = node_embeddings
    results["graph_embedding"] = graph_embedding
    
    # 5. Graph-CoT
    print("\n--- Bước 5: Graph-CoT Reasoning ---")
    cot_steps, evidence_nodes = reasoning_with_graph_cot(query, subgraph_nodes, subgraph_edges)
    print(f"Output: {len(cot_steps)} reasoning steps, {len(evidence_nodes)} evidence nodes")
    results["cot_steps"] = cot_steps
    results["evidence_nodes"] = evidence_nodes
    
    # 6a. Qwen-Med
    print("\n--- Bước 6a: Qwen-Med Answer Generation ---")
    evidence_texts = [
        "Bisphosphonates are the first-line therapy for osteoporosis, with alendronate being most common.",
        "Denosumab is a monoclonal antibody that inhibits RANKL, reducing bone resorption.",
        "Anabolic therapies like teriparatide stimulate bone formation in severe osteoporosis cases."
    ]
    draft_answer = generate_answer_with_qwen(query, evidence_texts, cot_steps)
    print(f"Output: Draft answer: {draft_answer[:100]}...")
    results["draft_answer"] = draft_answer
    
    # 6b. REFIND Verification
    print("\n--- Bước 6b: REFIND Verification ---")
    verified_answer, hallucinated_spans = verify_answer_with_refind(draft_answer, evidence_texts)
    print(f"Output: Verified answer (removed {len(hallucinated_spans)} spans)")
    results["verified_answer"] = verified_answer
    results["hallucinated_spans"] = hallucinated_spans
    
    # 6c. Formatting
    print("\n--- Bước 6c: Formatting ---")
    formatted_answer = format_verified_answer(verified_answer, evidence_nodes)
    print(f"Output: Final formatted answer with references")
    results["formatted_answer"] = formatted_answer
    
    return results

if __name__ == "__main__":
    print("=== MEDICAL RAG SYSTEM DEMO ===")
    print(f"Input Query: '{DEMO_QUERY}'")
    
    results = run_full_pipeline(DEMO_QUERY)
    
    print("\n=== FINAL ANSWER ===")
    print(results["formatted_answer"]) 