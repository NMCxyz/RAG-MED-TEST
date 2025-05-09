"""
Fusion Module
------------
Module kết hợp kết quả truy xuất từ nhiều nguồn (dense, sparse, BM25, reranker)
sử dụng phương pháp Reciprocal Rank Fusion (RRF).
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import math

def reciprocal_rank_fusion(result_lists: List[List[Dict[str, Any]]],
                         k: float = 60.0,
                         weights: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """
    Kết hợp nhiều danh sách kết quả xếp hạng bằng Reciprocal Rank Fusion
    
    Args:
        result_lists: Danh sách các danh sách kết quả cần kết hợp
                     Mỗi kết quả là dict với ít nhất {"id": str, "score": float}
        k: Hằng số RRF (mặc định = 60)
        weights: Trọng số cho từng danh sách kết quả (nếu None, tất cả bằng nhau)
        
    Returns:
        Danh sách kết quả đã kết hợp và xếp hạng lại
    """
    if not result_lists:
        return []
    
    print(f"[DEMO] Performing Reciprocal Rank Fusion on {len(result_lists)} result lists")
    
    # Nếu không có trọng số, gán bằng nhau
    if weights is None:
        weights = [1.0] * len(result_lists)
    elif len(weights) != len(result_lists):
        raise ValueError("Number of weights must match number of result lists")
    
    # Chuẩn hóa trọng số để tổng = 1
    total_weight = sum(weights)
    weights = [w/total_weight for w in weights]
    
    # Dict lưu điểm RRF theo ID
    rrf_scores: Dict[str, float] = {}
    
    # Dict lưu toàn bộ thông tin document theo ID
    all_docs: Dict[str, Dict[str, Any]] = {}
    
    # Tính điểm RRF cho mỗi document
    for i, results in enumerate(result_lists):
        weight = weights[i]
        
        # Tính điểm RRF với công thức: weight * 1/(k + rank)
        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            
            # Lưu thông tin document nếu chưa có
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
            
            # Cộng dồn điểm RRF
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            
            rrf_score = weight * (1.0 / (k + rank))
            rrf_scores[doc_id] += rrf_score
    
    # Tạo danh sách kết quả mới với điểm RRF
    fused_results = []
    for doc_id, rrf_score in rrf_scores.items():
        # Lấy thông tin document từ dict đã lưu
        doc = all_docs[doc_id].copy()
        
        # Gán điểm RRF
        doc["score"] = rrf_score
        doc["fusion_score"] = rrf_score
        
        # Nếu đã có điểm gốc, lưu lại để so sánh
        if "score" in all_docs[doc_id]:
            doc["original_score"] = all_docs[doc_id]["score"]
        
        fused_results.append(doc)
    
    # Sắp xếp theo điểm RRF giảm dần
    fused_results.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"[DEMO] RRF fusion completed, {len(fused_results)} unique documents ranked")
    
    return fused_results

def rerank_with_cross_encoder(results: List[Dict[str, Any]], 
                            query: str,
                            model_name: str = "BAAI/bge-reranker-base",
                            top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Sắp xếp lại kết quả bằng cross-encoder reranker
    
    Args:
        results: Danh sách kết quả cần rerank
        query: Câu truy vấn
        model_name: Tên mô hình cross-encoder
        top_k: Số lượng kết quả trả về (nếu None, trả về tất cả)
        
    Returns:
        Danh sách kết quả đã rerank
    """
    print(f"[DEMO] Reranking with {model_name}")
    
    # DEMO: Mô phỏng reranking
    # Trong thực tế, sẽ gọi mô hình cross-encoder thực (từ HuggingFace hoặc API)
    
    # Tạo bản sao để không thay đổi bản gốc
    reranked_results = results.copy()
    
    # Mô phỏng việc tính điểm mới với cross-encoder
    import numpy as np
    np.random.seed(hash(query) % 10000 + 2)  # Seed khác so với dense và sparse
    
    for doc in reranked_results:
        # Lưu điểm cũ
        doc["fusion_score"] = doc["score"]
        
        # Tạo nhiễu để mô phỏng reranker thực
        text = doc.get("text", "")
        query_text_match = sum(word in text.lower() for word in query.lower().split())
        
        # Điểm số mới dựa trên điểm cũ và độ khớp query-text
        base_score = doc["score"] * 0.5 + 0.3
        match_boost = min(0.2, query_text_match * 0.05)
        noise = np.random.uniform(-0.05, 0.05)
        
        # Điểm số mới
        rerank_score = min(0.99, max(0.01, base_score + match_boost + noise))
        doc["score"] = rerank_score
        doc["rerank_score"] = rerank_score
    
    # Sắp xếp lại theo điểm số mới
    reranked_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Cắt kết quả nếu cần
    if top_k is not None:
        reranked_results = reranked_results[:top_k]
    
    print(f"[DEMO] Reranking completed, top score: {reranked_results[0]['score']:.4f}")
    
    return reranked_results

def combine_evidence(retrieved_chunks: List[Dict[str, Any]], 
                    max_token_limit: int = 3000) -> str:
    """
    Kết hợp các đoạn văn bản đã truy xuất thành một đoạn chứng cứ (evidence)
    
    Args:
        retrieved_chunks: Danh sách các đoạn văn bản đã truy xuất
        max_token_limit: Giới hạn token tối đa (ước lượng ~4 ký tự/token)
        
    Returns:
        Đoạn văn bản chứng cứ đã kết hợp
    """
    import re
    
    # Ước lượng đơn giản 1 token = 4 ký tự
    def estimate_tokens(text: str) -> int:
        return len(text) // 4
    
    evidence_parts = []
    current_tokens = 0
    
    # Thêm đoạn vào evidence cho đến khi đạt giới hạn token
    for chunk in retrieved_chunks:
        chunk_text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "Unknown source")
        
        # Tiền xử lý đoạn: loại bỏ khoảng trắng dư
        chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
        
        # Thêm nguồn vào cuối đoạn
        processed_text = f"{chunk_text} [Source: {source}]"
        
        # Ước lượng số token
        chunk_tokens = estimate_tokens(processed_text)
        
        # Kiểm tra giới hạn token
        if current_tokens + chunk_tokens > max_token_limit:
            # Nếu đã gần đầy, chỉ lấy phần đầu của đoạn hiện tại
            remaining_tokens = max_token_limit - current_tokens
            if remaining_tokens > 100:  # Chỉ lấy nếu đủ có ý nghĩa
                truncated_text = processed_text[:remaining_tokens*4] + "... [truncated]"
                evidence_parts.append(truncated_text)
            break
        
        # Thêm đoạn vào evidence
        evidence_parts.append(processed_text)
        current_tokens += chunk_tokens
    
    # Kết hợp tất cả các đoạn
    combined_evidence = "\n\n".join(evidence_parts)
    
    return combined_evidence 