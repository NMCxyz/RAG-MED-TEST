"""
Sparse Retriever Module
---------------------
Module truy xuất sparse vector sử dụng mô hình SPLADE để tạo biểu diễn
lexical expansion và tìm kiếm văn bản liên quan.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
from transformers import AutoTokenizer, AutoModelForMaskedLM
import scipy.sparse as sp

class SPLADERetriever:
    def __init__(self, model_name="naver/splade-cocondenser-ensembledistil", 
                 device=None):
        """
        Khởi tạo Sparse Retriever với SPLADE
        
        Args:
            model_name: Tên mô hình SPLADE trên HuggingFace
            device: Thiết bị để chạy mô hình (None: tự động)
        """
        # Xác định thiết bị: GPU nếu có, ngược lại CPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Tải tokenizer và model từ HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Từ điển reverse lookup cho token ID và token
        self.id2token = {idx: token for token, idx in self.tokenizer.get_vocab().items()}
        
        # Lưu trữ cho document index
        self.doc_vectors = None
        self.document_lookup = {}
        
    def _encode_text(self, text: str, 
                    max_length: int = 512, 
                    is_query: bool = False) -> sp.csr_matrix:
        """
        Mã hóa văn bản thành sparse lexical vector dùng SPLADE
        
        Args:
            text: Văn bản cần mã hóa
            max_length: Độ dài tối đa token
            is_query: True nếu là truy vấn, False nếu là document
            
        Returns:
            Sparse vector (scipy CSR matrix)
        """
        # Đặt model ở chế độ eval
        self.model.eval()
        
        # Tokenize đầu vào
        tokens = self.tokenizer(text, 
                               return_tensors="pt", 
                               padding=True, 
                               truncation=True, 
                               max_length=max_length)
        
        # Chuyển tokens sang device đúng
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Vô hiệu hóa gradient
        with torch.no_grad():
            # Forward pass qua model
            outputs = self.model(**tokens)
            
            # SPLADE tính toán ReLU(log(1 + RELU(MLM logits)))
            logits = outputs.logits
            relu_log = torch.log(1 + torch.relu(logits))
            
            # Lấy giá trị lớn nhất cho mỗi token trong từ điển
            # Đối với query: lấy max cho mỗi vị trí
            # Đối với document: lấy max theo token_id
            if is_query:
                # Lấy max cho mỗi từ trong truy vấn (queries thường ngắn)
                aggregated_logits = torch.max(relu_log, dim=1)[0]  # [1, vocab_size]
            else:
                # Lấy max cho mỗi token_id trong văn bản (documents thường dài)
                aggregated_logits = torch.max(relu_log, dim=1)[0]  # [1, vocab_size]
                
            # Chuyển sang numpy
            weights = aggregated_logits.squeeze().cpu().numpy()
        
        # Chỉ giữ lại các token có weight > 0
        nonzero_indices = np.nonzero(weights)[0]
        nonzero_weights = weights[nonzero_indices]
        
        # Tạo sparse vector dạng CSR
        vocab_size = len(self.tokenizer.get_vocab())
        sparse_vector = sp.csr_matrix(
            (nonzero_weights, (np.zeros_like(nonzero_indices), nonzero_indices)),
            shape=(1, vocab_size)
        )
        
        return sparse_vector
    
    def encode_query(self, query: str) -> sp.csr_matrix:
        """
        Mã hóa truy vấn thành sparse vector
        
        Args:
            query: Câu truy vấn
            
        Returns:
            Sparse vector (scipy CSR matrix)
        """
        return self._encode_text(query, max_length=64, is_query=True)
    
    def encode_documents(self, 
                       documents: List[Dict[str, str]], 
                       batch_size: int = 8) -> List[sp.csr_matrix]:
        """
        Mã hóa danh sách văn bản thành sparse vectors
        
        Args:
            documents: Danh sách các văn bản [{"id": id, "text": text}, ...]
            batch_size: Kích thước batch
            
        Returns:
            Danh sách các sparse vector
        """
        from tqdm import tqdm
        
        # Trích xuất văn bản
        texts = [doc["text"] for doc in documents]
        all_vectors = []
        
        # Mã hóa từng batch
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding documents"):
            batch_texts = texts[i:i+batch_size]
            batch_vectors = [self._encode_text(text, is_query=False) for text in batch_texts]
            all_vectors.extend(batch_vectors)
        
        return all_vectors
    
    def build_index(self, documents: List[Dict[str, str]]):
        """
        Xây dựng index từ danh sách văn bản
        
        Args:
            documents: Danh sách các văn bản [{"id": id, "text": text}, ...]
        """
        print(f"Building SPLADE index with {len(documents)} documents...")
        
        # Mã hóa tất cả văn bản thành sparse vectors
        doc_vectors = self.encode_documents(documents)
        
        # Tạo vstack của tất cả sparse matrices
        self.doc_vectors = sp.vstack(doc_vectors)
        
        # Tạo lookup từ index đến document
        self.document_lookup = {i: doc for i, doc in enumerate(documents)}
        
        print(f"SPLADE index built with {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Tìm kiếm văn bản liên quan đến truy vấn
        
        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả trả về
            
        Returns:
            Danh sách các văn bản liên quan nhất
        """
        if self.doc_vectors is None:
            raise ValueError("Document index has not been built yet")
        
        # Mã hóa truy vấn thành sparse vector
        query_vector = self.encode_query(query)
        
        # Tính toán dot product (tương tự như cosine similarity cho sparse vectors)
        scores = self.doc_vectors.dot(query_vector.T).toarray().flatten()
        
        # Lấy top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Tạo kết quả
        results = []
        for i, idx in enumerate(top_indices):
            if idx in self.document_lookup:
                doc = self.document_lookup[idx].copy()
                doc["score"] = float(scores[idx])
                doc["rank"] = i
                results.append(doc)
        
        return results
    
    def get_query_tokens(self, query: str, min_weight: float = 0.1) -> Dict[str, float]:
        """
        Trả về các token và trọng số từ SPLADE cho truy vấn
        
        Args:
            query: Câu truy vấn
            min_weight: Trọng số tối thiểu để giữ lại token
            
        Returns:
            Dict mapping token -> weight
        """
        # Mã hóa truy vấn
        query_vector = self.encode_query(query)
        
        # Lấy các token IDs và weights
        indices = query_vector.indices
        weights = query_vector.data
        
        # Chỉ giữ những token có trọng số >= min_weight
        filtered_indices = [idx for i, idx in enumerate(indices) if weights[i] >= min_weight]
        filtered_weights = [weights[i] for i, idx in enumerate(indices) if weights[i] >= min_weight]
        
        # Chuyển đổi token IDs thành tokens
        tokens = {}
        for i, idx in enumerate(filtered_indices):
            if idx in self.id2token:
                token = self.id2token[idx]
                tokens[token] = filtered_weights[i]
        
        # Sắp xếp theo trọng số giảm dần
        sorted_tokens = {k: v for k, v in sorted(tokens.items(), key=lambda x: x[1], reverse=True)}
        
        return sorted_tokens
        
def retrieve_with_splade(query: str, top_k: int = 10) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Truy xuất văn bản sử dụng mô hình SPLADE
    
    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả trả về
        
    Returns:
        Tuple(sparse_vector, results):
            - sparse_vector: Dict token -> weight biểu diễn truy vấn
            - results: Danh sách kết quả truy xuất
    """
    # Khởi tạo retriever
    retriever = SPLADERetriever()
    
    # Lấy token weights cho query
    query_tokens = retriever.get_query_tokens(query)
    
    # Tìm kiếm nếu có index
    try:
        results = retriever.search(query, top_k)
        print(f"Retrieved {len(results)} results with SPLADE")
    except ValueError:
        # Nếu chưa có index, trả về kết quả mẫu
        print("SPLADE index not found. Using sample results for demonstration.")
        results = _get_sample_results(query, query_tokens, top_k)
    
    return query_tokens, results

def _get_sample_results(query: str, query_tokens: Dict[str, float], top_k: int) -> List[Dict[str, Any]]:
    """
    Tạo kết quả mẫu khi chưa có index
    """
    # Tạo các kết quả mẫu dựa trên từ khóa trong truy vấn
    sample_results = []
    
    # Lấy ra các token quan trọng nhất từ query
    important_tokens = list(query_tokens.keys())[:3]
    important_text = ", ".join(important_tokens)
    
    # Tạo kết quả mẫu
    for i in range(min(top_k, 10)):
        sample_results.append({
            "id": f"S{i+1:02d}",
            "text": f"Sample medical text related to {important_text}, result {i+1}",
            "score": 0.85 - (i*0.05),
            "rank": i
        })
    
    return sample_results 