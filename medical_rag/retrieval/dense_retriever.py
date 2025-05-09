"""
Dense Retriever Module
--------------------
Module truy xuất dense vector sử dụng mô hình BGE-M3 để tìm văn bản
liên quan đến truy vấn.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss

class BGEDenseRetriever:
    def __init__(self, model_name="BAAI/bge-m3", 
                 index_path=None, 
                 device=None):
        """
        Khởi tạo Dense Retriever với BGE-M3
        
        Args:
            model_name: Tên mô hình BGE trên HuggingFace
            index_path: Đường dẫn tới FAISS index (nếu có)
            device: Thiết bị để chạy mô hình (None: tự động)
        """
        # Xác định thiết bị: GPU nếu có, ngược lại CPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Khởi tạo mô hình BGE-M3
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Khởi tạo các thuộc tính cho FAISS index
        self.faiss_index = None
        self.document_lookup = {}
        
        # Nếu có sẵn index, load nó
        if index_path:
            self.load_index(index_path)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Mã hóa câu truy vấn thành vector
        
        Args:
            query: Câu truy vấn
            
        Returns:
            Vector biểu diễn câu truy vấn
        """
        # Đặt mô hình ở chế độ eval
        self.model.eval()
        
        # Mã hóa câu truy vấn
        with torch.no_grad():
            embeddings = self.model.encode(
                query,
                normalize_embeddings=True,  # Chuẩn hóa để dùng inner product
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
        return embeddings
    
    def encode_documents(self, documents: List[Dict[str, str]]) -> np.ndarray:
        """
        Mã hóa danh sách văn bản thành vectors
        
        Args:
            documents: Danh sách các văn bản 
                      [{"id": id, "text": text}, ...]
            
        Returns:
            Ma trận vectors biểu diễn văn bản
        """
        # Trích xuất chỉ phần văn bản
        texts = [doc["text"] for doc in documents]
        
        # Đặt mô hình ở chế độ eval
        self.model.eval()
        
        # Mã hóa văn bản
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,  # Chuẩn hóa để dùng inner product
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32  # Điều chỉnh theo GPU memory
            )
            
        return embeddings
    
    def build_index(self, documents: List[Dict[str, str]], save_path: Optional[str] = None):
        """
        Xây dựng FAISS index từ corpus
        
        Args:
            documents: Danh sách các văn bản [{"id": id, "text": text}, ...]
            save_path: Đường dẫn lưu index (nếu cần)
        """
        print(f"Building FAISS index with {len(documents)} documents...")
        
        # Mã hóa tất cả văn bản
        embeddings = self.encode_documents(documents)
        dim = embeddings.shape[1]  # Số chiều của embedding
        
        # Tạo FAISS index sử dụng inner product (vì embeddings đã được chuẩn hóa)
        self.faiss_index = faiss.IndexFlatIP(dim)
        
        # Thêm vectors vào index
        self.faiss_index.add(embeddings)
        
        # Tạo lookup để map từ index đến document
        self.document_lookup = {i: doc for i, doc in enumerate(documents)}
        
        print(f"Index built with {self.faiss_index.ntotal} vectors of {dim} dimensions")
        
        # Lưu index nếu cần
        if save_path:
            self._save_index(save_path)
    
    def load_index(self, index_path: str, lookup_path: Optional[str] = None):
        """
        Load FAISS index từ file
        
        Args:
            index_path: Đường dẫn tới FAISS index
            lookup_path: Đường dẫn tới document lookup (nếu lưu riêng)
        """
        import os
        import pickle
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(index_path)
        print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        
        # Load document lookup
        if lookup_path and os.path.exists(lookup_path):
            with open(lookup_path, 'rb') as f:
                self.document_lookup = pickle.load(f)
            print(f"Loaded document lookup with {len(self.document_lookup)} entries")
    
    def _save_index(self, save_path: str, lookup_path: Optional[str] = None):
        """
        Lưu FAISS index và document lookup
        
        Args:
            save_path: Đường dẫn lưu FAISS index
            lookup_path: Đường dẫn lưu document lookup (nếu cần lưu riêng)
        """
        import os
        import pickle
        
        # Lưu FAISS index
        faiss.write_index(self.faiss_index, save_path)
        print(f"Saved FAISS index to {save_path}")
        
        # Lưu document lookup
        if lookup_path:
            lookup_path_to_use = lookup_path
        else:
            # Mặc định lưu cùng thư mục với index, đổi đuôi
            lookup_path_to_use = os.path.splitext(save_path)[0] + "_lookup.pkl"
        
        with open(lookup_path_to_use, 'wb') as f:
            pickle.dump(self.document_lookup, f)
        print(f"Saved document lookup to {lookup_path_to_use}")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Tìm kiếm văn bản liên quan nhất đến câu truy vấn
        
        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả trả về
            
        Returns:
            Danh sách các văn bản liên quan nhất
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index has not been built or loaded")
        
        # Mã hóa câu truy vấn
        query_vector = self.encode_query(query)
        
        # Tìm kiếm top-k văn bản liên quan nhất
        scores, indices = self.faiss_index.search(
            query_vector.reshape(1, -1), top_k
        )
        
        # Kết quả trả về
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx in self.document_lookup:
                doc = self.document_lookup[idx].copy()
                doc["score"] = float(score)
                doc["rank"] = i
                results.append(doc)
        
        return results

def retrieve_with_bge_m3(query: str, top_k: int = 10) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Truy xuất văn bản sử dụng mô hình BGE-M3
    
    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả trả về
        
    Returns:
        Tuple(query_vector, results):
            - query_vector: Vector biểu diễn câu truy vấn
            - results: Danh sách kết quả truy xuất
    """
    # Khởi tạo retriever
    retriever = BGEDenseRetriever(model_name="BAAI/bge-m3")
    
    # Mã hóa câu truy vấn
    query_vector = retriever.encode_query(query)
    
    # Nếu đã có index, tìm kiếm
    try:
        results = retriever.search(query, top_k)
        print(f"Retrieved {len(results)} results with BGE-M3")
    except ValueError:
        # Nếu chưa có index, trả về kết quả mẫu để demo
        print("FAISS index not found. Using sample results for demonstration.")
        results = _get_sample_results(top_k)
    
    return query_vector, results

def _get_sample_results(top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Tạo kết quả mẫu khi chưa có index
    """
    sample_results = [
        {
            "id": f"C{i+1:02d}",
            "text": f"Sample medical text about osteoporosis treatment {i+1}",
            "score": 0.9 - (i*0.05),
            "rank": i
        }
        for i in range(min(top_k, 10))
    ]
    return sample_results 