"""
GFM Encoder Module
---------------
Module mã hóa đồ thị (Graph Encoding) sử dụng mô hình GFM (Graph Feature Mixer)
để biểu diễn tiểu đồ thị tri thức y khoa dưới dạng embedding.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def encode_graph_with_gfm(subgraph_nodes: List[int],
                        subgraph_edges: List[Tuple[int, str, int]],
                        hidden_dim: int = 256,
                        num_layers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mã hóa tiểu đồ thị tri thức y khoa sử dụng GFM Encoder
    
    Args:
        subgraph_nodes: Danh sách các nút trong tiểu đồ thị
        subgraph_edges: Danh sách các cạnh trong tiểu đồ thị
        hidden_dim: Kích thước vector biểu diễn ẩn
        num_layers: Số lớp Graph Transformer
        
    Returns:
        Tuple(node_embeddings, graph_embedding):
            - node_embeddings: Ma trận biểu diễn các nút (shape: [num_nodes, hidden_dim])
            - graph_embedding: Vector biểu diễn toàn bộ đồ thị (shape: [hidden_dim])
    """
    print(f"[DEMO] Encoding graph with GFM ({num_layers} layers, {hidden_dim} dimensions)")
    print(f"       Input: {len(subgraph_nodes)} nodes, {len(subgraph_edges)} edges")
    
    # DEMO: Mô phỏng quá trình mã hóa đồ thị với GFM
    # Trong thực tế, sẽ sử dụng mô hình GFM đã huấn luyện
    
    # Ánh xạ node_id -> vị trí trong ma trận
    node_to_idx = {node_id: idx for idx, node_id in enumerate(subgraph_nodes)}
    
    # Tạo ma trận kề dạng sparse
    num_nodes = len(subgraph_nodes)
    
    # Tạo danh sách các loại quan hệ
    relation_types = _extract_relation_types(subgraph_edges)
    relation_to_idx = {rel: idx for idx, rel in enumerate(relation_types)}
    
    # Tạo mask biểu thị cấu trúc đồ thị
    # Trong mô hình thực, đây sẽ là sparse adjacency matrix với nhiều kênh
    edge_index = []
    edge_type = []
    
    for src, rel, dst in subgraph_edges:
        if src in node_to_idx and dst in node_to_idx:
            src_idx = node_to_idx[src]
            dst_idx = node_to_idx[dst]
            rel_idx = relation_to_idx.get(rel, 0)
            
            edge_index.append((src_idx, dst_idx))
            edge_type.append(rel_idx)
    
    # Trích xuất đặc trưng nút ban đầu (khởi tạo từ node type và metadata)
    initial_node_features = _get_initial_node_features(subgraph_nodes, hidden_dim)
    
    # Mô phỏng quá trình lan truyền thông tin qua các lớp Graph Transformer
    node_embeddings = initial_node_features.copy()
    
    for layer in range(num_layers):
        # Mô phỏng đơn giản Message Passing trong GNN
        # Trong mô hình thực, đây sẽ là Transformer với self-attention
        
        # 1. Tạo messages từ các nút láng giềng
        messages = np.zeros_like(node_embeddings)
        
        for (src_idx, dst_idx), rel_idx in zip(edge_index, edge_type):
            # Mô phỏng thông điệp được tạo ra từ nút nguồn, điều chỉnh theo loại quan hệ
            # Trong thực tế, sẽ có ma trận transform phức tạp hơn
            message = node_embeddings[src_idx] * (1.0 / (1.0 + rel_idx * 0.1))
            messages[dst_idx] += message
        
        # 2. Cập nhật node embeddings
        # Mô phỏng đơn giản GRU update
        node_embeddings = 0.1 * messages + 0.9 * node_embeddings
        
        # 3. Thêm nhiễu ngẫu nhiên để tạo sự đa dạng
        noise = np.random.randn(*node_embeddings.shape) * 0.01
        node_embeddings += noise
        
        # 4. Chuẩn hóa lại
        norm = np.sqrt((node_embeddings * node_embeddings).sum(axis=1, keepdims=True))
        node_embeddings = node_embeddings / (norm + 1e-8)
    
    # Tạo graph embedding bằng cách tổng hợp (pooling) tất cả các node embeddings
    # Trong thực tế, sẽ sử dụng cơ chế attention phức tạp
    graph_embedding = np.mean(node_embeddings, axis=0)
    graph_embedding = graph_embedding / np.linalg.norm(graph_embedding)
    
    print(f"[DEMO] Generated node embeddings shape: {node_embeddings.shape}")
    print(f"       Graph embedding shape: {graph_embedding.shape}")
    
    return node_embeddings, graph_embedding

def _extract_relation_types(edges: List[Tuple[int, str, int]]) -> List[str]:
    """
    Trích xuất tất cả các loại quan hệ từ danh sách cạnh
    
    Args:
        edges: Danh sách cạnh với quan hệ
        
    Returns:
        Danh sách duy nhất các loại quan hệ
    """
    relation_types = set()
    
    for _, rel, _ in edges:
        relation_types.add(rel)
    
    return list(relation_types)

def _get_initial_node_features(nodes: List[int], hidden_dim: int) -> np.ndarray:
    """
    Tạo đặc trưng ban đầu cho các nút
    
    Args:
        nodes: Danh sách ID nút
        hidden_dim: Kích thước vector đặc trưng
        
    Returns:
        Ma trận đặc trưng nút ban đầu
    """
    from medical_rag.graph.seed_select import get_node_info
    
    num_nodes = len(nodes)
    node_features = np.zeros((num_nodes, hidden_dim))
    
    # Khởi tạo với nhiễu ngẫu nhiên nhỏ
    np.random.seed(42)
    node_features = np.random.randn(num_nodes, hidden_dim) * 0.01
    
    # Mã hóa loại nút
    node_type_encoding = {
        "Drug": 0,
        "Disease": 1,
        "Mechanism": 2,
        "Anatomy": 3,
        "Concept": 4
    }
    
    # Gán đặc trưng dựa trên loại nút
    for i, node_id in enumerate(nodes):
        # Lấy thông tin nút
        node_info = get_node_info(node_id)
        node_type = node_info.get("type", "Concept")
        
        # One-hot cho loại nút
        type_idx = node_type_encoding.get(node_type, 4)
        for j in range(5):
            node_features[i, j] = 1.0 if j == type_idx else 0.0
    
    # Chuẩn hóa
    norm = np.sqrt((node_features * node_features).sum(axis=1, keepdims=True))
    node_features = node_features / (norm + 1e-8)
    
    return node_features

def get_pairwise_distances(node_embeddings: np.ndarray) -> np.ndarray:
    """
    Tính toán ma trận khoảng cách cặp giữa các nút
    
    Args:
        node_embeddings: Ma trận biểu diễn các nút
        
    Returns:
        Ma trận khoảng cách cặp
    """
    num_nodes = node_embeddings.shape[0]
    
    # Tính khoảng cách cosine
    sim_matrix = np.zeros((num_nodes, num_nodes))
    
    # Tính tất cả các cặp khoảng cách
    for i in range(num_nodes):
        for j in range(num_nodes):
            # Khoảng cách cosine = 1 - cosine similarity
            sim = np.dot(node_embeddings[i], node_embeddings[j])
            sim_matrix[i, j] = sim
    
    # Chuyển đổi từ cosine similarity -> cosine distance
    dist_matrix = 1.0 - sim_matrix
    
    return dist_matrix

def get_node_attention_scores(query_embedding: np.ndarray, 
                            node_embeddings: np.ndarray) -> np.ndarray:
    """
    Tính toán điểm chú ý (attention score) giữa truy vấn và các nút
    
    Args:
        query_embedding: Vector biểu diễn truy vấn
        node_embeddings: Ma trận biểu diễn các nút
        
    Returns:
        Vector điểm chú ý
    """
    # Tính cosine similarity giữa truy vấn và mỗi nút
    attention_scores = np.zeros(node_embeddings.shape[0])
    
    for i in range(node_embeddings.shape[0]):
        attention_scores[i] = np.dot(query_embedding, node_embeddings[i])
    
    # Chuẩn hóa bằng softmax
    exp_scores = np.exp(attention_scores)
    attention_scores = exp_scores / exp_scores.sum()
    
    return attention_scores 