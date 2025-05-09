"""
Seed Selection Module
-------------------
Module chọn các nút hạt giống (seed nodes) trong đồ thị tri thức dựa trên
vector truy vấn để khởi tạo quá trình đi bộ đồ thị (graph walk).
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional

def select_seed_nodes(query_vector: np.ndarray, 
                     num_seeds: int = 5, 
                     similarity_threshold: float = 0.6) -> List[int]:
    """
    Chọn các nút hạt giống từ đồ thị tri thức y khoa dựa trên độ tương đồng cosine
    
    Args:
        query_vector: Vector đặc trưng của truy vấn
        num_seeds: Số lượng nút hạt giống cần chọn
        similarity_threshold: Ngưỡng độ tương đồng tối thiểu
        
    Returns:
        Danh sách các ID nút hạt giống
    """
    print(f"[DEMO] Selecting {num_seeds} seed nodes with threshold {similarity_threshold}")
    
    # DEMO: Mô phỏng việc tìm kiếm nút tương đồng trong đồ thị
    # Trong thực tế, sẽ tính similarity với các node-embeddings từ đồ thị thực
    
    # Tạo embedding giả cho các nút trong đồ thị tri thức
    node_embeddings = _get_demo_node_embeddings()
    
    # Tính độ tương đồng cosine giữa query_vector và tất cả node_embeddings
    similarities = {}
    for node_id, node_vector in node_embeddings.items():
        sim = np.dot(query_vector, node_vector)
        similarities[node_id] = sim
    
    # Lọc theo ngưỡng và sắp xếp theo độ tương đồng giảm dần
    filtered_nodes = [(node_id, sim) for node_id, sim in similarities.items() 
                     if sim >= similarity_threshold]
    filtered_nodes.sort(key=lambda x: x[1], reverse=True)
    
    # Chọn top-k nút có độ tương đồng cao nhất
    seed_nodes = [node_id for node_id, _ in filtered_nodes[:num_seeds]]
    
    # Nếu không đủ số lượng, giảm ngưỡng và chọn thêm
    if len(seed_nodes) < num_seeds:
        print(f"[DEMO] Warning: Only found {len(seed_nodes)} nodes above threshold")
        
        # Sắp xếp lại tất cả các nút theo độ tương đồng
        all_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Thêm nút cho đến khi đủ số lượng
        for node_id, sim in all_nodes:
            if node_id not in seed_nodes:
                seed_nodes.append(node_id)
                if len(seed_nodes) >= num_seeds:
                    break
    
    print(f"[DEMO] Selected seed nodes: {seed_nodes}")
    
    return seed_nodes

def _get_demo_node_embeddings() -> Dict[int, np.ndarray]:
    """
    Tạo các embeddings mẫu cho các nút trong đồ thị tri thức y khoa
    
    Returns:
        Dict ánh xạ ID nút -> vector embedding
    """
    # DEMO: Tạo embeddings ngẫu nhiên cho các nút
    np.random.seed(42)  # Cố định seed để kết quả nhất quán
    
    # Dict lưu embeddings của các nút
    node_embeddings = {}
    
    # Tạo embeddings ngẫu nhiên cho 1000 nút đầu tiên
    for node_id in range(1, 1001):
        # Vector ngẫu nhiên 768 chiều (cùng kích thước với query_vector)
        vec = np.random.randn(768).astype(np.float32)
        # Chuẩn hóa vector
        vec = vec / np.linalg.norm(vec)
        
        node_embeddings[node_id] = vec
    
    # Thêm một số nút đặc biệt tương ứng với các thực thể y khoa
    special_nodes = {
        # Drug nodes (ID: 1000-1099)
        1001: "alendronate",
        1002: "denosumab",
        1003: "teriparatide",
        1004: "romosozumab",
        1005: "zoledronic acid",
        
        # Disease nodes (ID: 1100-1199)
        1101: "osteoporosis",
        1102: "osteopenia",
        1103: "fracture",
        
        # Mechanism nodes (ID: 1200-1299)
        1201: "bone resorption",
        1202: "bone formation",
        1203: "RANKL inhibition",
        
        # Anatomy nodes (ID: 1300-1399)
        1301: "bone",
        1302: "skeleton",
    }
    
    # Tạo embeddings đặc biệt cho các nút y khoa (ngẫu nhiên nhưng có định hướng)
    for node_id, term in special_nodes.items():
        # Tạo vector ngẫu nhiên
        vec = np.random.randn(768).astype(np.float32)
        # Chuẩn hóa
        vec = vec / np.linalg.norm(vec)
        
        node_embeddings[node_id] = vec
    
    return node_embeddings

def get_node_info(node_id: int) -> Dict[str, Any]:
    """
    Lấy thông tin về một nút trong đồ thị tri thức
    
    Args:
        node_id: ID của nút
        
    Returns:
        Dict thông tin về nút
    """
    # DEMO: Mô phỏng việc truy xuất thông tin nút từ KG
    
    # Thông tin về các nút đặc biệt
    special_nodes = {
        # Drug nodes
        1001: {"type": "Drug", "name": "Alendronate", 
               "desc": "Bisphosphonate drug used to treat osteoporosis"},
        1002: {"type": "Drug", "name": "Denosumab", 
               "desc": "Monoclonal antibody that inhibits RANKL"},
        1003: {"type": "Drug", "name": "Teriparatide", 
               "desc": "Recombinant parathyroid hormone used as anabolic agent"},
        1004: {"type": "Drug", "name": "Romosozumab", 
               "desc": "Sclerostin inhibitor that increases bone formation"},
        1005: {"type": "Drug", "name": "Zoledronic acid", 
               "desc": "Potent bisphosphonate given intravenously"},
        
        # Disease nodes
        1101: {"type": "Disease", "name": "Osteoporosis", 
               "desc": "Skeletal disorder characterized by decreased bone density"},
        1102: {"type": "Disease", "name": "Osteopenia", 
               "desc": "Decreased bone density not severe enough to be classified as osteoporosis"},
        1103: {"type": "Disease", "name": "Fracture", 
               "desc": "Break in bone continuity"},
        
        # Mechanism nodes
        1201: {"type": "Mechanism", "name": "Bone resorption", 
               "desc": "Process of bone breakdown by osteoclasts"},
        1202: {"type": "Mechanism", "name": "Bone formation", 
               "desc": "Process of new bone creation by osteoblasts"},
        1203: {"type": "Mechanism", "name": "RANKL inhibition", 
               "desc": "Inhibition of RANK ligand, which activates osteoclasts"},
        
        # Anatomy nodes
        1301: {"type": "Anatomy", "name": "Bone", 
               "desc": "Rigid tissue that forms the skeleton"},
        1302: {"type": "Anatomy", "name": "Skeleton", 
               "desc": "Framework of bones that supports the body"},
    }
    
    # Nếu là nút đặc biệt, trả về thông tin chi tiết
    if node_id in special_nodes:
        return special_nodes[node_id]
    
    # Nếu không, tạo thông tin mặc định dựa trên ID
    node_type = "Concept"
    if 1 <= node_id <= 1000:
        node_type = "Concept"
    elif 1001 <= node_id <= 1099:
        node_type = "Drug"
    elif 1100 <= node_id <= 1199:
        node_type = "Disease"
    elif 1200 <= node_id <= 1299:
        node_type = "Mechanism"
    elif 1300 <= node_id <= 1399:
        node_type = "Anatomy"
    
    return {
        "type": node_type,
        "name": f"Node_{node_id}",
        "desc": f"Medical {node_type.lower()} node with ID {node_id}"
    } 