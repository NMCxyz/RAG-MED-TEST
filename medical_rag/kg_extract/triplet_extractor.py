"""
Triplet Extractor Module
------------------------
Module trích xuất các triple (chủ thể, quan hệ, khách thể) từ text y khoa
để tạo đồ thị tri thức y học (KG).
"""

import re
from typing import Dict, List, Tuple, Set, Optional, Union

# Kiểu dữ liệu cho triplet
Triplet = Tuple[str, str, str]  # (head, relation, tail)

def extract_triplets_from_chunk(chunk: Dict[str, str]) -> List[Triplet]:
    """
    Trích xuất các triplet (subject, relation, object) từ chunk văn bản y khoa
    
    Args:
        chunk: Chunk văn bản dạng {"id": str, "text": str}
        
    Returns:
        List các triplet (head, relation, tail)
    """
    # DEMO: Mô phỏng quá trình trích xuất, không gọi mô hình thực
    
    # Lấy text từ chunk
    text = chunk["text"]
    chunk_id = chunk.get("id", "unknown")
    
    print(f"[DEMO] Extracting triplets from chunk {chunk_id}")
    
    # Các mẫu quan hệ y khoa phổ biến
    med_relations = {
        "treats": [r"(?i)(treats|used in|treatment of|therapy for|indicated for)"],
        "causes": [r"(?i)(causes|induces|leads to|results in)"],
        "prevents": [r"(?i)(prevents|reduces risk of|prophylaxis for)"],
        "diagnoses": [r"(?i)(diagnoses|detects|identifies|screening for)"],
        "inhibits": [r"(?i)(inhibits|blocks|reduces|decreases)"],
        "activates": [r"(?i)(activates|stimulates|increases|enhances)"],
        "contraindicates": [r"(?i)(contraindicates|avoid in|not recommended for)"],
        "administers": [r"(?i)(administers|given as|delivered via|injected as)"],
    }
    
    # Thực thể y khoa (giả lập dùng từ điển đơn giản)
    medical_entities = {
        "Drug": ["bisphosphonate", "alendronate", "risedronate", "ibandronate", 
                "zoledronic acid", "denosumab", "teriparatide", "abaloparatide", 
                "romosozumab", "raloxifene", "estrogen", "calcitonin"],
        "Disease": ["osteoporosis", "osteopenia", "fracture", "bone loss", 
                   "hypocalcemia", "osteonecrosis"],
        "Protein": ["RANKL", "sclerostin", "estrogen receptor", "osteoblast", 
                   "osteoclast", "bone resorption", "bone formation"],
        "Anatomy": ["bone", "skeleton", "vertebra", "hip", "wrist", "joint"],
    }
    
    # Danh sách lưu kết quả
    triplets = []
    
    # DEMO: Dùng regex đơn giản để phát hiện (không phải NER thực)
    
    # Tìm các cặp thực thể và quan hệ
    for rel_type, patterns in med_relations.items():
        for pattern in patterns:
            # Tìm tất cả các xuất hiện của mẫu quan hệ này
            for match in re.finditer(pattern, text):
                rel_pos = match.start()
                
                # Tìm thực thể trước và sau vị trí relation
                head_entity = None
                head_type = None
                tail_entity = None
                tail_type = None
                
                # Tìm quanh vùng phù hợp với pattern
                context_before = text[max(0, rel_pos-100):rel_pos]
                context_after = text[match.end():min(len(text), match.end()+100)]
                
                # Tìm thực thể trước và sau
                for entity_type, entities in medical_entities.items():
                    # Tìm head entity (chủ thể) trong context trước
                    for entity in entities:
                        if entity.lower() in context_before.lower():
                            head_entity = entity
                            head_type = entity_type
                            break
                    
                    # Tìm tail entity (khách thể) trong context sau
                    for entity in entities:
                        if entity.lower() in context_after.lower():
                            tail_entity = entity
                            tail_type = entity_type
                            break
                
                # Nếu tìm được cả chủ thể và khách thể, tạo triplet
                if head_entity and tail_entity:
                    # Thêm typing cho rõ ràng hơn
                    head = f"{head_type}::{head_entity}"
                    tail = f"{tail_type}::{tail_entity}"
                    
                    triplet = (head, rel_type, tail)
                    triplets.append(triplet)
    
    # DEMO: Nếu không tìm thấy triplets nào từ regex, thêm một số mẫu cố định
    if not triplets and "bisphosphonate" in text.lower():
        triplets = [
            ("Drug::Alendronate", "treats", "Disease::Osteoporosis"),
            ("Drug::Alendronate", "inhibits", "Protein::Bone resorption"),
            ("Drug::Bisphosphonate", "administers", "Anatomy::Bone")
        ]
    elif not triplets and "denosumab" in text.lower():
        triplets = [
            ("Drug::Denosumab", "inhibits", "Protein::RANKL"),
            ("Drug::Denosumab", "administers", "Anatomy::Bone"),
            ("Protein::RANKL", "activates", "Protein::Osteoclast")
        ]
    elif not triplets:
        triplets = [
            ("Drug::Teriparatide", "activates", "Protein::Bone formation"),
            ("Disease::Osteoporosis", "causes", "Disease::Fracture")
        ]
    
    return triplets

def load_triplets_to_kg(triplets: List[Triplet], kg_connection_string: str) -> None:
    """
    Tải các triplet vào đồ thị tri thức (Neo4j KG)
    
    Args:
        triplets: Danh sách các triplet (head, relation, tail)
        kg_connection_string: Connection string đến Neo4j
    """
    # DEMO: Giả lập việc tải triplet vào Neo4j
    print(f"[DEMO] Loading {len(triplets)} triplets to knowledge graph")
    
    # Thực tế sẽ dùng driver Neo4j và Cypher query
    cypher_queries = []
    
    for head, relation, tail in triplets:
        # Parse type và entity
        head_type, head_entity = head.split("::")
        tail_type, tail_entity = tail.split("::")
        
        # Tạo Cypher query
        query = f"""
        MERGE (h:{head_type} {{name: '{head_entity}'}})
        MERGE (t:{tail_type} {{name: '{tail_entity}'}})
        MERGE (h)-[r:{relation}]->(t)
        RETURN h, r, t
        """
        
        cypher_queries.append(query)
    
    print(f"[DEMO] Generated {len(cypher_queries)} Cypher queries")
    # Trong ứng dụng thực sẽ thực thi các query này
    
    # Ví dụ demo một query:
    if cypher_queries:
        print(f"Sample query: {cypher_queries[0]}")
    
    return 