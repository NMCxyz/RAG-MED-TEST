"""
Semantic Chunker Module
-----------------------
Module xử lý tài liệu y khoa, tách thành các chunk có ngữ nghĩa hoàn chỉnh
thay vì cắt cứng theo số ký tự để tránh cắt đứt câu/bảng/biểu đồ y khoa.
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple

def semantic_chunk_pdf(pdf_path: str, 
                      min_chunk_size: int = 200,
                      max_chunk_size: int = 1000,
                      chunk_overlap: int = 50) -> List[Dict[str, str]]:
    """
    Tách PDF y khoa thành các chunk có ý nghĩa hoàn chỉnh.
    
    Args:
        pdf_path: Đường dẫn tới file PDF
        min_chunk_size: Kích thước tối thiểu của chunk (ký tự)
        max_chunk_size: Kích thước tối đa của chunk (ký tự)
        chunk_overlap: Số ký tự overlap giữa các chunk liên tiếp
        
    Returns:
        List các chunk dạng {"id": str, "text": str}
    """
    print(f"[DEMO] Reading PDF from {pdf_path}")
    
    # DEMO: Thay vì đọc PDF thực tế, ta dùng text mẫu
    sample_text = """
    Osteoporosis is a skeletal disorder characterized by compromised bone 
    strength predisposing to an increased risk of fracture. Bone strength 
    primarily reflects the integration of bone density and bone quality.
    
    Bisphosphonates are the cornerstone of osteoporosis treatment, with 
    alendronate being the most commonly prescribed. These medications work 
    by inhibiting osteoclast-mediated bone resorption, thus reducing bone turnover.
    
    TABLE 1: Common Bisphosphonates
    - Alendronate (Fosamax): 70mg weekly
    - Risedronate (Actonel): 35mg weekly
    - Ibandronate (Boniva): 150mg monthly
    - Zoledronic acid (Reclast): 5mg IV yearly
    
    Newer therapeutic approaches include:
    
    1. Denosumab (Prolia): A fully human monoclonal antibody that inhibits RANKL,
    thereby decreasing osteoclast formation and activity. Administered as a 
    subcutaneous injection every 6 months.
    
    2. Anabolic agents like teriparatide and abaloparatide stimulate bone formation
    and are typically reserved for patients with severe osteoporosis or those who
    have failed other therapies.
    
    3. Romosozumab (Evenity): A sclerostin inhibitor that both increases bone
    formation and decreases bone resorption. Limited to 12-month treatment course.
    
    Combination therapy approaches and sequential treatment regimens are being
    studied to optimize long-term efficacy while minimizing potential adverse effects.
    """
    
    # Phát hiện ranh giới ngữ nghĩa (đầu đoạn, bảng, danh sách)
    semantic_boundaries = _detect_semantic_boundaries(sample_text)
    
    # Tách theo ranh giới ngữ nghĩa và kích thước
    chunks = _create_semantic_chunks(sample_text, semantic_boundaries, 
                                    min_chunk_size, max_chunk_size, chunk_overlap)
    
    # Gán ID cho các chunk
    for i, chunk in enumerate(chunks):
        chunk["id"] = f"C{i+1:02d}"
    
    return chunks

def _detect_semantic_boundaries(text: str) -> List[int]:
    """
    Phát hiện vị trí ranh giới ngữ nghĩa trong văn bản y khoa:
    - Đầu đoạn văn mới
    - Bảng, biểu đồ
    - Tiêu đề, mục lớn
    - Danh sách đánh số/bullet
    
    Args:
        text: Văn bản cần phân tích
        
    Returns:
        Danh sách các vị trí (index) ranh giới ngữ nghĩa
    """
    # DEMO: Đơn giản hoá - tìm ranh giới là dòng trống
    boundaries = [0]  # Luôn bắt đầu từ đầu văn bản
    
    # Tìm dòng trống (2 ký tự xuống dòng liên tiếp)
    for match in re.finditer(r'\n\s*\n', text):
        boundaries.append(match.start())
    
    # Thêm các mẫu đánh dấu bảng/danh sách
    table_markers = [m.start() for m in re.finditer(r'TABLE|FIGURE|CHART', text, re.IGNORECASE)]
    list_markers = [m.start() for m in re.finditer(r'\n\s*[0-9]+\.\s|\n\s*•\s|\n\s*-\s', text)]
    
    # Kết hợp tất cả ranh giới và sắp xếp
    all_boundaries = sorted(set(boundaries + table_markers + list_markers))
    
    return all_boundaries

def _create_semantic_chunks(text: str, 
                           boundaries: List[int],
                           min_size: int,
                           max_size: int,
                           overlap: int) -> List[Dict[str, str]]:
    """
    Tạo các chunk văn bản từ ranh giới ngữ nghĩa, đảm bảo kích thước phù hợp
    
    Args:
        text: Văn bản gốc
        boundaries: Vị trí các ranh giới ngữ nghĩa
        min_size: Kích thước tối thiểu của chunk
        max_size: Kích thước tối đa của chunk
        overlap: Số ký tự overlap giữa các chunk
    
    Returns:
        Danh sách các chunk {"text": str}
    """
    chunks = []
    current_start = 0
    
    for i in range(1, len(boundaries)):
        # Nếu kích thước chunk hiện tại quá nhỏ, tiếp tục mở rộng
        if boundaries[i] - current_start < min_size and i < len(boundaries) - 1:
            continue
            
        # Nếu kích thước chunk hiện tại quá lớn, chia nhỏ
        if boundaries[i] - current_start > max_size:
            # Chia chunk lớn thành nhiều chunk nhỏ hơn
            sub_chunks = _split_large_chunk(text[current_start:boundaries[i]], max_size, overlap)
            for sub_chunk in sub_chunks:
                chunks.append({"text": sub_chunk})
        else:
            # Thêm chunk hiện tại
            chunks.append({"text": text[current_start:boundaries[i]].strip()})
        
        # Cập nhật vị trí bắt đầu mới, trừ đi overlap
        current_start = max(0, boundaries[i] - overlap)
    
    # Xử lý phần còn lại của văn bản
    if current_start < len(text):
        chunks.append({"text": text[current_start:].strip()})
    
    return chunks

def _split_large_chunk(text: str, max_size: int, overlap: int) -> List[str]:
    """
    Chia chunk lớn thành nhiều chunk nhỏ hơn, cố gắng cắt ở ranh giới câu
    
    Args:
        text: Chunk cần chia nhỏ
        max_size: Kích thước tối đa
        overlap: Kích thước overlap
        
    Returns:
        Danh sách các chunk nhỏ hơn
    """
    # Tìm vị trí kết thúc câu (dấu chấm, chấm than, chấm hỏi, xuống dòng)
    sentence_endings = [m.start() for m in re.finditer(r'[.!?]\s+|\n', text)]
    
    result = []
    current_start = 0
    
    while current_start < len(text):
        # Tìm vị trí kết thúc phù hợp gần max_size nhất
        best_end = len(text)
        for end in sentence_endings:
            if end - current_start > max_size:
                break
            best_end = end + 1  # +1 để bao gồm dấu kết thúc câu
        
        # Nếu không tìm được ranh giới câu, cắt cứng tại max_size
        if best_end == len(text) and current_start + max_size < len(text):
            best_end = current_start + max_size
        
        # Thêm chunk mới
        result.append(text[current_start:best_end].strip())
        
        # Cập nhật vị trí bắt đầu mới, trừ đi overlap
        current_start = max(0, best_end - overlap)
        
        # Tránh lặp vô hạn nếu không thể tiến triển
        if current_start == len(text) or best_end == len(text):
            break
    
    return result 