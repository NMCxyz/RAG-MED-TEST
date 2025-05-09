"""
Complexity Classifier Module
---------------------------
Module phân loại độ phức tạp của câu hỏi y khoa để lựa chọn
các tham số phù hợp cho việc truy xuất và trả lời.
"""

import re
from typing import Dict, List, Optional, Union

def classify_complexity(query: str) -> str:
    """
    Phân loại độ phức tạp của câu hỏi y khoa
    
    Args:
        query: Câu hỏi cần phân loại
        
    Returns:
        Nhãn độ phức tạp: "simple", "medium", hoặc "complex"
    """
    # DEMO: Giả lập một mô hình RoBERTa-cls đơn giản
    
    print(f"[DEMO] Analyzing query complexity: '{query}'")
    
    # Đặc trưng độ phức tạp
    # 1. Độ dài câu hỏi
    length = len(query.split())
    
    # 2. Số lượng thuật ngữ y khoa
    medical_terms = [
        "treatment", "therapy", "medication", "drug", "diagnosis",
        "prognosis", "etiology", "pathophysiology", "complication",
        "symptom", "syndrome", "disease", "disorder", "condition",
        "contraindication", "indication", "dosage", "administration",
        "mechanism", "risk factor", "prevention", "screening",
        "osteoporosis", "bisphosphonate", "bone density", "RANKL",
        "osteoblast", "osteoclast", "resorption", "fracture"
    ]
    
    term_count = sum(1 for term in medical_terms if term.lower() in query.lower())
    
    # 3. Số lượng câu hỏi so sánh (dùng "versus", "compared to", "better than", etc.)
    comparison_patterns = [
        r"versus", r"vs", r"compared to", r"better than", r"difference between",
        r"advantages", r"disadvantages", r"efficacy", r"effective"
    ]
    
    has_comparison = any(re.search(pattern, query, re.IGNORECASE) for pattern in comparison_patterns)
    
    # 4. Số lượng câu hỏi đa phần ("and", nhiều mệnh đề)
    complex_patterns = [
        r"what.*and.*why", r"how.*and.*when", r"why.*and.*how",
        r"what are the mechanisms.*and.*clinical implications",
        r"both.*and", r"not only.*but also", r"as well as"
    ]
    
    has_complex_structure = any(re.search(pattern, query, re.IGNORECASE) for pattern in complex_patterns)
    
    # Phân loại dựa trên các đặc trưng
    if (length > 15 or term_count >= 4 or has_comparison or has_complex_structure):
        complexity = "complex"
    elif (length > 8 or term_count >= 2):
        complexity = "medium"
    else:
        complexity = "simple"
    
    # Trường hợp đặc biệt - câu hỏi về liệu pháp mới luôn phức tạp
    if re.search(r"new|novel|emerging|recent|latest|state.?of.?art|advance", query, re.IGNORECASE):
        complexity = "complex"
    
    print(f"[DEMO] Query complexity classification: {complexity}")
    print(f"       - Length: {length} words")
    print(f"       - Medical terms: {term_count}")
    print(f"       - Has comparison: {has_comparison}")
    print(f"       - Has complex structure: {has_complex_structure}")
    
    return complexity

def get_complexity_features(query: str) -> Dict[str, Union[int, bool, float]]:
    """
    Trích xuất các đặc trưng độ phức tạp của câu hỏi y khoa
    
    Args:
        query: Câu hỏi cần phân tích
        
    Returns:
        Dict các đặc trưng độ phức tạp
    """
    # DEMO: Giống như hàm phân loại, nhưng trả về các đặc trưng thô
    # Trong thực tế, đây là feature extraction từ BERT/RoBERTa embedding
    
    length = len(query.split())
    
    # Thuật ngữ y khoa
    medical_terms = [
        "treatment", "therapy", "medication", "drug", "diagnosis",
        "prognosis", "etiology", "pathophysiology", "complication",
        "symptom", "syndrome", "disease", "disorder", "condition",
        "contraindication", "indication", "dosage", "administration",
        "mechanism", "risk factor", "prevention", "screening",
        "osteoporosis", "bisphosphonate", "bone density", "RANKL",
        "osteoblast", "osteoclast", "resorption", "fracture"
    ]
    
    term_count = sum(1 for term in medical_terms if term.lower() in query.lower())
    
    # Dạng câu hỏi
    question_types = {
        "what": bool(re.search(r"\bwhat\b", query, re.IGNORECASE)),
        "how": bool(re.search(r"\bhow\b", query, re.IGNORECASE)),
        "why": bool(re.search(r"\bwhy\b", query, re.IGNORECASE)),
        "when": bool(re.search(r"\bwhen\b", query, re.IGNORECASE)),
        "which": bool(re.search(r"\bwhich\b", query, re.IGNORECASE)),
        "comparison": bool(re.search(r"versus|vs|compared|better|difference", query, re.IGNORECASE)),
        "is_new_therapy": bool(re.search(r"new|novel|emerging|recent|latest", query, re.IGNORECASE))
    }
    
    # Trả về toàn bộ đặc trưng
    features = {
        "length": length,
        "term_count": term_count,
        **question_types
    }
    
    return features 