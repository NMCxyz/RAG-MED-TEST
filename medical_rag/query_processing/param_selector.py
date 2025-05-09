"""
Parameter Selector Module
------------------------
Module chọn các tham số phù hợp cho quá trình truy xuất dựa trên
độ phức tạp của câu hỏi y khoa.
"""

from typing import Dict, Any

def select_params(complexity_tag: str) -> Dict[str, Any]:
    """
    Chọn tham số phù hợp dựa trên độ phức tạp của câu hỏi
    
    Args:
        complexity_tag: Nhãn độ phức tạp ("simple", "medium", "complex")
        
    Returns:
        Dict các tham số cho quá trình truy xuất và xử lý
    """
    print(f"[DEMO] Selecting parameters for complexity: {complexity_tag}")
    
    # Bảng tham số dựa trên độ phức tạp
    # Các tham số bao gồm:
    # - k: Tham số alpha trong MixPR (PageRank cá nhân hoá)
    # - topK: Số lượng node lớn nhất trong đồ thị con
    # - num_rewrites: Số lượng viết lại truy vấn
    # - sparse_weight: Trọng số cho sparse retrieval
    # - dense_weight: Trọng số cho dense retrieval
    # - max_reasoning_steps: Số bước suy luận tối đa
    # - temperature: Nhiệt độ cho LLM
    param_table = {
        "simple": {
            "k": 0.5,            # Alpha thấp -> đồ thị nhỏ hơn, tập trung hơn
            "topK": 2000,        # Ít nút hơn
            "num_rewrites": 1,   # Không cần nhiều viết lại
            "sparse_weight": 0.7, # Chú trọng vào từ khoá
            "dense_weight": 0.3,
            "max_reasoning_steps": 3,
            "temperature": 0.3   # Ít ngẫu nhiên hơn
        },
        "medium": {
            "k": 0.6,
            "topK": 4000,
            "num_rewrites": 2,
            "sparse_weight": 0.5,
            "dense_weight": 0.5,
            "max_reasoning_steps": 5, 
            "temperature": 0.5
        },
        "complex": {
            "k": 0.8,            # Alpha cao -> đồ thị lớn hơn, khám phá nhiều hơn
            "topK": 6000,        # Nhiều nút hơn
            "num_rewrites": 3,   # Nhiều viết lại hơn
            "sparse_weight": 0.3, # Chú trọng vào ngữ nghĩa
            "dense_weight": 0.7,
            "max_reasoning_steps": 7,
            "temperature": 0.7   # Nhiều ngẫu nhiên hơn
        }
    }
    
    # Chọn tham số phù hợp
    if complexity_tag not in param_table:
        print(f"[WARNING] Unknown complexity tag: {complexity_tag}, defaulting to 'medium'")
        params = param_table["medium"]
    else:
        params = param_table[complexity_tag]
    
    # Hiển thị các tham số đã chọn
    print(f"[DEMO] Selected parameters:")
    for key, value in params.items():
        print(f"       - {key}: {value}")
    
    return params

def adjust_params_for_query_type(params: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Điều chỉnh tham số dựa trên loại truy vấn cụ thể
    
    Args:
        params: Dict tham số ban đầu
        query: Câu truy vấn
        
    Returns:
        Dict tham số đã điều chỉnh
    """
    import re
    
    # Sao chép tham số để không thay đổi bản gốc
    adjusted_params = params.copy()
    
    # Điều chỉnh cho trường hợp đặc biệt
    
    # Truy vấn về so sánh thuốc
    if re.search(r"(compare|versus|vs|difference between).*?(treatment|medication|drug|therapy)", query, re.IGNORECASE):
        adjusted_params["num_rewrites"] = max(params["num_rewrites"], 3)  # Ít nhất 3 rewrites
        adjusted_params["max_reasoning_steps"] = max(params["max_reasoning_steps"], 6)  # Nhiều bước suy luận
    
    # Truy vấn về liệu pháp mới
    if re.search(r"(new|novel|recent|latest|emerging).*?(treatment|therapy)", query, re.IGNORECASE):
        adjusted_params["k"] = 0.85  # Mở rộng đồ thị nhiều hơn
        adjusted_params["temperature"] = 0.6  # Khuyến khích đa dạng
    
    # Truy vấn về tác dụng phụ
    if re.search(r"(side effect|adverse|complication|risk|safety)", query, re.IGNORECASE):
        adjusted_params["sparse_weight"] = 0.6  # Tăng trọng số từ khoá
        adjusted_params["topK"] = int(adjusted_params["topK"] * 1.2)  # Mở rộng tìm kiếm
    
    # Truy vấn đơn giản về định nghĩa
    if re.search(r"(what is|define|explain|describe).*?", query, re.IGNORECASE) and len(query.split()) < 7:
        adjusted_params["k"] = 0.3  # Thu hẹp đồ thị
        adjusted_params["num_rewrites"] = 1  # Giảm số lượng viết lại
    
    # Kiểm tra xem có sự thay đổi nào không
    changed = any(adjusted_params[k] != params[k] for k in params)
    if changed:
        print(f"[DEMO] Adjusted parameters for query type:")
        for key in params:
            if adjusted_params[key] != params[key]:
                print(f"       - {key}: {params[key]} -> {adjusted_params[key]}")
    
    return adjusted_params 