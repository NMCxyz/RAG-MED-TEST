"""
Query Rewriter Module
--------------------
Module viết lại truy vấn theo phương pháp DMQR (Diverse Multi-Query Rewrite)
để tạo ra các biến thể truy vấn đa dạng nhằm cải thiện kết quả truy xuất.
"""

from typing import List, Dict, Optional
import re

def rewrite_query(query: str, num_rewrites: int = 3) -> List[str]:
    """
    Viết lại truy vấn để tạo ra các biến thể đa dạng theo phương pháp DMQR
    
    Args:
        query: Câu truy vấn gốc
        num_rewrites: Số lượng biến thể truy vấn cần tạo ra
        
    Returns:
        Danh sách các biến thể truy vấn, bao gồm cả truy vấn gốc ở vị trí đầu tiên
    """
    print(f"[DEMO] Generating {num_rewrites} query rewrites for: '{query}'")
    
    # DEMO: Giả lập mô hình DMQR-RAG LLM 
    # Trong thực tế, sẽ gọi một mô hình ngôn ngữ LoRA-tuned cho nhiệm vụ này
    
    # Trường hợp đặc biệt: truy vấn ngắn hoặc không cần viết lại
    if num_rewrites <= 1 or len(query.split()) < 3:
        print(f"[DEMO] Query too short or only one rewrite requested, returning original query")
        return [query]
    
    # Phân loại truy vấn thành các nhóm để áp dụng template phù hợp
    query_type = _classify_query_type(query)
    
    # Tạo các biến thể dựa trên loại truy vấn
    rewrites = [query]  # Luôn giữ lại truy vấn gốc
    
    if query_type == "medical_treatment":
        templates = [
            "What is the state-of-art treatment for [CONDITION]?",
            "Current guidelines for managing [CONDITION]",
            "Best practices in treating [CONDITION]",
            "Recent advances in [CONDITION] therapy",
            "Evidence-based approaches for [CONDITION]",
            "Clinical management of [CONDITION]",
            "Therapeutic options for [CONDITION]",
            "First-line treatments for [CONDITION]"
        ]
        
        # Tách điều kiện từ truy vấn
        condition_match = re.search(r"(treatment|therapy|medication|management) (?:for|of) (.+?)(?:\?|$)", query, re.IGNORECASE)
        if condition_match:
            condition = condition_match.group(2).strip()
        else:
            # Nếu không tìm được mẫu cụ thể, dùng truy vấn sau "for"/"about"
            parts = re.split(r"\b(for|about)\b", query, maxsplit=1, flags=re.IGNORECASE)
            condition = parts[-1].strip() if len(parts) > 1 else query
        
        # Tạo các biến thể
        for i in range(min(num_rewrites-1, len(templates))):
            rewrite = templates[i].replace("[CONDITION]", condition)
            rewrites.append(rewrite)
    
    elif query_type == "medical_comparison":
        templates = [
            "Compare efficacy of [THERAPY1] vs [THERAPY2] for [CONDITION]",
            "Differences between [THERAPY1] and [THERAPY2] in treating [CONDITION]",
            "Clinical trials comparing [THERAPY1] and [THERAPY2]",
            "Benefits and risks of [THERAPY1] versus [THERAPY2]",
            "How does [THERAPY1] compare to [THERAPY2] for [CONDITION] patients?",
            "[THERAPY1] or [THERAPY2]: which is more effective for [CONDITION]?",
        ]
        
        # Tách các thành phần so sánh
        therapies = _extract_comparison_entities(query)
        condition = _extract_condition(query)
        
        if len(therapies) >= 2 and condition:
            for i in range(min(num_rewrites-1, len(templates))):
                rewrite = templates[i].replace("[THERAPY1]", therapies[0])
                rewrite = rewrite.replace("[THERAPY2]", therapies[1])
                rewrite = rewrite.replace("[CONDITION]", condition)
                rewrites.append(rewrite)
    
    elif query_type == "medical_specific":
        templates = [
            "Explain mechanism of action of [TERM]",
            "How does [TERM] work in treating [CONDITION]?",
            "Clinical applications of [TERM]",
            "What is [TERM] used for in medicine?",
            "Latest research on [TERM]",
            "Describe the role of [TERM] in [CONDITION] management",
        ]
        
        # Tách thuật ngữ và điều kiện
        term = _extract_term(query)
        condition = _extract_condition(query)
        
        # Tạo các biến thể
        if term:
            for i in range(min(num_rewrites-1, len(templates))):
                template = templates[i]
                if "[CONDITION]" in template and not condition:
                    # Nếu không có condition nhưng template cần, chọn template khác
                    continue
                rewrite = template.replace("[TERM]", term)
                if condition:
                    rewrite = rewrite.replace("[CONDITION]", condition)
                rewrites.append(rewrite)
    
    else:  # general
        # Cho truy vấn tổng quát, dùng các kỹ thuật đa dạng hóa truy vấn
        rewrites.extend(_generate_general_rewrites(query, num_rewrites-1))
    
    # Đảm bảo số lượng biến thể đúng yêu cầu
    if len(rewrites) < num_rewrites:
        # Nếu chưa đủ số lượng, thêm các biến thể tổng quát
        rewrites.extend(_generate_general_rewrites(query, num_rewrites - len(rewrites)))
    
    # Lọc trùng lặp và giới hạn số lượng
    unique_rewrites = []
    for rewrite in rewrites:
        if rewrite not in unique_rewrites:
            unique_rewrites.append(rewrite)
    
    # Giới hạn số lượng và in kết quả
    result = unique_rewrites[:num_rewrites]
    
    print(f"[DEMO] Generated {len(result)} query rewrites:")
    for i, rewrite in enumerate(result):
        print(f"       - Rewrite {i}: {rewrite}")
    
    return result

def _classify_query_type(query: str) -> str:
    """Phân loại truy vấn để chọn template phù hợp"""
    
    # Truy vấn về điều trị
    if re.search(r"treatment|therapy|medication|cure|manage|treat", query, re.IGNORECASE):
        return "medical_treatment"
    
    # Truy vấn so sánh
    if re.search(r"compare|versus|vs\.|vs|difference|better|worse|efficacy", query, re.IGNORECASE):
        return "medical_comparison"
    
    # Truy vấn về thuật ngữ cụ thể
    if re.search(r"what is|how does|mechanism|explain|define", query, re.IGNORECASE):
        return "medical_specific"
    
    # Mặc định: truy vấn tổng quát
    return "general"

def _extract_comparison_entities(query: str) -> List[str]:
    """Trích xuất các thực thể so sánh từ truy vấn"""
    
    # Tìm mẫu so sánh
    comparison_patterns = [
        r"(?:compare|versus|vs\.?|difference between) (.+?) (?:and|vs\.?|or|versus) (.+?)(?:\s+(?:for|in|to|$))",
        r"(.+?) (?:versus|vs\.?|or|compared to) (.+?)(?:\s+(?:for|in|to|$))",
    ]
    
    for pattern in comparison_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return [match.group(1).strip(), match.group(2).strip()]
    
    # Mẫu cho truy vấn "A or B" 
    or_match = re.search(r"(?:Should I use|Is) (.+?) or (.+?) (?:for|better|to)", query, re.IGNORECASE)
    if or_match:
        return [or_match.group(1).strip(), or_match.group(2).strip()]
    
    # Mẫu backup
    words = query.split()
    if "or" in words:
        idx = words.index("or")
        if idx > 0 and idx < len(words) - 1:
            return [words[idx-1], words[idx+1]]
    
    # Không tìm thấy mẫu
    return ["bisphosphonates", "denosumab"]  # Giá trị mặc định

def _extract_condition(query: str) -> str:
    """Trích xuất điều kiện y khoa từ truy vấn"""
    
    # Mẫu trích xuất điều kiện
    condition_patterns = [
        r"(?:for|in|with|treating) (.+?)(?:\?|$|with|using)",
        r"(?:patient with|case of|suffering from) (.+?)(?:\?|$)",
    ]
    
    for pattern in condition_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Một số điều kiện đặc biệt
    common_conditions = ["osteoporosis", "diabetes", "hypertension", "arthritis", 
                        "asthma", "cancer", "depression", "heart disease"]
    
    for condition in common_conditions:
        if condition in query.lower():
            return condition
    
    # Không tìm thấy
    return "osteoporosis"  # Giá trị mặc định

def _extract_term(query: str) -> str:
    """Trích xuất thuật ngữ y khoa từ truy vấn"""
    
    # Mẫu "what is X" hoặc "how does X work"
    term_patterns = [
        r"(?:what is|what are|how does|explain|definition of|role of) (.+?)(?:\?|$|work|function|in)",
        r"(?:mechanism of action of|pharmacology of) (.+?)(?:\?|$|in)",
    ]
    
    for pattern in term_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Xác định các thuật ngữ y khoa trong câu
    med_terms = [
        "bisphosphonate", "alendronate", "denosumab", "teriparatide",
        "romosozumab", "estrogen", "RANKL", "calcium", "vitamin D",
        "osteoblast", "osteoclast", "bone density", "T-score", "Z-score"
    ]
    
    for term in med_terms:
        if term.lower() in query.lower():
            return term
    
    # Không tìm thấy
    return "bisphosphonate"  # Giá trị mặc định

def _generate_general_rewrites(query: str, num: int) -> List[str]:
    """Tạo các biến thể tổng quát cho truy vấn"""
    
    templates = [
        f"Information about {query}",
        f"Tell me about {query}",
        f"What do medical literature say about {query}?",
        f"Recent research on {query}",
        f"Medical guidelines for {query}",
        f"Scientific evidence regarding {query}",
        f"Clinical perspective on {query}",
        f"Latest findings about {query}",
    ]
    
    # Biến đổi đơn giản
    if "what" in query.lower():
        templates.append(query.replace("what", "explain").strip())
    
    if "how" in query.lower():
        templates.append(query.replace("how", "describe the way").strip())
    
    # Lấy n biến thể đầu tiên
    return templates[:num] 