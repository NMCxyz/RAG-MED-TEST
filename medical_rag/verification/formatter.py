"""
Formatter Module
--------------
Module định dạng câu trả lời y khoa cuối cùng với tham khảo học thuật
và trích dẫn nguồn.
"""

import re
from typing import Dict, List, Any, Optional

def format_verified_answer(verified_answer: str, 
                         evidence_nodes: List[int],
                         format_type: str = "markdown") -> str:
    """
    Định dạng câu trả lời đã xác minh với tham khảo học thuật
    
    Args:
        verified_answer: Câu trả lời đã được xác minh
        evidence_nodes: Danh sách các ID nút làm bằng chứng
        format_type: Loại định dạng ("markdown", "html", "latex")
        
    Returns:
        Câu trả lời đã được định dạng
    """
    print(f"[DEMO] Formatting answer in {format_type} format")
    
    # Loại bỏ các đánh dấu "unverified" nếu có
    cleaned_answer = re.sub(r'\[unverified\]', '', verified_answer)
    
    # Lấy thông tin về các nút bằng chứng
    evidence_info = _get_evidence_info(evidence_nodes)
    
    # Định dạng tham khảo
    if format_type == "markdown":
        formatted_answer = _format_markdown(cleaned_answer, evidence_info)
    elif format_type == "html":
        formatted_answer = _format_html(cleaned_answer, evidence_info)
    elif format_type == "latex":
        formatted_answer = _format_latex(cleaned_answer, evidence_info)
    else:
        # Mặc định là markdown
        formatted_answer = _format_markdown(cleaned_answer, evidence_info)
    
    return formatted_answer

def _get_evidence_info(evidence_nodes: List[int]) -> List[Dict[str, Any]]:
    """
    Lấy thông tin chi tiết về các nút bằng chứng
    
    Args:
        evidence_nodes: Danh sách các ID nút
        
    Returns:
        Danh sách thông tin chi tiết về các nút
    """
    # DEMO: Mô phỏng việc lấy thông tin từ các nút
    # Trong thực tế, sẽ truy vấn đồ thị tri thức hoặc vector store
    
    from medical_rag.graph.seed_select import get_node_info
    
    # Danh sách lưu thông tin
    evidence_info = []
    
    for node_id in evidence_nodes:
        # Lấy thông tin nút
        node_info = get_node_info(node_id)
        
        # Tạo thông tin tham khảo
        reference = {
            "id": node_id,
            "type": node_info.get("type", "Unknown"),
            "name": node_info.get("name", f"Node_{node_id}"),
            "desc": node_info.get("desc", "No description available"),
            "citation": _generate_citation(node_info)
        }
        
        evidence_info.append(reference)
    
    return evidence_info

def _generate_citation(node_info: Dict[str, Any]) -> str:
    """
    Tạo trích dẫn học thuật từ thông tin nút
    
    Args:
        node_info: Thông tin nút
        
    Returns:
        Trích dẫn học thuật
    """
    # DEMO: Mô phỏng việc tạo trích dẫn
    # Trong thực tế, sẽ có thông tin đầy đủ hơn về nguồn
    
    node_type = node_info.get("type", "")
    node_name = node_info.get("name", "")
    
    if node_type == "Drug":
        return f"DrugBank. (2023). {node_name} - Drug Information. Retrieved from https://drugbank.com/drugs/{node_name.lower()}"
    elif node_type == "Disease":
        return f"MedlinePlus. (2023). {node_name}. National Institutes of Health. Retrieved from https://medlineplus.gov/encyclopedia/{node_name.lower()}"
    elif node_type == "Mechanism":
        return f"Goodman & Gilman's. (2023). The Pharmacological Basis of Therapeutics: {node_name}. McGraw-Hill Medical."
    elif node_type == "Concept":
        return f"Medical Encyclopedia. (2023). {node_name}. Retrieved from medical-reference.org"
    else:
        return f"Medical Knowledge Base. (2023). {node_name}."

def _format_markdown(answer: str, evidence_info: List[Dict[str, Any]]) -> str:
    """
    Định dạng câu trả lời theo Markdown
    
    Args:
        answer: Câu trả lời
        evidence_info: Thông tin về các bằng chứng
        
    Returns:
        Câu trả lời định dạng Markdown
    """
    # Định dạng các đoạn văn
    paragraphs = answer.split("\n\n")
    formatted_paragraphs = []
    
    for i, para in enumerate(paragraphs):
        # Giữ nguyên các danh sách
        if para.startswith("- ") or re.match(r"^\d+\. ", para):
            formatted_paragraphs.append(para)
        else:
            # Thêm định dạng cho các đoạn văn thường
            formatted_paragraphs.append(para)
    
    # Tạo phần nội dung
    content = "\n\n".join(formatted_paragraphs)
    
    # Tạo phần tham khảo
    references = []
    for i, info in enumerate(evidence_info):
        ref = f"[{i+1}] {info['citation']}"
        references.append(ref)
    
    # Tạo tiêu đề
    header = "# Medical Response\n\n"
    
    # Tạo phần tham khảo
    if references:
        references_text = "\n## References\n\n" + "\n\n".join(references)
    else:
        references_text = ""
    
    # Kết hợp tất cả
    formatted_answer = f"{header}{content}\n{references_text}"
    
    return formatted_answer

def _format_html(answer: str, evidence_info: List[Dict[str, Any]]) -> str:
    """
    Định dạng câu trả lời theo HTML
    
    Args:
        answer: Câu trả lời
        evidence_info: Thông tin về các bằng chứng
        
    Returns:
        Câu trả lời định dạng HTML
    """
    # Chuyển đổi xuống dòng thành thẻ <p>
    paragraphs = answer.split("\n\n")
    html_paragraphs = []
    
    for para in paragraphs:
        # Xử lý danh sách
        if para.startswith("- "):
            items = para.split("\n- ")
            html_list = "<ul>\n"
            for item in items:
                if item:
                    html_list += f"  <li>{item.lstrip('- ')}</li>\n"
            html_list += "</ul>"
            html_paragraphs.append(html_list)
        elif re.match(r"^\d+\. ", para):
            items = re.split(r"\n\d+\. ", para)
            html_list = "<ol>\n"
            for item in items:
                if item:
                    html_list += f"  <li>{re.sub(r'^\d+\. ', '', item)}</li>\n"
            html_list += "</ol>"
            html_paragraphs.append(html_list)
        else:
            html_paragraphs.append(f"<p>{para}</p>")
    
    # Tạo phần nội dung
    content = "\n".join(html_paragraphs)
    
    # Tạo phần tham khảo
    references = []
    for i, info in enumerate(evidence_info):
        ref = f'<li id="ref-{i+1}">{info["citation"]}</li>'
        references.append(ref)
    
    # Tạo HTML hoàn chỉnh
    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Medical Response</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0 auto; max-width: 800px; padding: 20px; }}
    h1, h2 {{ color: #2c3e50; }}
    .references {{ margin-top: 30px; border-top: 1px solid #eee; padding-top: 20px; }}
    .references ol {{ padding-left: 20px; }}
  </style>
</head>
<body>
  <h1>Medical Response</h1>
  {content}
  
  <div class="references">
    <h2>References</h2>
    <ol>
      {"".join(references)}
    </ol>
  </div>
</body>
</html>"""
    
    return html

def _format_latex(answer: str, evidence_info: List[Dict[str, Any]]) -> str:
    """
    Định dạng câu trả lời theo LaTeX
    
    Args:
        answer: Câu trả lời
        evidence_info: Thông tin về các bằng chứng
        
    Returns:
        Câu trả lời định dạng LaTeX
    """
    # Xử lý các ký tự đặc biệt trong LaTeX
    latex_escape_chars = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}"
    }
    
    # Escape các ký tự đặc biệt
    for char, escape in latex_escape_chars.items():
        answer = answer.replace(char, escape)
    
    # Xử lý các đoạn văn
    paragraphs = answer.split("\n\n")
    latex_paragraphs = []
    
    for para in paragraphs:
        # Xử lý danh sách
        if para.startswith("- "):
            items = para.split("\n- ")
            latex_list = "\\begin{itemize}\n"
            for item in items:
                if item:
                    latex_list += f"  \\item {item.lstrip('- ')}\n"
            latex_list += "\\end{itemize}"
            latex_paragraphs.append(latex_list)
        elif re.match(r"^\d+\. ", para):
            items = re.split(r"\n\d+\. ", para)
            latex_list = "\\begin{enumerate}\n"
            for item in items:
                if item:
                    latex_list += f"  \\item {re.sub(r'^\d+\. ', '', item)}\n"
            latex_list += "\\end{enumerate}"
            latex_paragraphs.append(latex_list)
        else:
            latex_paragraphs.append(para)
    
    # Tạo phần nội dung
    content = "\n\n".join(latex_paragraphs)
    
    # Tạo phần tham khảo
    references = []
    for i, info in enumerate(evidence_info):
        # Escape các ký tự đặc biệt trong citation
        citation = info["citation"]
        for char, escape in latex_escape_chars.items():
            citation = citation.replace(char, escape)
            
        ref = f"\\bibitem{{{i+1}}} {citation}"
        references.append(ref)
    
    # Tạo LaTeX hoàn chỉnh
    latex = f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{natbib}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}

\\title{{Medical Response}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

{content}

\\begin{{thebibliography}}{{99}}
{"".join(references)}
\\end{{thebibliography}}

\\end{{document}}"""
    
    return latex

def generate_citation(source_type: str, 
                    source_info: Dict[str, Any], 
                    citation_style: str = "APA") -> str:
    """
    Tạo trích dẫn học thuật theo định dạng chuẩn
    
    Args:
        source_type: Loại nguồn ("journal", "book", "website", etc.)
        source_info: Thông tin về nguồn
        citation_style: Kiểu trích dẫn ("APA", "MLA", "Vancouver", etc.)
        
    Returns:
        Trích dẫn định dạng chuẩn
    """
    # DEMO: Mô phỏng việc tạo trích dẫn theo định dạng APA
    # Trong thực tế, sẽ sử dụng thư viện như citeproc-py
    
    if citation_style == "APA":
        if source_type == "journal":
            authors = source_info.get("authors", "")
            year = source_info.get("year", "")
            title = source_info.get("title", "")
            journal = source_info.get("journal", "")
            volume = source_info.get("volume", "")
            issue = source_info.get("issue", "")
            pages = source_info.get("pages", "")
            
            return f"{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}."
            
        elif source_type == "book":
            authors = source_info.get("authors", "")
            year = source_info.get("year", "")
            title = source_info.get("title", "")
            publisher = source_info.get("publisher", "")
            
            return f"{authors} ({year}). {title}. {publisher}."
            
        elif source_type == "website":
            authors = source_info.get("authors", "")
            year = source_info.get("year", "")
            title = source_info.get("title", "")
            website = source_info.get("website", "")
            url = source_info.get("url", "")
            
            return f"{authors} ({year}). {title}. {website}. Retrieved from {url}"
    
    elif citation_style == "MLA":
        # Implement MLA citation style
        pass
    
    elif citation_style == "Vancouver":
        # Implement Vancouver citation style
        pass
    
    # Mặc định trả về dạng text đơn giản
    return f"{source_info.get('authors', '')}. {source_info.get('title', '')}. {source_info.get('year', '')}." 