# RAG-MED: Hệ thống hỏi đáp y khoa dựa trên RAG

RAG-MED là hệ thống hỏi đáp y khoa tiên tiến sử dụng kiến trúc Retrieval-Augmented Generation (RAG) kết hợp với đồ thị tri thức y khoa, mô hình ngôn ngữ lớn chuyên biệt, và các kỹ thuật kiểm chứng thông tin.

## Kiến trúc hệ thống

Hệ thống RAG-MED bao gồm các thành phần chính sau:

1. **Chunker**: Tách văn bản y khoa thành các đơn vị có ngữ nghĩa hoàn chỉnh
2. **KG Extract**: Trích xuất triplet để xây dựng đồ thị tri thức y khoa
3. **Query Processing**: Phân loại độ phức tạp và viết lại truy vấn
4. **Retrieval**: Kết hợp truy xuất dense và sparse với fusion
5. **Graph Processing**: Xử lý đồ thị tri thức với MixPR và GFM
6. **Reasoning**: Suy luận Chain-of-Thought trên đồ thị
7. **Verification**: Kiểm chứng câu trả lời với kỹ thuật REFIND

## Luồng xử lý dữ liệu

```
Query → Complexity Classification → Parameter Selection → Query Rewriting 
    → Hybrid Retrieval (Dense + Sparse) → Fusion → Reranking
    → Graph Seed Selection → MixPR Router → GFM Encoding
    → Graph-CoT Reasoning → LLM Answer Generation → REFIND Verification → Formatting
```

## Các mô hình được sử dụng

- **BGE-M3**: Mô hình embedding đa ngữ (768 chiều)
- **SPLADE**: Mô hình sparse lexical expansion
- **MixPR**: Thuật toán PageRank cá nhân hóa
- **GFM Encoder**: Graph Transformer cho biểu diễn đồ thị
- **Qwen-Med**: Mô hình ngôn ngữ lớn chuyên biệt y khoa
- **REFIND**: Phương pháp phát hiện thông tin không chính xác

## Cài đặt

```bash
# Clone repository
git clone https://github.com/username/RAG-MED.git
cd RAG-MED

# Tạo môi trường Python
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
medical_rag/
├── chunker/              # Module xử lý tài liệu PDF/HTML
├── kg_extract/           # Module trích xuất triplet
├── query_processing/     # Phân loại và viết lại truy vấn
├── retrieval/            # Các phương pháp truy xuất
├── graph/                # Xử lý đồ thị tri thức
├── reasoning/            # Chain-of-Thought trên đồ thị
├── verification/         # Kiểm chứng và định dạng câu trả lời
└── main.py               # Luồng xử lý chính
```

## Sử dụng

```python
from medical_rag.main import run_full_pipeline

# Chạy pipeline đầy đủ
results = run_full_pipeline(
    query="What are the new treatments for osteoporosis?",
    pdf_path="path/to/medical_document.pdf"  # Optional
)

# Xem câu trả lời cuối cùng
print(results["formatted_answer"])
```

## Tài liệu tham khảo

1. "DMQR: A Dual-encoder Multi-query Retrieval Framework for Dense Retrieval", arXiv
2. "SPLADE: Sparse Lexical and Expansion Model for IR", arXiv
3. "MixPR: A Mixed Personalized Ranking System for Knowledge Graph Applications", arXiv
4. "GFM-RAG: Retrieval Augmented Graph and Language Models for Knowledge-intensive Tasks", arXiv
5. "Retrieval-Enhanced Text Generation with REFIND Hallucination Detection", arXiv

## Đóng góp

Chúng tôi khuyến khích đóng góp cho dự án! Vui lòng làm theo các bước:

1. Fork repository
2. Tạo branch mới
3. Thực hiện thay đổi
4. Tạo Pull Request

## Giấy phép

MIT 