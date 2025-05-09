# HƯỚNG DẪN NHANH HỆ THỐNG RAG Y KHOA

## Cài đặt

```bash
# Cài đặt thư viện cần thiết
pip install -r requirements.txt

# Thư viện bổ sung cho benchmark
pip install datasets rouge_score nltk pandas matplotlib seaborn

# Thiết lập các biến môi trường (thay thế bằng API key của bạn)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
```

## Cấu trúc hệ thống

```
medical_rag/
├── chunker/               # Phân chia văn bản thành đoạn có ngữ nghĩa
├── kg_extract/            # Trích xuất triplet từ văn bản
├── query_processing/      # Xử lý truy vấn
├── retrieval/             # Truy xuất văn bản liên quan
├── graph/                 # Xử lý đồ thị tri thức y khoa
├── reasoning/             # Suy luận trên đồ thị và văn bản
├── verification/          # Kiểm tra ảo giác và định dạng
└── main.py                # Luồng xử lý toàn bộ
```

## Chạy demo

```bash
# Chạy demo với truy vấn mẫu
python medical_rag/main.py

# Chạy với truy vấn tùy chỉnh 
python medical_rag/main.py --query "So sánh hiệu quả của denosumab và alendronate trong điều trị loãng xương"
```

## Chạy benchmark

```bash
# Chạy benchmark với HotpotQA (50 mẫu, domain y khoa)
cd benchmark
python benchmark.py --datasets hotpotqa --num_samples 50 --filter_domain medicine

# Chạy benchmark với GPU
python benchmark.py --datasets hotpotqa --device cuda --timeout 120

# Chạy benchmark với tùy chọn nâng cao
python benchmark.py --datasets hotpotqa --num_samples 100 --filter_domain medicine,science --split validation --output_dir custom_results
```

## Tham số quan trọng

### Benchmark

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--datasets` | Danh sách dataset | "hotpotqa" |
| `--num_samples` | Số lượng mẫu | 100 |
| `--output_dir` | Thư mục lưu kết quả | "benchmark_results" |
| `--timeout` | Timeout mỗi query (giây) | 60 |
| `--device` | Thiết bị (cuda/cpu) | auto |
| `--filter_domain` | Lọc domain | "medicine" |
| `--split` | Phân vùng dữ liệu | "validation" |

## Models & APIs

### LLM y khoa

- **Qwen-Med**: Mô hình y khoa mặc định
- **GPT-4**: Dự phòng khi không có Qwen-Med
- **Claude-3**: Hỗ trợ qua Anthropic API

### Embeddings & Retrievers

- **BGE-M3**: Dense retriever
- **SPLADE**: Sparse retriever
- **RRF Fusion**: Kết hợp kết quả truy xuất

### Xử lý đồ thị

- **Seed Selector**: Chọn nút gốc từ vector truy vấn
- **MixPR Router**: Định tuyến trên đồ thị tri thức
- **GFM Encoder**: Mã hóa đồ thị thành vector

## Neo4j Database

```bash
# Khởi động Neo4j 
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Truy cập Neo4j Browser
http://localhost:7474
```

## Xem kết quả Benchmark

```bash
# Xem kết quả tóm tắt
cat benchmark_results/hotpotqa_metrics_*.csv

# Xem trực quan
cd benchmark_results/visualizations
```

## Xử lý lỗi thường gặp

1. **ImportError: No module named 'xxx'**
   ```bash
   pip install xxx
   ```

2. **OpenAI API error: "Rate limit exceeded"**
   - Đợi 1 phút trước khi thử lại
   - Hoặc sử dụng LLM khác: `export MEDICAL_LLM_MODEL="claude-3-opus-20240229"`

3. **Neo4j connection error**
   - Đảm bảo Neo4j đang chạy: `docker ps | grep neo4j`
   - Kiểm tra URI và thông tin đăng nhập

4. **CUDA out of memory**
   - Giảm batch_size trong file cấu hình
   - Hoặc chạy trên CPU: `--device cpu`

5. **Lỗi khi tải HotpotQA dataset**
   - Đảm bảo có kết nối internet
   - Thử tải offline: `datasets download hotpot_qa` 