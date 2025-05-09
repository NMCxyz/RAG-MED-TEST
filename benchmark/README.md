# Benchmarking RAG y khoa

Thư mục này chứa các công cụ để đánh giá hiệu suất hệ thống RAG y khoa trên các bộ dữ liệu tiêu chuẩn.

## Giới thiệu

Hệ thống đánh giá này sử dụng các bộ dữ liệu câu hỏi-trả lời phổ biến để đo lường chất lượng và hiệu suất của hệ thống RAG y khoa. Metrics được sử dụng bao gồm:

- **Exact Match**: Tỷ lệ câu trả lời khớp chính xác với ground truth
- **F1 Score**: Điểm F1 dựa trên overlapping tokens
- **ROUGE**: Đánh giá sự tương đồng giữa câu trả lời và ground truth
- **BLEU**: Điểm BLEU đánh giá chất lượng dịch thuật
- **BERTScore**: Đánh giá sự tương đồng ngữ nghĩa giữa câu trả lời và ground truth
- **Latency**: Thời gian xử lý trung bình

## Các bộ dữ liệu hỗ trợ

- **HotpotQA**: Bộ dữ liệu câu hỏi đòi hỏi suy luận đa bước, phù hợp cho việc đánh giá RAG kết hợp với suy luận

## Cài đặt

Đảm bảo bạn đã cài đặt tất cả các thư viện cần thiết trong `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ngoài ra, bạn cần cài đặt thêm các gói cần thiết cho benchmark:

```bash
pip install datasets rouge_score nltk pandas matplotlib seaborn
```

## Sử dụng

### Chạy benchmark trên HotpotQA

```bash
python benchmark.py --datasets hotpotqa --num_samples 100 --filter_domain medicine
```

### Tham số chính

- `--datasets`: Danh sách các bộ dữ liệu (mặc định: "hotpotqa")
- `--num_samples`: Số lượng mẫu cần đánh giá (mặc định: 100)
- `--output_dir`: Thư mục lưu kết quả (mặc định: "benchmark_results")
- `--timeout`: Timeout cho mỗi câu hỏi (mặc định: 60 giây)
- `--device`: Thiết bị chạy model ("cuda" hoặc "cpu")

### Tham số cho HotpotQA

- `--split`: Phân vùng dữ liệu ("train", "validation") (mặc định: "validation")
- `--filter_domain`: Lọc theo domain, phân tách bằng dấu phẩy (mặc định: "medicine")

## Ví dụ

### Chạy benchmark trên 50 mẫu từ HotpotQA với domain y tế và khoa học

```bash
python benchmark.py --datasets hotpotqa --num_samples 50 --filter_domain medicine,science
```

### Chạy benchmark với GPU

```bash
python benchmark.py --datasets hotpotqa --device cuda
```

## Kết quả

Kết quả benchmark sẽ được lưu trong thư mục `benchmark_results` (hoặc thư mục được chỉ định). Kết quả bao gồm:

1. File JSON chứa metrics chi tiết và kết quả từng mẫu
2. File CSV chứa bảng tổng hợp metrics 
3. Biểu đồ trực quan trong thư mục `visualizations` (nếu chạy nhiều bộ dữ liệu)

## Lưu ý

- Đối với mỗi câu hỏi, hệ thống sẽ sử dụng context từ bộ dữ liệu thay vì thực hiện retrieval
- Thời gian chạy có thể khá lâu tùy thuộc vào số lượng mẫu và cấu hình hệ thống
- Đảm bảo đã thiết lập các biến môi trường cần thiết cho LLM APIs (OPENAI_API_KEY, v.v.) 