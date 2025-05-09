#!/usr/bin/env python
"""
Setup Benchmark
--------------
Script tự động thiết lập môi trường benchmark cho hệ thống RAG y khoa
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def setup_benchmark_environment(force_overwrite=False):
    """
    Thiết lập môi trường benchmark
    
    Args:
        force_overwrite: Ghi đè các file đã tồn tại
    """
    print("=== Thiết lập môi trường benchmark cho RAG y khoa ===")
    
    # Tạo cấu trúc thư mục
    benchmark_dir = Path("benchmark")
    benchmark_dir.mkdir(exist_ok=True)
    
    # Tạo các file benchmark
    files_to_create = {
        "benchmark/hotpotqa_evaluator.py": """
\"\"\"
HotpotQA Evaluator
----------------
Module đánh giá hệ thống RAG y khoa sử dụng tập dữ liệu HotpotQA
\"\"\"

import os
import json
import time
import logging
import argparse
import numpy as np
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import pandas as pd
import requests
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Thêm thư mục gốc vào path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module chính của RAG
from medical_rag.query_processing.complexity_classifier import classify_complexity
from medical_rag.query_processing.param_selector import select_params
from medical_rag.query_processing.query_rewriter import rewrite_query
from medical_rag.retrieval.dense_retriever import retrieve_with_bge_m3
from medical_rag.retrieval.sparse_retriever import retrieve_with_splade
from medical_rag.retrieval.fusion import reciprocal_rank_fusion, combine_evidence
from medical_rag.graph.seed_select import select_seed_nodes
from medical_rag.graph.mixpr_router import get_subgraph_with_mixpr
from medical_rag.graph.gfm_encoder import encode_graph_with_gfm
from medical_rag.reasoning.graph_cot import reasoning_with_graph_cot
from medical_rag.reasoning.qwen_med import generate_answer_with_llm
from medical_rag.verification.refind import verify_answer_with_refind

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HotpotQAEvaluator:
    \"\"\"Class đánh giá hệ thống RAG y khoa sử dụng HotpotQA\"\"\"
    
    def __init__(self, 
                model_path: Optional[str] = None,
                device: Optional[str] = None,
                output_dir: str = "benchmark_results"):
        \"\"\"
        Khởi tạo evaluator
        
        Args:
            model_path: Đường dẫn tới model BERTScore nếu dùng local
            device: Thiết bị để chạy model ("cuda", "cpu")
            output_dir: Thư mục lưu kết quả đánh giá
        \"\"\"
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Khởi tạo ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Khởi tạo model cho BERTScore (local hoặc qua API)
        self.bert_model = None
        self.bert_tokenizer = None
        if model_path:
            try:
                logger.info(f"Loading BERTScore model from {model_path}")
                self.bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.bert_model.to(self.device)
                self.bert_model.eval()
            except Exception as e:
                logger.error(f"Error loading BERTScore model: {e}")
        
        logger.info(f"HotpotQA Evaluator initialized on {self.device}")
    
    def load_hotpotqa_data(self, 
                          split: str = "validation", 
                          subset: Optional[int] = None,
                          filter_domain: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        \"\"\"
        Tải dữ liệu HotpotQA
        
        Args:
            split: Phân vùng dữ liệu ("train", "validation")
            subset: Số lượng mẫu để đánh giá (None: tất cả)
            filter_domain: Lọc theo các domain cụ thể (ví dụ: ["medicine", "science"])
            
        Returns:
            Danh sách các mẫu dữ liệu
        \"\"\"
        logger.info(f"Loading HotpotQA {split} set")
        
        try:
            # Tải dataset từ Hugging Face
            dataset = load_dataset("hotpot_qa", "distractor", split=split)
            
            # Chuyển sang định dạng list để dễ xử lý
            samples = []
            for item in dataset:
                sample = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "context": [c for title_ctxs in item["context"]["sentences"] for c in title_ctxs],
                    "type": item["type"],
                    "level": item["level"]
                }
                samples.append(sample)
            
            # Lọc theo domain nếu cần
            if filter_domain:
                # Chú ý: HotpotQA không có metadata về domain,
                # nên cần phải dùng keyword matching
                filtered_samples = []
                domain_keywords = {
                    "medicine": ["disease", "treatment", "patient", "medical", "health", "doctor", "hospital", "drug"],
                    "science": ["scientist", "research", "discovery", "physics", "chemistry", "biology"],
                    # Thêm các domain khác nếu cần
                }
                
                keywords = []
                for domain in filter_domain:
                    if domain in domain_keywords:
                        keywords.extend(domain_keywords[domain])
                
                for sample in samples:
                    question = sample["question"].lower()
                    if any(keyword in question for keyword in keywords):
                        filtered_samples.append(sample)
                
                samples = filtered_samples
                logger.info(f"Filtered to {len(samples)} samples in domains: {filter_domain}")
            
            # Lấy subset nếu cần
            if subset and subset < len(samples):
                samples = samples[:subset]
                logger.info(f"Using subset of {subset} samples")
            
            logger.info(f"Loaded {len(samples)} samples from HotpotQA {split} set")
            return samples
            
        except Exception as e:
            logger.error(f"Error loading HotpotQA data: {e}")
            # Tạo mẫu dữ liệu giả lập nếu có lỗi
            logger.info("Using dummy data instead")
            return self._get_dummy_data(subset or 10)
    
    def _get_dummy_data(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        \"\"\"Tạo dữ liệu giả lập để kiểm thử\"\"\"
        dummy_samples = []
        
        # Một số câu hỏi liên quan đến y khoa
        medical_questions = [
            {
                "question": "What medication is used to treat osteoporosis and also works by inhibiting osteoclast-mediated bone resorption?",
                "answer": "Alendronate",
                "context": [
                    "Bisphosphonates are a class of drugs used to treat osteoporosis and similar diseases.",
                    "Alendronate is a bisphosphonate that works by inhibiting osteoclast-mediated bone resorption.",
                    "Osteoporosis is characterized by reduced bone mineral density and increased risk of fracture."
                ],
                "type": "comparison",
                "level": "hard"
            },
            {
                "question": "Which drug developed by Amgen is a monoclonal antibody used to treat osteoporosis?",
                "answer": "Denosumab",
                "context": [
                    "Denosumab is a human monoclonal antibody developed by Amgen.",
                    "It is marketed under the brand names Prolia and Xgeva for treating osteoporosis.",
                    "Denosumab inhibits RANKL, preventing osteoclast formation and decreasing bone resorption."
                ],
                "type": "bridge",
                "level": "medium"
            },
            {
                "question": "What is the mechanism of action of Romosozumab that makes it different from bisphosphonates in treating osteoporosis?",
                "answer": "It inhibits sclerostin, increasing bone formation while decreasing bone resorption",
                "context": [
                    "Romosozumab is an antibody that targets sclerostin, a protein that inhibits bone formation.",
                    "By inhibiting sclerostin, Romosozumab increases bone formation while decreasing bone resorption.",
                    "Bisphosphonates only decrease bone resorption without stimulating bone formation."
                ],
                "type": "comparison",
                "level": "hard"
            }
        ]
        
        # Nhân bản các câu hỏi để đạt được số lượng mẫu yêu cầu
        for i in range(num_samples):
            idx = i % len(medical_questions)
            sample = medical_questions[idx].copy()
            # Thêm chút biến thể để tránh trùng lặp hoàn toàn
            sample["id"] = f"dummy-{i+1}"
            dummy_samples.append(sample)
        
        return dummy_samples
    
    def run_benchmark(self, 
                     samples: List[Dict[str, Any]], 
                     batch_size: int = 1, 
                     timeout: int = 60,
                     save_results: bool = True) -> Dict[str, Any]:
        \"\"\"
        Chạy benchmark trên các mẫu dữ liệu
        
        Args:
            samples: Danh sách các mẫu dữ liệu
            batch_size: Kích thước batch (hiện chỉ hỗ trợ batch_size=1)
            timeout: Thời gian tối đa cho mỗi câu hỏi (giây)
            save_results: Lưu kết quả vào file hay không
            
        Returns:
            Kết quả đánh giá
        \"\"\"
        logger.info(f"Running benchmark on {len(samples)} samples with timeout={timeout}s")
        
        all_results = []
        metrics = {
            "exact_match": 0,
            "f1": 0,
            "precision": 0,
            "recall": 0,
            "rouge1": 0,
            "rouge2": 0,
            "rougeL": 0,
            "bleu": 0,
            "bert_score": 0,
            "success_rate": 0,
            "avg_latency": 0
        }
        
        total_samples = len(samples)
        successful_samples = 0
        total_latency = 0
        
        # Chạy benchmark trên từng mẫu
        for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
            try:
                # Đặt timeout cho mỗi mẫu
                start_time = time.time()
                
                # Chạy pipeline RAG cho câu hỏi
                result = self._run_rag_pipeline(sample, timeout)
                
                # Tính thời gian xử lý
                latency = time.time() - start_time
                total_latency += latency
                
                # Tính các metrics
                if result["success"]:
                    successful_samples += 1
                    sample_metrics = self._calculate_metrics(
                        prediction=result["answer"],
                        reference=sample["answer"],
                        sample=sample
                    )
                    
                    # Cập nhật metrics tổng
                    for key in sample_metrics:
                        if key in metrics:
                            metrics[key] += sample_metrics[key]
                
                # Lưu kết quả mẫu
                sample_result = {
                    "id": sample.get("id", f"sample-{i}"),
                    "question": sample["question"],
                    "ground_truth": sample["answer"],
                    "predicted_answer": result.get("answer", ""),
                    "success": result["success"],
                    "error": result.get("error", ""),
                    "latency": latency,
                    "metrics": sample_metrics if result["success"] else {}
                }
                all_results.append(sample_result)
                
                # Log kết quả mỗi 10 mẫu
                if (i+1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{total_samples} samples")
                    
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                all_results.append({
                    "id": sample.get("id", f"sample-{i}"),
                    "question": sample["question"],
                    "ground_truth": sample["answer"],
                    "predicted_answer": "",
                    "success": False,
                    "error": str(e),
                    "latency": time.time() - start_time,
                    "metrics": {}
                })
        
        # Tính trung bình các metrics
        if successful_samples > 0:
            for key in metrics:
                if key != "success_rate" and key != "avg_latency":
                    metrics[key] /= successful_samples
        
        # Tính success rate và latency
        metrics["success_rate"] = successful_samples / total_samples if total_samples > 0 else 0
        metrics["avg_latency"] = total_latency / total_samples if total_samples > 0 else 0
        
        # Log kết quả tổng thể
        logger.info(f"Benchmark completed. Success rate: {metrics['success_rate']:.2f}, Avg latency: {metrics['avg_latency']:.2f}s")
        logger.info(f"Metrics: {metrics}")
        
        # Lưu kết quả nếu cần
        if save_results:
            results_file = os.path.join(self.output_dir, f"hotpotqa_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_file, 'w') as f:
                json.dump({
                    "metrics": metrics,
                    "samples": all_results
                }, f, indent=2)
            logger.info(f"Results saved to {results_file}")
            
            # Lưu metrics dạng CSV để dễ phân tích
            metrics_df = pd.DataFrame([metrics])
            metrics_csv = os.path.join(self.output_dir, f"hotpotqa_metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            metrics_df.to_csv(metrics_csv, index=False)
            logger.info(f"Metrics saved to {metrics_csv}")
        
        return {
            "metrics": metrics,
            "samples": all_results
        }
    
    def _run_rag_pipeline(self, sample: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        \"\"\"
        Chạy pipeline RAG y khoa trên một mẫu
        
        Args:
            sample: Mẫu dữ liệu với câu hỏi và bối cảnh
            timeout: Thời gian tối đa (giây)
            
        Returns:
            Kết quả của pipeline
        \"\"\"
        try:
            # Trích xuất câu hỏi và bối cảnh
            query = sample["question"]
            contexts = sample["context"]
            
            # Bước 1: Query Processing
            complexity_tag = classify_complexity(query)
            params = select_params(complexity_tag)
            rewritten_queries = rewrite_query(query, params["num_rewrites"])
            
            # Bước 2: Retrieval
            # Thay vì dùng retriever thực, sử dụng contexts từ dataset
            dense_vector, _ = retrieve_with_bge_m3(rewritten_queries[0], top_k=1)
            
            # Tạo kết quả giả lập cho retrieval từ context có sẵn
            retrieved_chunks = []
            for i, ctx in enumerate(contexts):
                retrieved_chunks.append({
                    "id": f"ctx-{i}",
                    "text": ctx,
                    "score": 0.9 - (i * 0.05) if i < 10 else 0.4,
                    "rank": i
                })
            
            # Kết hợp contexts thành một đoạn evidence
            evidence_texts = [chunk["text"] for chunk in retrieved_chunks[:params["topK"]]]
            
            # Bước 3: Graph Processing
            seed_nodes = select_seed_nodes(dense_vector)
            subgraph_nodes, subgraph_edges = get_subgraph_with_mixpr(seed_nodes, params["k"], params["topK"])
            
            # Bước 4: Reasoning
            cot_steps, evidence_nodes = reasoning_with_graph_cot(
                query=query,
                subgraph_nodes=subgraph_nodes,
                subgraph_edges=subgraph_edges,
                max_steps=params["max_reasoning_steps"]
            )
            
            # Bước 5: Answer Generation
            draft_answer = generate_answer_with_llm(
                query=query,
                evidence_texts=evidence_texts,
                cot_steps=cot_steps
            )
            
            # Bước 6: Verification
            verified_answer, _ = verify_answer_with_refind(draft_answer, evidence_texts)
            
            return {
                "success": True,
                "answer": verified_answer
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _calculate_metrics(self, 
                         prediction: str, 
                         reference: str,
                         sample: Dict[str, Any]) -> Dict[str, float]:
        \"\"\"
        Tính các metrics đánh giá
        
        Args:
            prediction: Câu trả lời dự đoán
            reference: Câu trả lời chuẩn
            sample: Mẫu dữ liệu gốc
            
        Returns:
            Các metrics đánh giá
        \"\"\"
        # Chuẩn hóa câu trả lời
        pred = prediction.strip().lower()
        ref = reference.strip().lower()
        
        # Exact Match
        exact_match = 1.0 if pred == ref else 0.0
        
        # Token-level F1
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        common_tokens = pred_tokens.intersection(ref_tokens)
        
        if not pred_tokens:
            precision = 0.0
        else:
            precision = len(common_tokens) / len(pred_tokens)
            
        if not ref_tokens:
            recall = 0.0
        else:
            recall = len(common_tokens) / len(ref_tokens)
            
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # ROUGE
        rouge_scores = self.rouge_scorer.score(reference, prediction)
        
        # BLEU
        try:
            # Tokenize reference và prediction
            ref_tokens = [reference.split()]
            pred_tokens = prediction.split()
            
            # Tính BLEU với smoothing
            smoothing = SmoothingFunction().method1
            bleu = corpus_bleu([ref_tokens], [pred_tokens], smoothing_function=smoothing)
        except:
            bleu = 0.0
        
        # BERTScore (nếu model đã được khởi tạo)
        bert_score = 0.0
        if self.bert_model and self.bert_tokenizer:
            try:
                # Tokenize 
                inputs = self.bert_tokenizer(
                    [reference], 
                    [prediction], 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    logits = outputs.logits
                    
                # Lấy điểm tương đồng (xác suất của lớp "entailment")
                bert_score = torch.softmax(logits, dim=1)[:, 1].item()
            except Exception as e:
                logger.error(f"Error calculating BERTScore: {e}")
        
        # Kết hợp tất cả metrics
        metrics = {
            "exact_match": exact_match,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            "bleu": bleu,
            "bert_score": bert_score
        }
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system on HotpotQA")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to use")
    parser.add_argument("--subset", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Directory to save results")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per query in seconds")
    parser.add_argument("--device", type=str, default=None, help="Device to run models (cuda or cpu)")
    parser.add_argument("--filter_domain", type=str, default="medicine", help="Filter by domain (comma-separated)")
    
    args = parser.parse_args()
    
    # Parse domain filter
    filter_domain = args.filter_domain.split(",") if args.filter_domain else None
    
    # Khởi tạo evaluator
    evaluator = HotpotQAEvaluator(
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Tải dữ liệu
    samples = evaluator.load_hotpotqa_data(
        split=args.split,
        subset=args.subset,
        filter_domain=filter_domain
    )
    
    # Chạy benchmark
    results = evaluator.run_benchmark(
        samples=samples,
        timeout=args.timeout,
        save_results=True
    )
    
    # In kết quả tổng quan
    print("\\n=== BENCHMARK RESULTS ===")
    print(f"Success rate: {results['metrics']['success_rate']:.2f}")
    print(f"Average latency: {results['metrics']['avg_latency']:.2f}s")
    print(f"F1 score: {results['metrics']['f1']:.2f}")
    print(f"ROUGE-L: {results['metrics']['rougeL']:.2f}")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
""",
        "benchmark/benchmark.py": """
\"\"\"
Benchmark Runner
----------------
Script để chạy benchmark hệ thống RAG y khoa trên nhiều bộ dữ liệu
và tổng hợp kết quả.
\"\"\"

import os
import json
import time
import argparse
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Thêm thư mục gốc vào path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các evaluator
from hotpotqa_evaluator import HotpotQAEvaluator

def run_all_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    \"\"\"
    Chạy benchmark trên tất cả các bộ dữ liệu được chỉ định
    
    Args:
        args: Tham số dòng lệnh
        
    Returns:
        Kết quả benchmark tổng hợp
    \"\"\"
    results = {}
    
    # Tạo thư mục kết quả
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Benchmark trên HotpotQA
    if "hotpotqa" in args.datasets:
        print("\\n=== Running HotpotQA benchmark ===")
        hotpotqa_results = run_hotpotqa_benchmark(args)
        results["hotpotqa"] = hotpotqa_results
    
    # Có thể thêm các bộ dữ liệu khác ở đây
    
    # Tổng hợp và hiển thị kết quả
    summarize_results(results, args.output_dir)
    
    return results

def run_hotpotqa_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    \"\"\"
    Chạy benchmark trên tập dữ liệu HotpotQA
    
    Args:
        args: Tham số dòng lệnh
        
    Returns:
        Kết quả benchmark
    \"\"\"
    # Khởi tạo evaluator
    hotpotqa_evaluator = HotpotQAEvaluator(
        device=args.device,
        output_dir=os.path.join(args.output_dir, "hotpotqa")
    )
    
    # Tải dữ liệu
    filter_domain = args.filter_domain.split(",") if args.filter_domain else None
    samples = hotpotqa_evaluator.load_hotpotqa_data(
        split=args.split,
        subset=args.num_samples,
        filter_domain=filter_domain
    )
    
    # Chạy benchmark
    results = hotpotqa_evaluator.run_benchmark(
        samples=samples,
        timeout=args.timeout,
        save_results=True
    )
    
    return results

def summarize_results(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    \"\"\"
    Tổng hợp và hiển thị kết quả benchmark
    
    Args:
        results: Kết quả benchmark từ các dataset
        output_dir: Thư mục lưu kết quả
    \"\"\"
    print("\\n=== BENCHMARK SUMMARY ===")
    
    # Thu thập metrics từ tất cả các dataset
    all_metrics = []
    for dataset_name, dataset_results in results.items():
        metrics = dataset_results["metrics"]
        metrics["dataset"] = dataset_name
        all_metrics.append(metrics)
    
    # Tạo DataFrame từ metrics
    metrics_df = pd.DataFrame(all_metrics)
    
    # Hiển thị bảng kết quả tóm tắt
    summary_columns = ["dataset", "success_rate", "f1", "rougeL", "bleu", "avg_latency"]
    summary_df = metrics_df[summary_columns].sort_values("f1", ascending=False)
    
    print("\\nResults Summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Lưu kết quả tổng hợp
    summary_path = os.path.join(output_dir, f"benchmark_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    metrics_df.to_csv(summary_path, index=False)
    print(f"\\nSummary saved to {summary_path}")
    
    # Visualize results if there are multiple datasets
    if len(results) > 1:
        create_visualization(metrics_df, output_dir)

def create_visualization(metrics_df: pd.DataFrame, output_dir: str) -> None:
    \"\"\"
    Tạo các biểu đồ trực quan hóa kết quả benchmark
    
    Args:
        metrics_df: DataFrame chứa metrics
        output_dir: Thư mục lưu biểu đồ
    \"\"\"
    try:
        # Tạo thư mục cho biểu đồ
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set style
        sns.set(style="whitegrid")
        
        # 1. F1 Score Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="dataset", y="f1", data=metrics_df)
        ax.set_title("F1 Score by Dataset", fontsize=15)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_xlabel("Dataset", fontsize=12)
        
        # Add value labels on bars
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.4f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "f1_comparison.png"))
        
        # 2. Multiple Metrics Comparison
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ["f1", "rougeL", "bleu", "exact_match"]
        
        # Melt the dataframe for easier plotting
        plot_df = pd.melt(metrics_df, 
                          id_vars=["dataset"], 
                          value_vars=metrics_to_plot,
                          var_name="Metric", 
                          value_name="Score")
        
        ax = sns.barplot(x="dataset", y="Score", hue="Metric", data=plot_df)
        ax.set_title("Performance Metrics by Dataset", fontsize=15)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_xlabel("Dataset", fontsize=12)
        ax.legend(title="Metric")
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "metrics_comparison.png"))
        
        # 3. Latency Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="dataset", y="avg_latency", data=metrics_df)
        ax.set_title("Average Latency by Dataset", fontsize=15)
        ax.set_ylabel("Latency (seconds)", fontsize=12)
        ax.set_xlabel("Dataset", fontsize=12)
        
        # Add value labels on bars
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}s", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "latency_comparison.png"))
        
        print(f"Visualizations saved to {viz_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for the medical RAG system")
    
    # Tham số chung
    parser.add_argument("--datasets", type=str, default="hotpotqa", 
                        help="Comma-separated list of datasets to benchmark (hotpotqa)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate per dataset")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout per query in seconds")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run models (cuda or cpu)")
    
    # Tham số cho HotpotQA
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split to use (train, validation)")
    parser.add_argument("--filter_domain", type=str, default="medicine",
                        help="Filter by domain (comma-separated)")
    
    args = parser.parse_args()
    
    # Chuyển đổi danh sách datasets
    args.datasets = args.datasets.split(",")
    
    # Chạy tất cả benchmarks
    start_time = time.time()
    results = run_all_benchmarks(args)
    total_time = time.time() - start_time
    
    print(f"\\nTotal benchmark time: {total_time:.2f} seconds")
    
if __name__ == "__main__":
    main()
""",
        "benchmark/README.md": """
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
"""
    }
    
    # Tạo các file benchmark
    for file_path, content in files_to_create.items():
        path = Path(file_path)
        
        # Kiểm tra nếu file đã tồn tại và không ghi đè
        if path.exists() and not force_overwrite:
            print(f"File {file_path} đã tồn tại. Bỏ qua.")
            continue
            
        # Tạo thư mục cha nếu chưa tồn tại
        path.parent.mkdir(exist_ok=True, parents=True)
        
        # Ghi nội dung file
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.lstrip())
            
        print(f"Đã tạo file {file_path}")
    
    # Làm cho file benchmark.py có quyền thực thi
    benchmark_path = Path("benchmark/benchmark.py")
    if benchmark_path.exists():
        benchmark_path.chmod(0o755)
        
    print("\n=== Thiết lập benchmark hoàn tất ===")
    print("""
Để chạy benchmark:
1. cd benchmark
2. python benchmark.py --num_samples 50 --filter_domain medicine

Tham khảo thêm hướng dẫn trong QUICKSTART.md và benchmark/README.md
    """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup benchmark environment for RAG system")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing files")
    
    args = parser.parse_args()
    
    setup_benchmark_environment(force_overwrite=args.force) 